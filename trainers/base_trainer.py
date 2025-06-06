"""训练器基类"""
import os
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import load, save, count_parameters


class BaseTrainer(ABC):
    """训练器基类 - 定义通用的训练流程"""
    
    def __init__(self, config, device_manager, logger, tokenizer=None):
        self.config = config
        self.device_manager = device_manager
        self.logger = logger
        self.tokenizer = tokenizer
        
        # 组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None
        
        # 数据加载器
        self.train_loader = None
        self.test_loader = None
        self.train_sampler = None
        self.test_sampler = None
        
        # TensorBoard
        self.writer = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = -1e9
        self.last_epoch = -1
    
    @abstractmethod
    def build_model(self):
        """构建模型 - 子类必须实现"""
        pass
    
    def build_optimizer(self):
        """构建优化器 - 子类可以覆盖"""
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        adjusted_lr = self.device_manager.adjust_learning_rate(
            self.config.LEARNING_RATE
        ) if self.config.ADJUST_LR_FOR_MULTI_GPU else self.config.LEARNING_RATE
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=adjusted_lr,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        if self.device_manager.is_main_process():
            self.logger.info(f"优化器已创建 - LR: {adjusted_lr}")
    
    def build_scheduler(self):
        """构建学习率调度器"""
        from tools.optims import LinearWarmupCosineLRScheduler
        
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            self.config.EPOCHS,
            self.config.MIN_LR,
            self.config.LEARNING_RATE,
            decay_rate=None,
            warmup_start_lr=self.config.WARMUP_LR,
            warmup_steps=self.config.WARMUP_STEPS,
        )
        self.logger.info("学习率调度器已创建")
    
    def build_criterion(self):
        """构建损失函数 - 子类可以覆盖"""
        self.criterion = None
    
    def setup_tensorboard(self):
        """设置TensorBoard"""
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_log_dir = os.path.join(
            self.config.TENSORBOARD_DIR,
            f"{self.config.MODEL_NAME}_{self.config.PHASE}_{current_time}"
        )
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.logger.info(f"TensorBoard日志目录: {tensorboard_log_dir}")
    
    def setup_mixed_precision(self):
        """设置混合精度训练"""
        if self.config.USE_MIXED_PRECISION:
            self.scaler = torch.amp.GradScaler("cuda")
            self.logger.info("混合精度训练已启用")
        else:
            self.scaler = None
    
    def load_checkpoint(self):
        """加载检查点"""
        if self.config.CHECKPOINT_PATH_FROM and os.path.exists(self.config.CHECKPOINT_PATH_FROM):
            loaded_data = load(
                self.config.CHECKPOINT_PATH_FROM,
                self.model,
                self.optimizer,
                self.scheduler,
                load_model="full"
            )
            if isinstance(loaded_data, tuple):
                self.last_epoch, metrics = loaded_data
                if metrics is not None and isinstance(metrics, tuple):
                    self.best_metric, test_metric = metrics
                    self.logger.info(
                        f"从 {self.config.CHECKPOINT_PATH_FROM} 加载模型 - "
                        f"轮次: {self.last_epoch}, 最佳指标: {self.best_metric}"
                    )
                else:
                    self.logger.info(f"从 {self.config.CHECKPOINT_PATH_FROM} 加载模型")
            else:
                self.last_epoch = -1
                self.logger.info("检查点加载失败，从头开始训练")
    
    def save_checkpoint(self, epoch, metrics, filename):
        """保存检查点"""
        save_path = os.path.join(self.config.CHECKPOINT_PATH_TO, filename)
        save(save_path, self.model, self.optimizer, self.scheduler, epoch, metrics)
        self.logger.info(f"检查点已保存: {save_path}")
    
    def log_model_info(self):
        """记录模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.logger.info("=" * 60)
        self.logger.info("模型参数统计:")
        self.logger.info(f"  总参数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  冻结参数: {frozen_params:,}")
        self.logger.info("=" * 60)
    
    @abstractmethod
    def train_epoch(self, epoch):
        """训练一个epoch - 子类必须实现"""
        pass
    
    @abstractmethod
    def evaluate(self, data_loader, mode='test', epoch=None):
        """评估模型 - 子类必须实现"""
        pass
    
    @abstractmethod
    def get_main_metric(self, result):
        """获取主要评估指标 - 子类必须实现"""
        pass
    
    @abstractmethod
    def get_save_filename(self, epoch, result, is_best=False):
        """获取保存文件名 - 子类必须实现"""
        pass
    
    def should_evaluate(self, epoch):
        """判断是否应该在当前epoch进行评估"""
        eval_freq = getattr(self.config, 'EVAL_FREQ', 5)
        return (epoch + 1) % eval_freq == 0 or epoch == self.config.EPOCHS - 1
    
    def run(self):
        """完整的训练流程"""
        self.logger.info(f"开始训练阶段: {self.config.PHASE}")
        
        # 1. 构建模型
        self.logger.info("正在构建模型...")
        self.build_model()
        self.log_model_info()
        
        # 2. 包装模型（多GPU）
        self.model = self.device_manager.wrap_model(self.model)
        
        # 3. 构建优化器和调度器
        self.build_optimizer()
        self.build_scheduler()
        self.build_criterion()
        
        # 4. 设置混合精度和TensorBoard
        self.setup_mixed_precision()
        self.setup_tensorboard()
        
        # 5. 加载检查点
        self.load_checkpoint()
        
        # 6. 训练循环
        for epoch in range(self.last_epoch + 1, self.config.EPOCHS):
            self.current_epoch = epoch
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch}/{self.config.EPOCHS}")
            self.logger.info(f"{'='*60}")
            
            # 分布式训练时设置epoch
            if self.device_manager.distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            self.logger.info(f"训练损失: {train_loss:.4f}")
            
            # 评估
            if self.should_evaluate(epoch):
                test_loss, result = self.evaluate(self.test_loader, mode='test', epoch=epoch)
                self.logger.info(f"测试损失: {test_loss:.4f}")
                
                # 获取主要指标
                current_metric = self.get_main_metric(result)
                self.logger.info(f"当前指标: {current_metric:.4f}")
                
                # 保存检查点
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    self.logger.info(f"🎉 新的最佳模型! 指标: {self.best_metric:.4f}")
                
                filename = self.get_save_filename(epoch, result, is_best)
                self.save_checkpoint(epoch, (test_loss, result), filename)
            else:
                # 定期保存（不评估的epoch）
                filename = f"epoch_{epoch}.pth"
                self.save_checkpoint(epoch, None, filename)
            
            # 清理内存
            torch.cuda.empty_cache()
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("训练完成!")
        self.logger.info(f"最佳指标: {self.best_metric:.4f}")
        self.logger.info(f"{'='*60}")

