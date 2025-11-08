"""训练器基类"""
import os
import json
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import load, save, count_parameters, release_process_memory


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
    
    def get_raw_model(self):
        """
        获取原始模型（处理DDP/DataParallel包装）
        
        Returns:
            原始模型对象
        """
        return self.model.module if hasattr(self.model, 'module') else self.model
    
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
        
        # 从CHECKPOINT_PATH_TO中提取results/之后的路径作为前缀
        checkpoint_path = getattr(self.config, "CHECKPOINT_PATH_TO", "")
        path_suffix = ""
        if isinstance(checkpoint_path, str) and checkpoint_path:
            boundary = "results/"
            if boundary in checkpoint_path:
                path_suffix = checkpoint_path.split(boundary, 1)[1]
            else:
                path_suffix = checkpoint_path
            path_suffix = path_suffix.strip("/").replace("/", "_")
            if path_suffix:
                path_suffix = path_suffix + "_"
        
        tensorboard_log_dir = os.path.join(
            self.config.TENSORBOARD_DIR,
            f"{self.config.MODEL_NAME}_{self.config.PHASE}_{path_suffix}_{current_time}"
        )
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.logger.info(f"TensorBoard日志目录: {tensorboard_log_dir}")
        
        # 保存配置信息到结果目录
        self._save_config_to_dir(self.config.CHECKPOINT_PATH_TO)
    
    def _save_config_to_dir(self, output_dir):
        """
        将配置信息保存到指定目录
        
        Args:
            output_dir: 输出目录路径
        """
        try:
            # 确保目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 提取配置信息
            config_dict = {}
            for attr in dir(self.config):
                # 跳过私有属性和方法
                if not attr.startswith('_') and not callable(getattr(self.config, attr)):
                    value = getattr(self.config, attr)
                    # 确保值可以序列化为JSON
                    try:
                        json.dumps(value)
                        config_dict[attr] = value
                    except (TypeError, ValueError):
                        # 不可序列化的对象转换为字符串
                        config_dict[attr] = str(value)
            
            # 保存为JSON文件
            config_file = os.path.join(output_dir, "config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置信息已保存到: {config_file}")
            
        except Exception as e:
            self.logger.warning(f"保存配置信息时出错: {e}")
    
    def setup_mixed_precision(self):
        """设置混合精度训练"""
        if self.config.USE_MIXED_PRECISION:
            self.scaler = torch.cuda.amp.GradScaler()
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
        """记录模型信息 - 详细的组件级别统计"""
        self.logger.info("=" * 80)
        self.logger.info("模型组件参数统计:")
        self.logger.info("-" * 80)
        
        total_params = 0
        trainable_params = 0
        
        # 获取原始模型（处理DDP包装）
        model = self.get_raw_model()
        
        # 统计各个组件
        components = []
        
        # 1. 目标检测器
        if hasattr(model, 'object_detector') and model.object_detector is not None:
            detector_total = sum(p.numel() for p in model.object_detector.parameters())
            detector_trainable = sum(p.numel() for p in model.object_detector.parameters() if p.requires_grad)
            components.append(("目标检测器 (Object Detector)", detector_total, detector_trainable))
            
            # 检测器子组件
            if hasattr(model.object_detector, 'detector'):
                det_total = sum(p.numel() for p in model.object_detector.detector.parameters())
                det_trainable = sum(p.numel() for p in model.object_detector.detector.parameters() if p.requires_grad)
                components.append(("  └─ Faster R-CNN", det_total, det_trainable))
            
            if hasattr(model.object_detector, 'feature_projector'):
                fp_total = sum(p.numel() for p in model.object_detector.feature_projector.parameters())
                fp_trainable = sum(p.numel() for p in model.object_detector.feature_projector.parameters() if p.requires_grad)
                components.append(("  └─ 特征投影层", fp_total, fp_trainable))
            
            if hasattr(model.object_detector, 'missing_region_tokens'):
                mrt_total = model.object_detector.missing_region_tokens.numel()
                mrt_trainable = mrt_total if model.object_detector.missing_region_tokens.requires_grad else 0
                components.append(("  └─ 缺失区域Token", mrt_total, mrt_trainable))
        
        # 2. 图像编码器 (ViT)
        if hasattr(model, 'image_encoder') and model.image_encoder is not None:
            vit_total = sum(p.numel() for p in model.image_encoder.parameters())
            vit_trainable = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
            components.append(("图像编码器 (ViT)", vit_total, vit_trainable))
        
        # 3. 解码器
        if hasattr(model, 'findings_decoder') and model.findings_decoder is not None:
            decoder_total = sum(p.numel() for p in model.findings_decoder.parameters())
            decoder_trainable = sum(p.numel() for p in model.findings_decoder.parameters() if p.requires_grad)
            components.append(("报告解码器 (Decoder)", decoder_total, decoder_trainable))
        
        # 4. RGAT模块
        if hasattr(model, 'rgat') and model.rgat is not None:
            rgat_total = sum(p.numel() for p in model.rgat.parameters())
            rgat_trainable = sum(p.numel() for p in model.rgat.parameters() if p.requires_grad)
            components.append(("RGAT图推理模块", rgat_total, rgat_trainable))
            
            # RGAT子组件
            if hasattr(model.rgat, 'disease_embeddings'):
                de_total = model.rgat.disease_embeddings.numel()
                de_trainable = de_total if model.rgat.disease_embeddings.requires_grad else 0
                components.append(("  └─ 疾病嵌入", de_total, de_trainable))
            
            if hasattr(model.rgat, 'stage1_aa'):
                s1_total = sum(p.numel() for p in model.rgat.stage1_aa.parameters())
                s1_trainable = sum(p.numel() for p in model.rgat.stage1_aa.parameters() if p.requires_grad)
                components.append(("  └─ 阶段1 (A→A)", s1_total, s1_trainable))
            
            if hasattr(model.rgat, 'stage2_ad'):
                s2_total = sum(p.numel() for p in model.rgat.stage2_ad.parameters())
                s2_trainable = sum(p.numel() for p in model.rgat.stage2_ad.parameters() if p.requires_grad)
                components.append(("  └─ 阶段2 (A→D)", s2_total, s2_trainable))
            
            if hasattr(model.rgat, 'stage3_dd'):
                s3_total = sum(p.numel() for p in model.rgat.stage3_dd.parameters())
                s3_trainable = sum(p.numel() for p in model.rgat.stage3_dd.parameters() if p.requires_grad)
                components.append(("  └─ 阶段3 (D→D)", s3_total, s3_trainable))
            
            if hasattr(model.rgat, 'disease_classifier'):
                dc_total = sum(p.numel() for p in model.rgat.disease_classifier.parameters())
                dc_trainable = sum(p.numel() for p in model.rgat.disease_classifier.parameters() if p.requires_grad)
                components.append(("  └─ 疾病分类器", dc_total, dc_trainable))
        
        # 5. CXR-BERT (如果存在)
        if hasattr(model, 'cxr_bert') and model.cxr_bert is not None:
            cxr_total = sum(p.numel() for p in model.cxr_bert.parameters())
            cxr_trainable = sum(p.numel() for p in model.cxr_bert.parameters() if p.requires_grad)
            components.append(("CXR-BERT", cxr_total, cxr_trainable))
        
        # 6. 投影层
        projection_layers = []
        if hasattr(model, 'visual_projection'):
            projection_layers.append(('visual_projection', model.visual_projection))
        if hasattr(model, 'text_projection'):
            projection_layers.append(('text_projection', model.text_projection))
        if hasattr(model, 'region_visual_projection'):
            projection_layers.append(('region_visual_projection', model.region_visual_projection))
        if hasattr(model, 'region_text_projection'):
            projection_layers.append(('region_text_projection', model.region_text_projection))
        
        if projection_layers:
            proj_total = sum(sum(p.numel() for p in layer.parameters()) for _, layer in projection_layers)
            proj_trainable = sum(sum(p.numel() for p in layer.parameters() if p.requires_grad) for _, layer in projection_layers)
            components.append(("投影层", proj_total, proj_trainable))
            
            for name, layer in projection_layers:
                layer_total = sum(p.numel() for p in layer.parameters())
                layer_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                components.append((f"  └─ {name}", layer_total, layer_trainable))
        
        # 打印组件统计
        for name, total, trainable in components:
            frozen = total - trainable
            trainable_pct = (trainable / total * 100) if total > 0 else 0
            if trainable == total:
                status = "✓ 全部可训练"
            elif trainable == 0:
                status = "✗ 全部冻结"
            else:
                status = f"◐ {trainable_pct:.1f}% 可训练"
            
            self.logger.info(f"{name:40s} | 总参数: {total:>12,} | 可训练: {trainable:>12,} | {status}")
        
        # 总计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.logger.info("-" * 80)
        self.logger.info(f"{'总计':40s} | 总参数: {total_params:>12,} | 可训练: {trainable_params:>12,} | 冻结: {frozen_params:>12,}")
        self.logger.info(f"{'可训练比例':40s} | {trainable_params / total_params * 100:.2f}%")
        self.logger.info("=" * 80)
    
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
            
            # 【修复】训练后立即清理内存
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            # 进一步归还进程空闲内存
            release_process_memory()
            
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
                
                # 【修复】评估后也清理内存
                del test_loss, result
                release_process_memory()
            else:
                # 定期保存（不评估的epoch）
                filename = f"epoch_{epoch}.pth"
                self.save_checkpoint(epoch, None, filename)
            
            # 【修复】每个epoch结束后彻底清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            release_process_memory()
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("训练完成!")
        self.logger.info(f"最佳指标: {self.best_metric:.4f}")
        self.logger.info(f"{'='*60}")

