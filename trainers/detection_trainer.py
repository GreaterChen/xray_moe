"""目标检测阶段的训练器"""
import torch
from trainers.base_trainer import BaseTrainer
from models.fast_rcnn_classifier import DetectionOnlyFastRCNN
from utils import load, train, test_detection


class DetectionTrainer(BaseTrainer):
    """目标检测阶段的训练器"""
    
    def __init__(self, config, device_manager, logger, tokenizer=None):
        super().__init__(config, device_manager, logger, tokenizer)
        self.phase_name = "TRAIN_DETECTION"
    
    def build_model(self):
        """构建目标检测模型"""
        self.logger.info("构建目标检测模型...")
        
        # 创建目标检测器
        self.model = DetectionOnlyFastRCNN()
        
        # 如果有预训练权重，加载它
        if hasattr(self.config, 'DETECTION_CHECKPOINT_PATH_FROM') and \
           self.config.DETECTION_CHECKPOINT_PATH_FROM:
            self.logger.info(f"加载预训练检测器: {self.config.DETECTION_CHECKPOINT_PATH_FROM}")
            _, _ = load(self.config.DETECTION_CHECKPOINT_PATH_FROM, self.model)
        
        self.logger.info("目标检测模型构建完成")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        train_loss = train(
            self.config,
            self.train_loader,
            self.model,
            self.optimizer,
            None,  # criterion (detection使用内置损失)
            self.config.EPOCHS,
            epoch,
            scheduler=self.scheduler,
            device=self.device_manager.device,
            kw_src=self.config.KW_SRC if hasattr(self.config, 'KW_SRC') else None,
            kw_tgt=self.config.KW_TGT if hasattr(self.config, 'KW_TGT') else None,
            scaler=self.scaler,
            writer=self.writer,
            device_manager=self.device_manager
        )
        return train_loss
    
    def evaluate(self, data_loader, mode='test', epoch=None):
        """评估模型"""
        test_loss, result = test_detection(
            config=self.config,
            model=self.model,
            data_loader=data_loader,
            logger=self.logger,
            mode=mode,
            epoch=epoch if epoch is not None else self.current_epoch,
            writer=self.writer,
            device=self.device_manager.device
        )
        return test_loss, result
    
    def get_main_metric(self, result):
        """获取主要评估指标"""
        # 目标检测通常使用 mAP 作为主要指标
        return result.get("mAP", result.get("detection_acc", 0.0))
    
    def get_save_filename(self, epoch, result, is_best=False):
        """获取保存文件名"""
        if is_best:
            prefix = "best_"
        else:
            prefix = f"epoch_{epoch}_"
        
        main_metric = self.get_main_metric(result)
        return f"{prefix}detection_metric_{main_metric:.4f}.pth"

