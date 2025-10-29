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
            _, _ = load(self.config.DETECTION_CHECKPOINT_PATH_FROM, self.model, device=self.device_manager.device)
        
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
        """评估模型 - 全面的评估系统"""
        confidence_threshold = getattr(self.config, 'DETECTION_CONFIDENCE_THRESHOLD', 0.5)
        test_loss, result = test_detection(
            config=self.config,
            model=self.model,
            data_loader=data_loader,
            logger=self.logger,
            mode=mode,
            confidence_threshold=confidence_threshold,
            device=self.device_manager.device,
            epoch=epoch if epoch is not None else self.current_epoch,
            writer=self.writer
        )
        return test_loss, result
    
    def get_main_metric(self, result):
        """获取主要评估指标"""
        # 使用mAP@0.5作为主要指标
        return result.get("mAP@0.5", result.get("mAP", 0.0))
    
    def get_save_filename(self, epoch, result, is_best=False):
        """获取保存文件名"""
        if is_best:
            prefix = "best_"
        else:
            prefix = f"epoch_{epoch}_"
        
        mAP_05 = result.get("mAP@0.5", 0.0)
        mAP_all = result.get("mAP", 0.0)
        return f"{prefix}mAP05_{mAP_05:.4f}_mAP_{mAP_all:.4f}.pth"

