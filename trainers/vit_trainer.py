"""ViT预训练阶段的训练器"""
import torch
from trainers.base_trainer import BaseTrainer
from models.medical_report_generator import MedicalReportGenerator
from models.fast_rcnn_classifier import DetectionOnlyFastRCNN, EnhancedFastRCNN
from models.vit import MedicalVisionTransformer
from models.cxr_bert import CXR_BERT_FeatureExtractor
from utils import load, train, test_vit
from datasets import MIMIC


class ViTPretrainTrainer(BaseTrainer):
    """ViT预训练阶段的训练器"""
    
    def __init__(self, config, device_manager, logger, tokenizer=None):
        super().__init__(config, device_manager, logger, tokenizer)
        self.phase_name = "PRETRAIN_VIT"
    
    def build_model(self):
        """构建ViT预训练模型"""
        self.logger.info("构建ViT预训练模型...")
        
        # 1. 加载检测器
        self.logger.info("加载目标检测器...")
        detection_model = DetectionOnlyFastRCNN()
        _, _ = load(self.config.DETECTION_CHECKPOINT_PATH_FROM, detection_model, device=self.device_manager.device, load_model="full")
        
        # 2. 创建增强型FastRCNN
        enhanced_rcnn = EnhancedFastRCNN(
            pretrained_detector=detection_model,
            num_regions=29,
            feature_dim=768
        )
        
        # 3. 初始化ViT
        self.logger.info("初始化Vision Transformer...")
        vit_model = MedicalVisionTransformer()
        
        # 4. 初始化CXR-BERT
        self.logger.info("初始化CXR-BERT特征提取器...")
        cxr_bert = CXR_BERT_FeatureExtractor()
        
        # 5. 组装医学报告生成模型
        self.model = MedicalReportGenerator(
            config=self.config,
            object_detector=enhanced_rcnn,
            image_encoder=vit_model,
            cxr_bert=cxr_bert
        )
        
        # 6. 加载解剖区域数据库
        if getattr(self.config, 'ENABLE_REGION_ITC', True):
            anatomical_db_path = getattr(self.config, 'ANATOMICAL_DATABASE_PATH', None)
            if anatomical_db_path:
                MIMIC.load_anatomical_embeddings(anatomical_db_path)
                self.logger.info("✅ 解剖区域数据库已加载")
        
        self.logger.info("ViT预训练模型构建完成")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        train_loss = train(
            self.config,
            self.train_loader,
            self.model,
            self.optimizer,
            self.criterion,
            self.config.EPOCHS,
            epoch,
            scheduler=self.scheduler,
            device=self.device_manager.device,
            kw_src=self.config.KW_SRC,
            kw_tgt=self.config.KW_TGT,
            scaler=self.scaler,
            writer=self.writer,
            device_manager=self.device_manager
        )
        return train_loss
    
    def evaluate(self, data_loader, mode='test', epoch=None):
        """评估模型"""
        test_loss, result = test_vit(
            config=self.config,
            model=self.model,
            data_loader=data_loader,
            logger=self.logger,
            mode=mode,
            epoch=epoch if epoch is not None else self.current_epoch,
            writer=self.writer
        )
        return test_loss, result
    
    def get_main_metric(self, result):
        """获取主要评估指标"""
        return result["overall_metrics"]["ce_f1"]
    
    def get_save_filename(self, epoch, result, is_best=False):
        """获取保存文件名"""
        if is_best:
            prefix = "best_"
        else:
            prefix = f"epoch_{epoch}_"
        
        ce_f1 = result["overall_metrics"]["ce_f1"]
        return f"{prefix}ce_f1_{ce_f1:.4f}.pth"

