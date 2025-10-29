"""BERT微调阶段的训练器"""
import torch
from trainers.base_trainer import BaseTrainer
from models.medical_report_generator import MedicalReportGenerator
from models.bert_adapter import BertAdapter
from models.model_builder import build_detection_model, build_vit_model, freeze_model_parameters
from utils import train, test_llm, load
from metrics import compute_scores


class BertFinetuneTrainer(BaseTrainer):
    """BERT微调阶段的训练器"""
    
    def __init__(self, config, device_manager, logger, tokenizer):
        super().__init__(config, device_manager, logger, tokenizer)
        self.phase_name = "FINETUNE_BERT"
        self.chexbert_metrics = None
    
    def build_model(self):
        """构建BERT微调模型"""
        self.logger.info("构建BERT微调模型...")
        
        # 1. 使用公共函数构建检测器
        enhanced_rcnn = build_detection_model(self.config, self.logger, device=self.device_manager.device)
        
        # 2. 使用公共函数构建ViT
        vit_model = build_vit_model(
            self.config,
            load_pretrained=False,  # BERT微调阶段不预加载ViT权重
            logger=self.logger,
            device=self.device_manager.device
        )
        
        # 3. 创建BERT解码器
        self.logger.info("初始化BERT解码器...")
        bert_model = BertAdapter(
            config=self.config,
            tokenizer=self.tokenizer,
            hidden_dim=768,
            max_length=100
        )
        
        # 4. 组装医学报告生成模型
        self.model = MedicalReportGenerator(
            config=self.config,
            object_detector=enhanced_rcnn,
            image_encoder=vit_model,
            findings_decoder=bert_model
        )
        
        # 5. 冻结检测器参数
        freeze_model_parameters(
            self.model.object_detector,
            self.logger,
            "目标检测器"
        )
        
        self.logger.info("✅ BERT微调模型构建完成")
    
    def build_optimizer(self):
        """构建优化器 - 只优化特定参数"""
        trainable_params = []
        # 添加解码器的所有可训练参数
        trainable_params.extend([
            p for p in self.model.findings_decoder.parameters() if p.requires_grad
        ])
        # 添加ViT中可训练的参数（LoRA等）
        trainable_params.extend([
            p for p in self.model.image_encoder.parameters() if p.requires_grad
        ])
        
        adjusted_lr = self.device_manager.adjust_learning_rate(
            self.config.LEARNING_RATE
        ) if self.config.ADJUST_LR_FOR_MULTI_GPU else self.config.LEARNING_RATE
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=adjusted_lr,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        trainable_count = sum(p.numel() for p in trainable_params)
        self.logger.info(f"优化器已创建 - LR: {adjusted_lr}, 可训练参数: {trainable_count:,}")
    
    def build_criterion(self):
        """构建损失函数和评估器"""
        self.criterion = None
        
        # 初始化CheXbert评估器
        try:
            from tools.metrics_clinical import CheXbertMetrics
            self.chexbert_metrics = CheXbertMetrics(
                checkpoint_path=self.config.CHEXBERT_CHECKPOINT_PATH,
                mbatch_size=self.config.VAL_BATCH_SIZE,
                device=str(self.device_manager.device)
            )
            self.logger.info("✅ CheXbert评估器初始化成功")
        except Exception as e:
            self.logger.warning(f"⚠️  CheXbert评估器初始化失败: {e}")
            self.chexbert_metrics = None
    
    def load_checkpoint(self):
        """加载检查点 - BERT特殊处理"""
        # 如果有专门的decoder checkpoint路径
        if hasattr(self.config, 'DECODER_CHECKPOINT_PATH_FROM') and \
           self.config.DECODER_CHECKPOINT_PATH_FROM:
            _, _ = load(
                self.config.DECODER_CHECKPOINT_PATH_FROM,
                self.model.findings_decoder.decoder,
                self.optimizer,
                self.scheduler,
                load_model="decoder",
                device=self.device_manager.device
            )
            self.logger.info(f"从 {self.config.DECODER_CHECKPOINT_PATH_FROM} 加载解码器权重")
        else:
            # 使用标准加载方式
            super().load_checkpoint()
    
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
        test_loss, result = test_llm(
            config=self.config,
            data_loader=data_loader,
            model=self.model,
            logger=self.logger,
            metric_ftns=compute_scores,
            mode=mode,
            device=self.device_manager.device,
            epoch=epoch if epoch is not None else self.current_epoch,
            writer=self.writer,
            chexbert_metrics=self.chexbert_metrics
        )
        return test_loss, result
    
    def get_main_metric(self, result):
        """获取主要评估指标"""
        # 优先使用CheXbert的ce_f1，否则使用BLEU-1
        if "chexbert_metrics" in result and "ce_f1" in result["chexbert_metrics"]:
            return result["chexbert_metrics"]["ce_f1"]
        elif "report_generation_metrics" in result:
            return result["report_generation_metrics"]["BLEU_1"]
        return 0.0
    
    def get_save_filename(self, epoch, result, is_best=False):
        """获取保存文件名"""
        prefix = "best_" if is_best else f"epoch_{epoch}_"
        
        # 构建文件名
        bleu1 = result["report_generation_metrics"]["BLEU_1"]
        filename = f"{prefix}bleu_{bleu1:.4f}"
        
        # 如果有CheXbert指标，也加上
        if "chexbert_metrics" in result and "ce_f1" in result["chexbert_metrics"]:
            ce_f1 = result["chexbert_metrics"]["ce_f1"]
            filename += f"_ce_f1_{ce_f1:.4f}"
        
        return filename + ".pth"

