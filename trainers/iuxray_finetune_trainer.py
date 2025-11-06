"""IU_XRAY端到端微调阶段的训练器"""
import torch
from trainers.base_trainer import BaseTrainer
from models.medical_report_generator import MedicalReportGenerator
from models.bert_adapter import BertAdapter
from models.model_builder import build_detection_model, build_vit_model, freeze_model_parameters
from utils import train, test_llm, load
from metrics import compute_scores


class IUXRAYFinetuneTrainer(BaseTrainer):
    """IU_XRAY端到端微调阶段的训练器
    
    在MIMIC数据集上训练完成后，在IU_XRAY数据集上进行端到端微调和测试。
    所有组件（检测器、ViT、解码器、RGAT）都保持可训练。
    """
    
    def __init__(self, config, device_manager, logger, tokenizer):
        super().__init__(config, device_manager, logger, tokenizer)
        self.phase_name = "FINETUNE_IUXRAY"
        self.chexbert_metrics = None
    
    def build_model(self):
        """构建IU_XRAY微调模型 - 加载MIMIC训练好的模型"""
        self.logger.info("构建IU_XRAY端到端微调模型...")
        
        # 1. 使用公共函数构建检测器（加载预训练权重）
        enhanced_rcnn = build_detection_model(self.config, self.logger, device=self.device_manager.device)
        
        # 2. 使用公共函数构建ViT（加载预训练权重）
        vit_model = build_vit_model(
            self.config,
            load_pretrained=True,
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
        
        # 5. 冻结策略：所有组件保持可训练（端到端微调）
        # 检测器的检测部分在build_detection_model时已经冻结
        # 其他部分保持可训练
        
        # 统计可训练参数
        detector_trainable = sum(p.numel() for p in self.model.object_detector.parameters() if p.requires_grad)
        vit_trainable = sum(p.numel() for p in self.model.image_encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.model.findings_decoder.parameters() if p.requires_grad)
        
        self.logger.info(f"✅ IU_XRAY微调模型构建完成")
        self.logger.info(f"  - 检测器可训练参数: {detector_trainable:,}")
        self.logger.info(f"  - ViT可训练参数: {vit_trainable:,}")
        self.logger.info(f"  - 解码器可训练参数: {decoder_trainable:,}")
    
    def build_optimizer(self):
        """构建优化器 - 优化所有可训练组件"""
        # 获取原始模型（处理DDP包装）
        model = self.get_raw_model()
        
        trainable_params = []
        
        # 1. 添加检测器的可训练参数
        detector_params = [p for p in model.object_detector.parameters() if p.requires_grad]
        trainable_params.extend(detector_params)
        
        # 2. 添加ViT的所有可训练参数
        vit_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
        trainable_params.extend(vit_params)
        
        # 3. 添加解码器的所有可训练参数
        decoder_params = [p for p in model.findings_decoder.parameters() if p.requires_grad]
        trainable_params.extend(decoder_params)
        
        # 4. 添加RGAT模块的所有可训练参数
        rgat_params = []
        if hasattr(model, 'rgat') and model.rgat is not None:
            rgat_params = [p for p in model.rgat.parameters() if p.requires_grad]
            trainable_params.extend(rgat_params)
        
        # 使用较小的学习率进行微调
        adjusted_lr = self.device_manager.adjust_learning_rate(
            self.config.LEARNING_RATE
        ) if self.config.ADJUST_LR_FOR_MULTI_GPU else self.config.LEARNING_RATE
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=adjusted_lr,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        if self.device_manager.is_main_process():
            self.logger.info(f"✅ 优化器已创建")
            self.logger.info(f"  - 学习率: {adjusted_lr}")
            self.logger.info(f"  - 检测器参数: {len(detector_params):,}")
            self.logger.info(f"  - ViT参数: {len(vit_params):,}")
            self.logger.info(f"  - 解码器参数: {len(decoder_params):,}")
            self.logger.info(f"  - RGAT参数: {len(rgat_params):,}")
            self.logger.info(f"  - 总可训练参数: {len(trainable_params):,}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        # 使用utils中的train函数
        train_loss = train(
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            criterion=self.criterion,
            scheduler=self.scheduler,
            epoch=epoch,
            device_manager=self.device_manager,
            writer=self.writer,
            config=self.config,
            max_len=self.config.MAX_LEN_FINDINGS,
            tokenizer=self.tokenizer
        )
        return train_loss
    
    def evaluate(self, data_loader, mode='test', epoch=None):
        """评估模型"""
        # 导入CheXbert评估器（延迟导入避免循环依赖）
        if self.chexbert_metrics is None and self.config.CHEXBERT_CHECKPOINT_PATH:
            try:
                from metrics import CheXbertMetrics
                self.chexbert_metrics = CheXbertMetrics(
                    bert_path=self.config.BERT_PRETRAINED_PATH,
                    checkpoint_path=self.config.CHEXBERT_CHECKPOINT_PATH,
                    device=self.device_manager.device
                )
                self.logger.info("✅ CheXbert评估器已初始化")
            except Exception as e:
                self.logger.warning(f"⚠️  无法初始化CheXbert评估器: {e}")
                self.chexbert_metrics = None
        
        # 使用utils中的test_llm函数
        test_loss, result = test_llm(
            model=self.model,
            data_loader=data_loader,
            criterion=self.criterion,
            device_manager=self.device_manager,
            writer=self.writer,
            epoch=epoch,
            config=self.config,
            max_len=self.config.MAX_LEN_FINDINGS,
            tokenizer=self.tokenizer,
            chexbert_metrics=self.chexbert_metrics,
            mode=mode
        )
        
        # 打印评估结果
        if self.device_manager.is_main_process():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"{mode.upper()} 评估结果 (Epoch {epoch}):")
            self.logger.info(f"{'='*60}")
            for metric_name, value in result.items():
                if isinstance(value, float):
                    self.logger.info(f"{metric_name:30s}: {value:.4f}")
            self.logger.info(f"{'='*60}\n")
        
        return test_loss, result
    
    def get_main_metric(self, result):
        """获取主要评估指标 - 使用BLEU-4作为主要指标"""
        return result.get('BLEU_4', 0.0)
    
    def get_save_filename(self, epoch, result, is_best=False):
        """获取保存文件名"""
        if is_best:
            return f"best_iuxray_finetune_epoch_{epoch}.pth"
        else:
            return f"iuxray_finetune_epoch_{epoch}.pth"

