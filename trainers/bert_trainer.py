"""BERT微调阶段的训练器（支持多种decoder）"""
import torch
from trainers.base_trainer import BaseTrainer
from models.medical_report_generator import MedicalReportGenerator
from models.bert_adapter import BertAdapter
from models.qwenvl_decoder import QwenVLAdapter
from models.model_builder import build_detection_model, build_vit_model, freeze_model_parameters, build_image_encoder
from utils import train, test_llm, load
from metrics import compute_scores


class BertFinetuneTrainer(BaseTrainer):
    """BERT微调阶段的训练器"""
    
    def __init__(self, config, device_manager, logger, tokenizer):
        super().__init__(config, device_manager, logger, tokenizer)
        self.phase_name = "FINETUNE_BERT"
        self.chexbert_metrics = None
    
    def build_model(self):
        """构建微调模型（支持BERT和Qwen2.5-VL decoder）"""
        decoder_type = getattr(self.config, 'DECODER_TYPE', 'bert').lower()
        self.logger.info(f"构建微调模型 (decoder类型: {decoder_type})...")
        
        # 1. 使用公共函数构建检测器
        enhanced_rcnn = build_detection_model(self.config, self.logger, device=self.device_manager.device)
        
        # 2. 构建图像编码器（支持多类型）
        image_encoder = build_image_encoder(
            self.config,
            logger=self.logger,
            device=self.device_manager.device
        )
        
        # 3. 根据配置创建解码器
        if decoder_type in ['qwen2vl', 'qwenvl']:
            self.logger.info("初始化Qwen VL解码器...")
            qwen_model_name = getattr(self.config, 'QWEN_MODEL_NAME', 'Qwen/Qwen3-VL-4B-Instruct')
            decoder_model = QwenVLAdapter(
                config=self.config,
                tokenizer=self.tokenizer,
                hidden_dim=768,
                max_length=196,
                qwen_model_name=qwen_model_name
            )
            self.logger.info(f"✅ Qwen VL解码器初始化完成 (模型: {qwen_model_name})")
        else:
            # 默认使用BERT解码器
            self.logger.info("初始化BERT解码器...")
            decoder_model = BertAdapter(
                config=self.config,
                tokenizer=self.tokenizer,
                hidden_dim=768,
                max_length=100
            )
            self.logger.info("✅ BERT解码器初始化完成")
        
        # 4. 组装医学报告生成模型
        self.model = MedicalReportGenerator(
            config=self.config,
            object_detector=enhanced_rcnn,
            image_encoder=image_encoder,
            findings_decoder=decoder_model
        )
        
        # 5. 冻结策略：
        # - 检测器的检测部分已在 EnhancedFastRCNN 初始化时冻结
        # - 特征提取部分（feature_projector, missing_region_tokens）保持可训练
        # - ViT 保持可训练
        # 因此不需要额外的冻结操作
        
        # 统计可训练参数
        detector_trainable = sum(p.numel() for p in self.model.object_detector.parameters() if p.requires_grad)
        vit_trainable = sum(p.numel() for p in self.model.image_encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.model.findings_decoder.parameters() if p.requires_grad)
        
        self.logger.info(f"✅ {decoder_type.upper()}微调模型构建完成")
        self.logger.info(f"  - 检测器可训练参数: {detector_trainable:,} (仅特征提取层)")
        self.logger.info(f"  - ViT可训练参数: {vit_trainable:,}")
        self.logger.info(f"  - 解码器可训练参数: {decoder_trainable:,}")
    
    def build_optimizer(self):
        """构建优化器 - 支持分层学习率（LoRA、ViT、其他模块使用不同学习率）"""
        # 获取原始模型（处理DDP包装）
        model = self.get_raw_model()

        # 获取配置
        decoder_type = getattr(self.config, 'DECODER_TYPE', 'bert').lower()
        use_lora = getattr(self.config, 'USE_LORA', False)
        use_layerwise_lr = getattr(self.config, 'USE_LAYERWISE_LR', False)

        # 基础学习率（考虑多GPU调整）
        base_lr = self.device_manager.adjust_learning_rate(
            self.config.LEARNING_RATE
        ) if self.config.ADJUST_LR_FOR_MULTI_GPU else self.config.LEARNING_RATE

        # 如果启用分层学习率，则按模块分组
        if use_layerwise_lr:
            self.logger.info("🔧 使用分层学习率构建优化器...")

            # 获取学习率倍数
            bert_scale = getattr(self.config, 'BERT_LR_SCALE', 0.1)  # BERT参数使用更低的学习率
            lora_scale = getattr(self.config, 'LORA_LR_SCALE', 1.0)
            vit_scale = getattr(self.config, 'VIT_LR_SCALE', 0.5)
            other_scale = getattr(self.config, 'OTHER_LR_SCALE', 1.0)

            # 计算各组学习率
            bert_lr = base_lr * bert_scale
            lora_lr = base_lr * lora_scale
            vit_lr = base_lr * vit_scale
            other_lr = base_lr * other_scale

            # 参数分组
            bert_params = []
            lora_params = []
            decoder_non_lora_params = []
            vit_params = []
            other_params = []

            # 1. 解码器参数：区分 BERT、LoRA 和 其他
            for name, param in model.findings_decoder.named_parameters():
                if param.requires_grad:
                    # BERT解码器的text_decoder参数
                    if 'text_decoder' in name or 'bert' in name.lower():
                        bert_params.append(param)
                    # LoRA 参数通常名字包含 'lora'
                    elif 'lora' in name.lower():
                        lora_params.append(param)
                    else:
                        decoder_non_lora_params.append(param)

            # 2. ViT 参数
            vit_params = [p for p in model.image_encoder.parameters() if p.requires_grad]

            # 3. 其他参数（检测器特征提取层、RGAT、投影层等）
            for name, module in model.named_children():
                if name not in ['findings_decoder', 'image_encoder']:
                    other_params.extend([p for p in module.parameters() if p.requires_grad])

            # 创建参数分组
            param_groups = []

            if bert_params:
                param_groups.append({
                    'params': bert_params,
                    'lr': bert_lr,
                    'name': 'bert'
                })

            if lora_params:
                param_groups.append({
                    'params': lora_params,
                    'lr': lora_lr,
                    'name': 'lora'
                })

            if vit_params:
                param_groups.append({
                    'params': vit_params,
                    'lr': vit_lr,
                    'name': 'vit'
                })

            # 合并 decoder 非 LoRA/BERT 参数和其他参数
            combined_other_params = decoder_non_lora_params + other_params
            if combined_other_params:
                param_groups.append({
                    'params': combined_other_params,
                    'lr': other_lr,
                    'name': 'other'
                })

            # 创建优化器
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.WEIGHT_DECAY
            )

            # 统计和日志
            bert_count = sum(p.numel() for p in bert_params)
            lora_count = sum(p.numel() for p in lora_params)
            vit_count = sum(p.numel() for p in vit_params)
            other_count = sum(p.numel() for p in combined_other_params)
            total_count = bert_count + lora_count + vit_count + other_count

            self.logger.info("✅ 分层学习率优化器已创建:")
            self.logger.info(f"  📊 参数分组统计:")
            if bert_params:
                self.logger.info(f"    • BERT 参数: {bert_count:,} ({bert_count/total_count*100:.1f}%) - LR={bert_lr:.2e}")
            if lora_params:
                self.logger.info(f"    • LoRA 参数: {lora_count:,} ({lora_count/total_count*100:.1f}%) - LR={lora_lr:.2e}")
            if vit_params:
                self.logger.info(f"    • ViT 参数: {vit_count:,} ({vit_count/total_count*100:.1f}%) - LR={vit_lr:.2e}")
            if combined_other_params:
                self.logger.info(f"    • 其他参数: {other_count:,} ({other_count/total_count*100:.1f}%) - LR={other_lr:.2e}")
            self.logger.info(f"  📈 总计: {total_count:,} 可训练参数")
            self.logger.info(f"  ⚙️  Weight Decay: {self.config.WEIGHT_DECAY}")

        else:
            # 不使用分层学习率：所有参数统一学习率
            self.logger.info("🔧 使用统一学习率构建优化器...")

            trainable_params = []

            # 1. 检测器特征提取层
            detector_params = [p for p in model.object_detector.parameters() if p.requires_grad]
            trainable_params.extend(detector_params)

            # 2. ViT
            vit_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
            trainable_params.extend(vit_params)

            # 3. 解码器
            decoder_params = [p for p in model.findings_decoder.parameters() if p.requires_grad]
            trainable_params.extend(decoder_params)

            # 4. RGAT
            rgat_params = []
            if hasattr(model, 'rgat') and model.rgat is not None:
                rgat_params = [p for p in model.rgat.parameters() if p.requires_grad]
                trainable_params.extend(rgat_params)

            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=base_lr,
                weight_decay=self.config.WEIGHT_DECAY
            )

            # 统计
            detector_count = sum(p.numel() for p in detector_params)
            vit_count = sum(p.numel() for p in vit_params)
            decoder_count = sum(p.numel() for p in decoder_params)
            rgat_count = sum(p.numel() for p in rgat_params)
            total_count = detector_count + vit_count + decoder_count + rgat_count

            self.logger.info(f"✅ 优化器已创建 - LR: {base_lr:.2e}")
            self.logger.info(f"  - 检测器特征提取层: {detector_count:,} 参数")
            self.logger.info(f"  - ViT: {vit_count:,} 参数")
            self.logger.info(f"  - 解码器: {decoder_count:,} 参数")
            self.logger.info(f"  - RGAT: {rgat_count:,} 参数")
            self.logger.info(f"  - 总计: {total_count:,} 可训练参数")
        
    def build_criterion(self):
        """构建损失函数和评估器"""
        self.criterion = None
        
        # 初始化CheXbert评估器
        try:
            from tools.metrics_clinical import CheXbertMetrics
            self.chexbert_metrics = CheXbertMetrics(
                checkpoint_path=self.config.CHEXBERT_CHECKPOINT_PATH,
                mbatch_size=self.config.VAL_BATCH_SIZE,
                device=str(self.device_manager.device),
                bert_pretrained_path=getattr(self.config, 'CHEXPERT_PRETRAIN_PATH', 'bert-base-uncased')
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
            # 获取原始模型（处理DDP包装）
            model = self.get_raw_model()
            _, _ = load(
                self.config.DECODER_CHECKPOINT_PATH_FROM,
                model,
                self.optimizer,
                self.scheduler,
                load_model="full",
                device=self.device_manager.device
            )
            self.logger.info(f"从 {self.config.DECODER_CHECKPOINT_PATH_FROM} 加载完整模型权重")
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

