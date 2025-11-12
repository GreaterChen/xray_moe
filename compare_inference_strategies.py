#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比不同推理策略的效果
比较 do_sample 采样策略 和 beam search 策略的生成质量

使用方法:
    python3 compare_inference_strategies.py \
        --checkpoint /path/to/checkpoint.pth \
        --output_dir ./comparison_results
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import warnings
from datetime import datetime
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import logging as hf_logging

# 屏蔽警告
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import config
from datasets import MIMIC, mimic_collate_fn, IUXRAY, iuxray_collate_fn
from models.medical_report_generator import MedicalReportGenerator
from models.bert_adapter import BertAdapter
from models.qwenvl_decoder import QwenVLAdapter
from models.model_builder import build_detection_model, build_vit_model, freeze_model_parameters, build_image_encoder
from metrics import compute_scores
from tools.metrics_clinical import CheXbertMetrics
from utils import setup_logger, load
from device_utils import DeviceManager


class InferenceComparator:
    """推理策略对比器"""
    
    def __init__(
        self, 
        config, 
        checkpoint_path,
        device="cuda",
        output_dir="./comparison_results"
    ):
        """
        初始化推理对比器
        
        Args:
            config: 配置对象
            checkpoint_path: 模型权重路径
            device: 计算设备
            output_dir: 输出目录
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        self.logger, _ = setup_logger(
            log_dir=os.path.join(output_dir, "logs"), 
            is_main_process=True
        )
        
        self.logger.info("=" * 80)
        self.logger.info("推理策略对比器初始化")
        self.logger.info("=" * 80)
        self.logger.info(f"模型权重: {checkpoint_path}")
        self.logger.info(f"计算设备: {self.device}")
        self.logger.info(f"输出目录: {output_dir}")
        
        # 初始化tokenizer
        self.tokenizer = self._setup_tokenizer()
        
        # 构建和加载模型
        self.model = self._build_model()
        self._load_checkpoint()
        
        # 初始化CheXbert评估器
        self.chexbert_metrics = self._setup_chexbert()
        
    def _setup_tokenizer(self):
        """设置tokenizer"""
        self.logger.info("初始化tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", 
            local_files_only=True
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    def _build_model(self):
        """构建模型（支持BERT和Qwen2.5-VL decoder）"""
        decoder_type = getattr(self.config, 'DECODER_TYPE', 'bert').lower()
        self.logger.info(f"构建模型 (decoder类型: {decoder_type})...")
        
        # 1. 构建检测器
        enhanced_rcnn = build_detection_model(
            self.config, 
            self.logger, 
            device=self.device
        )
        
        # 2. 构建ViT
        image_encoder = build_image_encoder(
            self.config,
            logger=None,
            device=self.device
        )
        
        # 3. 根据配置创建解码器
        if decoder_type == 'qwen2vl':
            self.logger.info("初始化Qwen VL解码器...")
            qwen_model_name = getattr(self.config, 'QWEN_MODEL_NAME', 'Qwen/Qwen3-VL-4B-Instruct')
            decoder_model = QwenVLAdapter(
                config=self.config,
                tokenizer=self.tokenizer,
                hidden_dim=768,
                max_length=196,
                qwen_model_name=qwen_model_name
            )
            self.logger.info(f"✅ Qwen2.5-VL解码器初始化完成 (模型: {qwen_model_name})")
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
        
        # 4. 组装模型
        model = MedicalReportGenerator(
            config=self.config,
            object_detector=enhanced_rcnn,
            image_encoder=image_encoder,
            findings_decoder=decoder_model
        )
        
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"✅ {decoder_type.upper()}模型构建完成")
        return model
    
    def _load_checkpoint(self):
        """加载模型权重"""
        self.logger.info(f"加载模型权重: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"找不到模型权重文件: {self.checkpoint_path}")
        
        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 如果checkpoint是完整训练状态，提取模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DDP包装的权重（移除'module.'前缀）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.logger.info("✅ 模型权重加载完成")
    
    def _setup_chexbert(self):
        """初始化CheXbert评估器"""
        if not hasattr(self.config, 'CHEXBERT_CHECKPOINT_PATH') or \
           not self.config.CHEXBERT_CHECKPOINT_PATH:
            self.logger.warning("未配置CheXbert路径，跳过临床效果评估")
            return None
        
        try:
            self.logger.info("初始化CheXbert评估器...")
            chexbert = CheXbertMetrics(
                checkpoint_path=self.config.CHEXBERT_CHECKPOINT_PATH,
                mbatch_size=16,
                device=str(self.device),
                bert_pretrained_path=getattr(
                    self.config, 
                    'BERT_PRETRAINED_PATH', 
                    'bert-base-uncased'
                )
            )
            self.logger.info("✅ CheXbert评估器初始化成功")
            return chexbert
        except Exception as e:
            self.logger.warning(f"⚠️  CheXbert评估器初始化失败: {e}")
            return None
    
    def create_test_loader(self, subset_size=None):
        """创建测试数据加载器"""
        self.logger.info("创建测试数据集...")
        
        input_size = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)
        dataset_name = getattr(self.config, 'DATASET_NAME', 'MIMIC')
        
        if dataset_name == 'IUXRAY':
            # IU_XRAY数据集
            IUXRAY.load_shared_data(ann_path=self.config.IUXRAY_ANN_PATH)
            
            test_data = IUXRAY(
                ann_path=self.config.IUXRAY_ANN_PATH,
                images_dir=self.config.IUXRAY_IMAGES_DIR,
                input_size=input_size,
                random_transform=False,
                tokenizer=self.tokenizer,
                mode="test"
            )
            collate_fn = iuxray_collate_fn
        else:
            # MIMIC数据集
            MIMIC.load_shared_data(
                directory=self.config.DATA_DIR,
                ann_dir=self.config.ANN_DIR,
                mode="INFER",
                binary_mode=True,
                split_csv_path=self.config.SPLIT_CSV_PATH,
                generation_target=self.config.GENERATION_TARGET
            )
            
            test_data = MIMIC(
                directory=self.config.DATA_DIR,
                ann_dir=self.config.ANN_DIR,
                images_dir=self.config.IMAGES_DIR,
                input_size=input_size,
                random_transform=False,
                tokenizer=self.tokenizer,
                mode="test",
                subset_size=subset_size,
                generation_target=self.config.GENERATION_TARGET
            )
            collate_fn = mimic_collate_fn
        
        test_loader = data.DataLoader(
            test_data,
            batch_size=getattr(self.config, 'VAL_BATCH_SIZE', 16),
            shuffle=False,
            num_workers=getattr(self.config, 'NUM_WORKERS', 4),
            collate_fn=collate_fn
        )
        
        self.logger.info(f"✅ 测试集大小: {len(test_data)}")
        return test_loader
    
    @torch.no_grad()
    def generate_with_strategy(
        self, 
        data_loader, 
        strategy_name,
        generation_params
    ):
        """
        使用指定策略进行生成
        
        Args:
            data_loader: 数据加载器
            strategy_name: 策略名称
            generation_params: 生成参数字典
        
        Returns:
            results: 包含生成结果的字典
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"使用策略: {strategy_name}")
        self.logger.info(f"生成参数: {generation_params}")
        self.logger.info(f"{'='*80}\n")
        
        self.model.eval()
        
        all_generated = []
        all_ground_truth = []
        all_study_ids = []
        
        for batch_idx, batch_data in enumerate(tqdm(
            data_loader, 
            desc=f"生成中 ({strategy_name})"
        )):
            # 准备输入数据
            images = batch_data["image"].to(self.device)
            history_input_ids = batch_data["history"]["input_ids"].to(self.device)
            history_attention_mask = batch_data["history"]["attention_mask"].to(self.device)
            ground_truth = batch_data["findings"]["ground_truth"]
            study_ids = batch_data.get("study_id", [f"sample_{batch_idx}_{i}" for i in range(len(ground_truth))])
            
            # 前向传播获取视觉特征
            visual_features, visual_attention_mask = self.model(
                images=images,
                history_input_ids=history_input_ids,
                history_attention_mask=history_attention_mask,
                mode="encode_only"
            )
            
            # 使用指定参数生成文本
            generated_texts = self.model.findings_decoder.generate(
                history_input_ids=history_input_ids,
                history_attention_mask=history_attention_mask,
                visual_features=visual_features,
                visual_attention_mask=visual_attention_mask,
                **generation_params
            )
            
            all_generated.extend(generated_texts)
            all_ground_truth.extend(ground_truth)
            all_study_ids.extend(study_ids)
        
        # 保存生成结果
        results = {
            "study_ids": all_study_ids,
            "generated": all_generated,
            "ground_truth": all_ground_truth,
            "strategy": strategy_name,
            "params": generation_params
        }
        
        return results
    
    def evaluate_results(self, results, strategy_name):
        """
        评估生成结果
        
        Args:
            results: 生成结果字典
            strategy_name: 策略名称
        
        Returns:
            metrics: 评估指标字典
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"评估策略: {strategy_name}")
        self.logger.info(f"{'='*80}\n")
        
        generated = results["generated"]
        ground_truth = results["ground_truth"]
        
        # 1. 计算NLG指标
        self.logger.info("计算NLG指标...")
        nlg_metrics = compute_scores(
            gts={i: [gt] for i, gt in enumerate(ground_truth)},
            res={i: [gen] for i, gen in enumerate(generated)}
        )
        
        # 2. 计算CheXbert临床效果指标
        chexbert_results = {}
        if self.chexbert_metrics is not None:
            self.logger.info("计算临床效果指标...")
            try:
                chexbert_results = self.chexbert_metrics.compute(
                    generated_reports=generated,
                    ground_truth_reports=ground_truth
                )
            except Exception as e:
                self.logger.warning(f"CheXbert计算失败: {e}")
        
        # 3. 统计生成文本的长度
        gen_lengths = [len(text.split()) for text in generated]
        length_stats = {
            "mean_length": np.mean(gen_lengths),
            "std_length": np.std(gen_lengths),
            "min_length": np.min(gen_lengths),
            "max_length": np.max(gen_lengths),
            "median_length": np.median(gen_lengths)
        }
        
        # 整合所有指标
        all_metrics = {
            "strategy": strategy_name,
            "nlg_metrics": nlg_metrics,
            "chexbert_metrics": chexbert_results,
            "length_stats": length_stats
        }
        
        # 打印结果
        self._print_metrics(all_metrics)
        
        return all_metrics
    
    def _print_metrics(self, metrics):
        """打印评估指标"""
        self.logger.info("\n" + "="*80)
        self.logger.info(f"策略: {metrics['strategy']}")
        self.logger.info("="*80)
        
        # NLG指标
        self.logger.info("\n📊 NLG指标:")
        nlg = metrics['nlg_metrics']
        for key, value in nlg.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.4f}")
        
        # CheXbert指标
        if metrics['chexbert_metrics']:
            self.logger.info("\n🏥 临床效果指标:")
            chex = metrics['chexbert_metrics']
            for key, value in chex.items():
                if isinstance(value, (int, float)):
                    self.logger.info(f"  {key}: {value:.4f}")
        
        # 长度统计
        self.logger.info("\n📏 生成长度统计:")
        length = metrics['length_stats']
        for key, value in length.items():
            self.logger.info(f"  {key}: {value:.2f}")
    
    def save_results(self, all_results, all_metrics):
        """
        保存对比结果
        
        Args:
            all_results: 所有生成结果列表
            all_metrics: 所有评估指标列表
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存生成的文本到CSV
        for results in all_results:
            strategy_name = results['strategy']
            df = pd.DataFrame({
                'study_id': results['study_ids'],
                'generated': results['generated'],
                'ground_truth': results['ground_truth']
            })
            csv_path = os.path.join(
                self.output_dir, 
                f"generated_{strategy_name}_{timestamp}.csv"
            )
            df.to_csv(csv_path, index=False)
            self.logger.info(f"生成结果已保存: {csv_path}")
        
        # 2. 保存指标对比到CSV
        comparison_data = []
        for metrics in all_metrics:
            row = {
                'strategy': metrics['strategy'],
                'params': str(metrics.get('params', {}))
            }
            
            # NLG指标
            for key, value in metrics['nlg_metrics'].items():
                if isinstance(value, (int, float)):
                    row[f'nlg_{key}'] = value
            
            # CheXbert指标
            if metrics['chexbert_metrics']:
                for key, value in metrics['chexbert_metrics'].items():
                    if isinstance(value, (int, float)):
                        row[f'chexbert_{key}'] = value
            
            # 长度统计
            for key, value in metrics['length_stats'].items():
                row[f'length_{key}'] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(
            self.output_dir, 
            f"metrics_comparison_{timestamp}.csv"
        )
        comparison_df.to_csv(comparison_path, index=False)
        self.logger.info(f"指标对比已保存: {comparison_path}")
        
        # 3. 打印最终对比总结
        self._print_comparison_summary(comparison_df)
    
    def _print_comparison_summary(self, comparison_df):
        """打印对比总结"""
        self.logger.info("\n" + "="*80)
        self.logger.info("📈 最终对比总结")
        self.logger.info("="*80 + "\n")
        
        # 关键指标对比
        key_metrics = [
            'nlg_BLEU_1', 'nlg_BLEU_4', 'nlg_METEOR', 'nlg_ROUGE_L',
            'chexbert_ce_f1', 'chexbert_ce_precision', 'chexbert_ce_recall',
            'length_mean_length'
        ]
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                self.logger.info(f"\n{metric}:")
                for _, row in comparison_df.iterrows():
                    self.logger.info(f"  {row['strategy']}: {row[metric]:.4f}")
        
        # 推荐最佳策略
        if 'chexbert_ce_f1' in comparison_df.columns:
            best_idx = comparison_df['chexbert_ce_f1'].idxmax()
            best_strategy = comparison_df.loc[best_idx, 'strategy']
            best_score = comparison_df.loc[best_idx, 'chexbert_ce_f1']
            self.logger.info(f"\n🏆 最佳策略 (基于CE F1): {best_strategy} (F1={best_score:.4f})")
        elif 'nlg_BLEU_4' in comparison_df.columns:
            best_idx = comparison_df['nlg_BLEU_4'].idxmax()
            best_strategy = comparison_df.loc[best_idx, 'strategy']
            best_score = comparison_df.loc[best_idx, 'nlg_BLEU_4']
            self.logger.info(f"\n🏆 最佳策略 (基于BLEU-4): {best_strategy} (BLEU-4={best_score:.4f})")
    
    def run_comparison(self, subset_size=None):
        """
        运行完整对比实验
        
        Args:
            subset_size: 测试子集大小（None表示使用全部数据）
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("开始推理策略对比实验")
        self.logger.info("="*80 + "\n")
        
        # 创建测试数据加载器
        test_loader = self.create_test_loader(subset_size=subset_size)
        
        # 定义要对比的策略
        strategies = {
            "beam_search_3": {
                "num_beams": 3,
                "do_sample": False,
                "max_new_tokens": getattr(self.config, 'GEN_MAX_NEW_TOKENS', 150),
                "repetition_penalty": getattr(self.config, 'GEN_REPETITION_PENALTY', 1.0)
            },
            "beam_search_5": {
                "num_beams": 5,
                "do_sample": False,
                "max_new_tokens": getattr(self.config, 'GEN_MAX_NEW_TOKENS', 150),
                "repetition_penalty": getattr(self.config, 'GEN_REPETITION_PENALTY', 1.0)
            },
            "sampling_temp0.7": {
                "num_beams": 1,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": getattr(self.config, 'GEN_MAX_NEW_TOKENS', 150),
                "repetition_penalty": getattr(self.config, 'GEN_REPETITION_PENALTY', 1.0)
            },
            "sampling_temp1.0": {
                "num_beams": 1,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
                "max_new_tokens": getattr(self.config, 'GEN_MAX_NEW_TOKENS', 150),
                "repetition_penalty": getattr(self.config, 'GEN_REPETITION_PENALTY', 1.0)
            },
            "greedy": {
                "num_beams": 1,
                "do_sample": False,
                "max_new_tokens": getattr(self.config, 'GEN_MAX_NEW_TOKENS', 150),
                "repetition_penalty": getattr(self.config, 'GEN_REPETITION_PENALTY', 1.0)
            }
        }
        
        all_results = []
        all_metrics = []
        
        # 对每个策略进行生成和评估
        for strategy_name, params in strategies.items():
            # 生成
            results = self.generate_with_strategy(
                test_loader, 
                strategy_name, 
                params
            )
            results['params'] = params
            all_results.append(results)
            
            # 评估
            metrics = self.evaluate_results(results, strategy_name)
            metrics['params'] = params
            all_metrics.append(metrics)
        
        # 保存结果
        self.save_results(all_results, all_metrics)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("✅ 对比实验完成！")
        self.logger.info("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对比不同推理策略的效果")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="测试子集大小（用于快速测试，None表示使用全部数据）"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建对比器
        comparator = InferenceComparator(
            config=config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            output_dir=args.output_dir
        )
        
        # 运行对比实验
        comparator.run_comparison(subset_size=args.subset_size)
        
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"\n实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

