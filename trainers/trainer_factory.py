"""训练器工厂类"""
from trainers.vit_trainer import ViTPretrainTrainer
from trainers.bert_trainer import BertFinetuneTrainer
from trainers.detection_trainer import DetectionTrainer
from trainers.iuxray_finetune_trainer import IUXRAYFinetuneTrainer


class TrainerFactory:
    """训练器工厂 - 根据配置创建对应的训练器"""
    
    _trainers = {
        "PRETRAIN_VIT": ViTPretrainTrainer,
        "FINETUNE_BERT": BertFinetuneTrainer,
        "TRAIN_DETECTION": DetectionTrainer,
        "FINETUNE_IUXRAY": IUXRAYFinetuneTrainer,
    }
    
    @classmethod
    def create_trainer(cls, config, device_manager, logger, tokenizer=None):
        """
        根据配置创建对应的训练器
        
        Args:
            config: 配置对象
            device_manager: 设备管理器
            logger: 日志记录器
            tokenizer: 分词器（某些训练器需要）
            
        Returns:
            Trainer实例
            
        Raises:
            ValueError: 如果训练阶段不支持
        """
        phase = config.PHASE
        
        if phase not in cls._trainers:
            raise ValueError(
                f"不支持的训练阶段: {phase}\n"
                f"支持的阶段: {list(cls._trainers.keys())}"
            )
        
        trainer_class = cls._trainers[phase]
        
        # BERT训练器和IU_XRAY训练器需要tokenizer
        if phase in ["FINETUNE_BERT", "FINETUNE_IUXRAY"]:
            if tokenizer is None:
                raise ValueError(f"{phase} 阶段需要提供tokenizer")
            return trainer_class(config, device_manager, logger, tokenizer)
        else:
            return trainer_class(config, device_manager, logger)
    
    @classmethod
    def register_trainer(cls, phase_name, trainer_class):
        """
        注册新的训练器类型
        
        Args:
            phase_name: 训练阶段名称
            trainer_class: 训练器类
        """
        cls._trainers[phase_name] = trainer_class
    
    @classmethod
    def list_supported_phases(cls):
        """列出所有支持的训练阶段"""
        return list(cls._trainers.keys())

