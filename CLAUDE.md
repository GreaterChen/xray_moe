# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical X-ray report generation system that uses a multi-stage training approach combining object detection, vision transformers (ViT), and language models (BERT/Qwen2.5-VL) to generate radiology reports from chest X-ray images. The system supports both MIMIC-CXR and IU-XRAY datasets and includes distributed training capabilities.

## Training Pipeline Architecture

The project follows a **multi-stage training pipeline** controlled by the `PHASE` configuration:

1. **TRAIN_DETECTION**: Train object detector to identify 29 anatomical regions
2. **PRETRAIN_VIT**: Pretrain vision transformer with contrastive learning (region-text alignment)
3. **FINETUNE_BERT**: Fine-tune decoder with visual features and optional RGAT (Relational Graph Attention Network) for clinical reasoning
4. **FINETUNE_IUXRAY**: Transfer learning to IU-XRAY dataset

### Key Components

- **Object Detector**: `DetectionOnlyFastRCNN` / `EnhancedFastRCNN` - Detects 29 anatomical regions
- **Vision Encoder**: `MedicalVisionTransformer` (detection+vit) or `PatchOnlyVisionTransformer` (vit_only)
- **RGAT Module**: `ThreeStageRGAT` - Three-stage clinical reasoning network (A→A, A→D, D→D) using anatomical and disease graphs
- **Decoder**: BERT or Qwen2.5-VL with optional LoRA fine-tuning
- **Main Model**: `MedicalReportGenerator` orchestrates all components

### Encoder Strategies

The system supports two encoding strategies (set via `ENCODER_TYPE` in config):

- **`detection+vit`** (default): Object detector extracts 29 region features → ViT processes them
- **`vit_only`**: Direct ViT encoding without object detection (using `PatchOnlyVisionTransformer`)

## Configuration System

Configuration uses a two-layer approach:

- `configs/default_config.py`: Default values for all settings
- `configs/local_config.py`: Override defaults with environment-specific values (this file takes precedence)

**Important config locations:**
- Dataset paths: `DATA_DIR`, `ANN_DIR`, `IMAGES_DIR`
- Model checkpoints: `DETECTION_CHECKPOINT_PATH_FROM`, `VIT_CHECKPOINT_PATH_FROM`, `CHECKPOINT_PATH_FROM`
- Training phase: `PHASE` (determines which trainer is used)
- Encoder type: `ENCODER_TYPE` (detection+vit or vit_only)
- Decoder type: `DECODER_TYPE` (bert or qwen2vl)
- Graph matrices for RGAT: `AA_ADJ_PATH`, `DD_ADJ_PATH`, `DA_ADJ_PATH`

The config is accessed globally via: `from configs import config`

## Common Development Commands

### Training

```bash
# Single GPU training
python train.py

# Multi-GPU distributed training (DDP)
torchrun --nproc_per_node=4 train.py

# Change training phase by modifying configs/local_config.py:
# PHASE = "TRAIN_DETECTION"  # or "PRETRAIN_VIT", "FINETUNE_BERT", "FINETUNE_IUXRAY"
```

### Evaluation

```bash
# Compare inference strategies
python compare_inference_strategies.py

# Calculate metrics
python calculate_all_metrics.py

# Evaluate detection performance
python evaluate_detection.py
```

### Configuration

Before training, ensure `configs/local_config.py` is properly configured:
- Set correct paths for datasets and checkpoints
- Set `PHASE` to the desired training stage
- Configure GPU settings (`USE_CUDA`, `CUDA_VISIBLE_DEVICES`, `USE_DISTRIBUTED`)

## Code Architecture

### Trainer Factory Pattern

The project uses a factory pattern for trainers (`trainers/trainer_factory.py`):

- `TrainerFactory.create_trainer(config, ...)` automatically instantiates the correct trainer based on `config.PHASE`
- Each trainer inherits from `BaseTrainer` and implements stage-specific logic
- Supported trainers: `DetectionTrainer`, `ViTPretrainTrainer`, `BertFinetuneTrainer`, `IUXRAYFinetuneTrainer`

### Model Building

Centralized model construction in `models/model_builder.py`:

- `build_detection_model()`: Creates and loads pretrained object detector
- `build_vit_model()`: Creates ViT with optional pretrained weights
- `build_image_encoder()`: Returns appropriate encoder based on `ENCODER_TYPE` config
- `freeze_model_parameters()`: Freezes model parameters for multi-stage training

### Device Management

`DeviceManager` class (`device_utils.py`) handles all device-related logic:

- Automatically detects single GPU / multi-GPU / CPU environments
- Supports both DataParallel (DP) and DistributedDataParallel (DDP)
- Use `device_manager.wrap_model(model)` to automatically wrap models for the current environment
- Use `device_manager.get_sampler(dataset)` to get appropriate sampler (DistributedSampler for DDP)

### Distributed Training

The codebase fully supports DDP (see `docs/DDP分布式训练技术文档.md` for detailed explanation):

- Launch with `torchrun --nproc_per_node=N train.py`
- `DeviceManager` handles process group initialization
- `setup_for_distributed()` ensures only main process prints to console
- Always call `train_sampler.set_epoch(epoch)` at the start of each epoch for proper shuffling

### Data Loading

Two dataset implementations:

- `MIMIC` class: MIMIC-CXR dataset with multi-view support
- `IUXRAY` class: IU-XRAY dataset

Both use class-level shared data loading via `load_shared_data()` classmethod to avoid redundant loading across workers.

Key features:
- Supports anatomical region embeddings for region-text contrastive learning
- Handles bounding box targets for detection training
- Custom collate functions: `mimic_collate_fn`, `iuxray_collate_fn`

## Important Implementation Details

### RGAT (Relational Graph Attention Network)

The RGAT module (`models/rgat.py`) implements three-stage clinical reasoning:

1. **Stage 1 (A→A)**: Anatomical region context aggregation using AA adjacency matrix
2. **Stage 2 (A→D)**: Disease-specific feature aggregation using DA matrix (anatomy-to-disease)
3. **Stage 3 (D→D)**: Disease relationship reasoning using DD matrix (disease co-occurrence)

RGAT is initialized during `FINETUNE_BERT` phase if graph matrices are provided in config.

### Contrastive Learning

During `PRETRAIN_VIT` phase, three types of contrastive learning are supported (controlled by `CONTRASTIVE_LOSS_TYPE`):

- **`region`**: Complex region-level contrastive learning considering NLP status and same-text regions
- **`clip`**: Standard CLIP-style image-report contrastive learning
- **`simple_region_clip`**: Simplified region-level CLIP (patch-sentence pairing)

### Multi-Decoder Support

The system supports two decoder types:

- **BERT**: Standard BERT decoder with learnable decoder embeddings
- **Qwen2.5-VL**: Large language model with optional LoRA fine-tuning (set `USE_LORA=True` to save memory)

Decoder type is controlled by `DECODER_TYPE` config parameter.

### Checkpoint Management

- Detection model checkpoints: Load with `load(path, model, load_model="full")`
- ViT checkpoints: Load with `load(path, model, load_model="vit")`
- DDP models: Access state dict via `model.module.state_dict()`
- Only main process (`rank == 0`) should save checkpoints to avoid conflicts

## File Organization

```
xray_moe/
├── configs/           # Configuration management
│   ├── default_config.py
│   ├── local_config.py  # Override defaults here
│   ├── constants.py     # ANATOMY_ORDER, DISEASE_ORDER
│   └── config_manager.py
├── models/            # Model implementations
│   ├── medical_report_generator.py  # Main orchestrator
│   ├── fast_rcnn_classifier.py      # Object detector
│   ├── vit.py                        # Vision transformers
│   ├── rgat.py                       # Clinical reasoning network
│   ├── bert_adapter.py               # BERT decoder
│   ├── qwenvl_decoder.py             # Qwen decoder
│   └── model_builder.py              # Model construction utilities
├── trainers/          # Training logic
│   ├── base_trainer.py
│   ├── detection_trainer.py
│   ├── vit_trainer.py
│   ├── bert_trainer.py
│   ├── iuxray_finetune_trainer.py
│   └── trainer_factory.py
├── utils/             # Helper utilities
│   ├── checkpoint_utils.py
│   ├── eval_utils.py
│   ├── detection_metrics.py
│   └── train_utils.py
├── datasets.py        # Dataset implementations
├── device_utils.py    # Device and distributed training management
├── train.py           # Main training script
├── losses.py          # Loss functions
└── docs/              # Documentation
```

## Dataset Structure

### MIMIC-CXR Expected Structure

```
DATA_DIR/
├── annotation.json    # ANN_DIR points to this
├── images_224/        # IMAGES_DIR - preprocessed images
└── extra/             # Additional data for RGAT
    ├── anatomy_distance_matrix/
    ├── disease_graph/
    └── anatomy_disease_matrix/
```

### IU-XRAY Expected Structure

```
IUXRAY_DIR/
├── annotation_with_history_view_labels_split_multi_view_entries_pa_ap_views.json
└── images_224/
```

## Key Constants

From `configs/constants.py`:

- **29 Anatomical Regions** (`ANATOMY_ORDER`): left hemidiaphragm, right atrium, right hilar structures, etc.
  - In code: indices 0-28
  - In bbox_targets: labels 1-29 (0 is background)
- **14 Diseases** (`DISEASE_ORDER`): Atelectasis, Cardiomegaly, Consolidation, etc. (indices 0-13)

These orderings must be consistent with graph matrices for RGAT to work correctly.

## Testing and Metrics

The project includes comprehensive evaluation tools:

- **CheXbert metrics**: Clinical efficacy metrics using CheXbert model
- **NLG metrics**: BLEU, METEOR, ROUGE-L via `pycocoevalcap`
- **Detection metrics**: Precision, recall, F1, mAP at multiple IoU thresholds
- **Inference strategies**: Compare different decoding strategies (greedy, beam search, nucleus sampling)

## Memory and Performance Considerations

- For large models (especially Qwen2.5-VL), enable LoRA: `USE_LORA = True`
- For IO-bound systems: Increase `NUM_WORKERS`, enable `PERSISTENT_WORKERS`
- For memory constraints: Reduce batch size or use gradient accumulation
- Use mixed precision training: `USE_MIXED_PRECISION = True`
- DDP is 20-30% faster than DataParallel for multi-GPU training

## Troubleshooting Common Issues

### DDP hangs or crashes
- Ensure `torchrun` is used for launching (not plain `python`)
- Check that all processes execute collective operations (barriers, all_reduce)
- Set `export NCCL_DEBUG=INFO` for debugging

### Out of memory
- Reduce batch size in config
- Enable LoRA for Qwen decoder: `USE_LORA = True`
- Use gradient checkpointing if available

### Checkpoint loading errors
- For DDP: Use `model.module.load_state_dict()` instead of `model.load_state_dict()`
- Check `load_model` parameter: "full" for detection, "vit" for ViT, "bert" for decoder

### Wrong training phase
- Verify `PHASE` setting in `configs/local_config.py`
- Ensure required checkpoint paths are set for the current phase
- Check that prerequisite phases have been completed (e.g., detection before ViT pretraining)
