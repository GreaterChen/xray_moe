"""数据库构建工具函数"""
import os
import torch
import pickle
from tqdm import tqdm
from utils.data_utils import prepare_batch_data, args_to_kwargs, data_distributor


def build_anatomical_database(
    config,
    model,
    data_loader,
    logger,
    device="cuda"
):
    """
    构建解剖区域特征数据库（分批保存版）
    
    Args:
        config: 配置对象
        model: MOE模型
        data_loader: 数据加载器（通常是训练集）
        logger: 日志记录器
        device: 计算设备
    """
    import numpy as np
    from datasets import MIMIC
    
    logger.info(f"开始构建解剖区域特征数据库...")
    logger.info(f"数据集大小: {len(data_loader.dataset)}")
    logger.info(f"每1000个batch保存一次")
    
    # 创建输出目录
    output_dir = config.DATABASE_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取解剖区域名称
    anatomical_regions = MIMIC.ANATOMICAL_REGIONS
    num_regions = len(anatomical_regions)
    
    logger.info(f"解剖区域数量: {num_regions}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 分批保存参数
    save_every_n_batches = 300
    current_part = 0
    
    # 初始化当前批次的存储结构
    def init_storage():
        return {}
    
    def save_current_batch(image_database, part_num):
        """保存当前批次的数据"""
        save_format = config.SAVE_FEATURES_FORMAT.lower()
        
        if save_format == "npz":
            # NPZ格式直接存储image_database
            filename = f"anatomical_database_part_{part_num:04d}.npz"
            np.savez_compressed(
                os.path.join(output_dir, filename),
                image_database=image_database
            )
            logger.info(f"第{part_num}部分已保存为NPZ格式: {filename}")
            
        elif save_format == "pkl":
            # Pickle格式直接存储image_database
            filename = f"anatomical_database_part_{part_num:04d}.pkl"
            pkl_path = os.path.join(output_dir, filename)
            with open(pkl_path, 'wb') as f:
                pickle.dump(image_database, f)
            logger.info(f"第{part_num}部分已保存为Pickle格式: {filename}")
        
        return filename
    
    def cleanup_memory():
        """清理内存和显存"""
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("内存和显存已清理")
    
    # 初始化存储
    image_database = init_storage()
    total_samples = 0
    total_detected_regions = 0
    saved_files = []
    
    # 处理每个批次
    with torch.no_grad():
        prog_bar = tqdm(data_loader, desc="构建数据库")
        
        for batch_idx, batch in enumerate(prog_bar):
            # 准备批次数据
            source, target, _ = prepare_batch_data(
                config,
                batch,
                data_loader,
                device,
                findings=False,  # 不需要findings
                history=False,   # 不需要history
                label=False,     # 不需要疾病标签
                bbox=True,       # 需要边界框
            )
            
            # 转换为kwargs格式
            source = args_to_kwargs(source)
            source["phase"] = "BUILD_DATABASE"
            source["mode"] = "eval"
            
            # 获取图像ID
            image_ids = batch["image_id"]
            
            # 模型推理
            outputs = data_distributor(model, source)
            
            # 提取特征
            region_features = outputs["region_features"]  # [B, 29, 768]
            region_detected = outputs["region_detected"]  # [B, 29]
            
            batch_size = region_features.size(0)
            
            # 处理每个样本
            for i in range(batch_size):
                sample_features = region_features[i].cpu().numpy()  # [29, 768]
                sample_detected = region_detected[i].cpu().numpy()  # [29]
                image_id = image_ids[i]
                sample_idx = total_samples + i
                
                # 构建该图像的区域特征字典
                image_regions = {}
                sample_detected_count = 0
                
                for region_idx in range(num_regions):
                    if sample_detected[region_idx]:  # 只保存检测到的区域
                        image_regions[region_idx] = {
                            "region_name": anatomical_regions[region_idx],
                            "feature": sample_features[region_idx]  # [768]
                        }
                        sample_detected_count += 1
                        total_detected_regions += 1
                
                # 保存到数据库
                image_database[image_id] = {
                    "regions": image_regions,
                    "sample_idx": sample_idx,
                    "detected_count": sample_detected_count
                }
            
            total_samples += batch_size
            
            # 更新进度条信息
            avg_detected = total_detected_regions / total_samples if total_samples > 0 else 0
            prog_bar.set_description(
                f"样本: {total_samples}, 平均检测区域: {avg_detected:.1f}/{num_regions}, 当前部分: {current_part}"
            )
            
            # 检查是否需要保存
            if (batch_idx + 1) % save_every_n_batches == 0:
                logger.info(f"\n到达{save_every_n_batches}个batch，开始保存第{current_part}部分...")
                
                # 保存当前批次
                filename = save_current_batch(image_database, current_part)
                saved_files.append(filename)
                
                # 统计当前部分信息
                current_images = len(image_database)
                current_detected = sum(img_data['detected_count'] for img_data in image_database.values())
                
                logger.info(f"第{current_part}部分统计:")
                logger.info(f"  图像数: {current_images}")
                logger.info(f"  检测区域数: {current_detected}")
                logger.info(f"  平均每图像检测区域数: {current_detected / current_images:.2f}")
                
                # 清理当前数据
                image_database = init_storage()
                current_part += 1
                
                # 清理内存
                cleanup_memory()
                
                logger.info(f"开始处理第{current_part}部分...\n")
    
    # 保存最后剩余的数据（如果有）
    if image_database:
        logger.info(f"\n保存最后剩余的数据（第{current_part}部分）...")
        filename = save_current_batch(image_database, current_part)
        saved_files.append(filename)
        
        # 统计最后部分信息
        current_images = len(image_database)
        current_detected = sum(img_data['detected_count'] for img_data in image_database.values())
        
        logger.info(f"第{current_part}部分统计:")
        logger.info(f"  图像数: {current_images}")
        logger.info(f"  检测区域数: {current_detected}")
        logger.info(f"  平均每图像检测区域数: {current_detected / current_images:.2f}")
    
    # 最终清理
    cleanup_memory()
    
    logger.info(f"\n=== 数据库构建完成 ===")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"总检测区域数: {total_detected_regions}")
    logger.info(f"平均每样本检测区域数: {total_detected_regions / total_samples:.2f}")
    logger.info(f"总共保存了 {len(saved_files)} 个文件:")
    
    for i, filename in enumerate(saved_files):
        logger.info(f"  {i+1}. {filename}")
    
    logger.info(f"\n数据库文件保存在: {output_dir}")
    
    # 创建文件列表
    files_list_path = os.path.join(output_dir, "database_files.txt")
    with open(files_list_path, 'w') as f:
        for filename in saved_files:
            f.write(f"{filename}\n")
    logger.info(f"文件列表已保存: database_files.txt")
    
    return {
        "output_dir": output_dir,
        "total_samples": total_samples,
        "total_detected_regions": total_detected_regions,
        "saved_files": saved_files,
        "num_parts": len(saved_files),
    }

