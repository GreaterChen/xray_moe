import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def merge_tensorboard_logs(log_dirs, output_dir):
    """
    合并多个TensorBoard日志目录中的事件文件
    
    参数:
    log_dirs: 包含TensorBoard日志文件的目录列表
    output_dir: 输出合并后日志文件的目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集所有事件
    all_events = []
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"处理文件: {log_dir}")
            try:
                loader = event_file_loader.EventFileLoader(log_dir)
                for event in loader.Load():
                    all_events.append((event.wall_time, event))
                print(f"从 {log_dir} 加载了 {len(all_events)} 个事件")
            except Exception as e:
                print(f"处理 {log_dir} 时出错: {e}")
        else:
            print(f"文件不存在: {log_dir}")
    
    # 按时间戳排序事件
    if all_events:
        all_events.sort(key=lambda x: x[0])
        
        # 写入新的事件文件
        output_file = os.path.join(output_dir, "events.out.tfevents.merged")
        print(f"写入事件到 {output_file}")
        with tf_record.TFRecordWriter(output_file) as writer:
            for _, event in all_events:
                writer.write(event.SerializeToString())
        
        print(f"已将日志合并到: {output_file}")
    else:
        print("没有事件被加载，无法创建合并文件")

# 使用示例
log_dir1 = "/home/chenlb/xray_moe/runs/MOE_FINETUNE_BERT_finetune_bert_vit_instruction-20250416-155933/events.out.tfevents.1744790373.ubuntu.1442452.0"  # 包含0-19轮结果的目录
log_dir2 = "/home/chenlb/xray_moe/runs/MOE_FINETUNE_BERT_finetune_bert_vit_instruction_extend-20250417-201257/events.out.tfevents.1744891977.ubuntu.329365.0"  # 包含20-30轮结果的目录
output_dir = "merged_logs"  # 合并后日志的输出目录

merge_tensorboard_logs([log_dir1, log_dir2], output_dir)