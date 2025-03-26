"""
配置管理器，用于加载和管理配置
"""

import os
import sys
from .default_config import DefaultConfig

def load_config():
    """
    加载配置。优先使用local_config.py中的配置，如果不存在则使用默认配置
    """
    try:
        from .local_config import LocalConfig
        config = LocalConfig()
        print("成功加载本地配置文件")
    except ImportError:
        config = DefaultConfig()
        print("警告：未找到本地配置文件，使用默认配置")
        print("请复制 configs/local_config.template.py 为 configs/local_config.py 并修改相应配置")
    
    return config

# 全局配置对象
config = load_config() 