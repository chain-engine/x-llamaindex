#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志管理模块
使用 loguru 实现结构化日志
"""

from typing import Callable, Final
from pathlib import Path

from loguru import logger

# 移除默认的处理器
logger.remove()

# 延迟加载日志配置
_log_config = None


def _get_log_config() -> dict:
    """获取日志配置（延迟加载）"""
    global _log_config
    if _log_config is None:
        try:
            from src.core import settings
            _log_config = {
                'file_path': settings.LOG_FILE_PATH,
                'level': settings.LOG_LEVEL,
                'rotation': settings.LOG_ROTATION,
                'retention': settings.LOG_RETENTION,
            }
        except Exception:
            _log_config = {
                'file_path': 'logs/rag_app.log',
                'level': 'INFO',
                'rotation': '1 day',
                'retention': '7 days',
            }
    return _log_config


# 获取配置
cfg = _get_log_config()
_log_file_path = cfg['file_path']
_log_level = cfg['level']
_log_rotation = cfg['rotation']
_log_retention = cfg['retention']

# 确保日志目录存在
log_dir = Path(_log_file_path).parent
log_dir.mkdir(parents=True, exist_ok=True)

# 配置日志格式
LOG_FORMAT: Final[str] = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 配置文件日志处理器
logger.add(
    sink=_log_file_path,
    rotation=_log_rotation,
    retention=_log_retention,
    compression="zip",
    level=_log_level,
    format=LOG_FORMAT,
    enqueue=True,
    encoding="utf-8"
)

# 配置控制台日志处理器
console_sink: Callable[[str], None] = lambda msg: print(msg, end="")
logger.add(
    sink=console_sink,
    level="INFO",
    format=LOG_FORMAT,
    enqueue=True
)


def get_logger(name: str = __name__):
    """获取带模块名的logger实例"""
    return logger.bind(name=name)


# 导出
__all__: Final[list[str]] = ['logger', 'get_logger']
