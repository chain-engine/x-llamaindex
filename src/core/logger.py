#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志管理模块
使用 loguru 实现结构化日志
"""

import os
import sys
from typing import Callable, Final
from loguru import logger
from pathlib import Path

# 移除默认的处理器
logger.remove()

# 确保日志目录存在
log_dir: Path = Path("logs")
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
    sink=log_dir / "rag_app_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    level="DEBUG",
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


# 导出logger实例
__all__: Final[list[str]] = ['logger', 'get_logger']
