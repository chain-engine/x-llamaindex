#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理器模块

提供文本分割和元数据提取功能
"""

from src.processors.splitter import TextSplitter, ChunkingStrategy
from src.processors.metadata import MetadataExtractor

__all__ = ["TextSplitter", "ChunkingStrategy", "MetadataExtractor"]
