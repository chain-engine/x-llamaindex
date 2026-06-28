#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础加载器接口

定义文档加载器的标准接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

from llama_index.core import Document

from src.core import get_logger

logger = get_logger(__name__)


class BaseLoader(ABC):
    """基础加载器抽象类

    所有文档加载器都应该继承此类并实现 load 方法
    """

    def __init__(self, encoding: str = "utf-8"):
        """初始化加载器

        Args:
            encoding: 文件编码，默认 utf-8
        """
        self.encoding = encoding
        self._loaded_count = 0

    @abstractmethod
    def load(self, source: str | Path, **kwargs) -> List[Document]:
        """加载文档

        Args:
            source: 数据源路径或URL
            **kwargs: 额外参数

        Returns:
            加载的文档列表
        """
        pass

    def load_batch(
        self, sources: List[str | Path], **kwargs
    ) -> List[Document]:
        """批量加载文档

        Args:
            sources: 数据源列表
            **kwargs: 额外参数

        Returns:
            所有加载的文档列表
        """
        all_documents: List[Document] = []

        for source in sources:
            try:
                documents = self.load(source, **kwargs)
                all_documents.extend(documents)
                self._loaded_count += len(documents)
                logger.info(f"Loaded {len(documents)} documents from {source}")
            except Exception as e:
                logger.error(f"Failed to load {source}: {e}")
                continue

        logger.info(f"Total loaded: {self._loaded_count} documents")
        return all_documents

    @property
    def loaded_count(self) -> int:
        """获取已加载的文档数量"""
        return self._loaded_count

    def reset_count(self) -> None:
        """重置计数器"""
        self._loaded_count = 0
