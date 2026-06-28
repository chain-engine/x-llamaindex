#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础索引管理器接口

定义索引管理器的标准接口
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any

from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever

from src.core import get_logger

logger = get_logger(__name__)


class BaseIndexManager(ABC):
    """基础索引管理器抽象类

    所有索引管理器都应该继承此类
    """

    def __init__(self):
        """初始化索引管理器"""
        self._index = None
        self._document_count = 0
        self._node_count = 0

    @abstractmethod
    def build_index(
        self,
        documents: List[Document],
        **kwargs,
    ) -> Any:
        """从文档构建索引

        Args:
            documents: 文档列表
            **kwargs: 额外参数

        Returns:
            构建的索引
        """
        pass

    @abstractmethod
    def build_index_from_nodes(
        self,
        nodes: List[TextNode],
        **kwargs,
    ) -> Any:
        """从节点构建索引

        Args:
            nodes: 节点列表
            **kwargs: 额外参数

        Returns:
            构建的索引
        """
        pass

    @abstractmethod
    def insert_documents(self, documents: List[Document]) -> None:
        """插入新文档到索引

        Args:
            documents: 文档列表
        """
        pass

    @abstractmethod
    def insert_nodes(self, nodes: List[TextNode]) -> None:
        """插入新节点到索引

        Args:
            nodes: 节点列表
        """
        pass

    @abstractmethod
    def delete(self, doc_ids: List[str]) -> None:
        """删除文档

        Args:
            doc_ids: 文档 ID 列表
        """
        pass

    @abstractmethod
    def get_retriever(self, **kwargs) -> BaseRetriever:
        """获取检索器

        Args:
            **kwargs: 检索器参数

        Returns:
            检索器实例
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """保存索引到磁盘

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """从磁盘加载索引

        Args:
            path: 加载路径
        """
        pass

    @property
    def index(self) -> Any:
        """获取索引实例"""
        return self._index

    @property
    def document_count(self) -> int:
        """获取文档数量"""
        return self._document_count

    @property
    def node_count(self) -> int:
        """获取节点数量"""
        return self._node_count
