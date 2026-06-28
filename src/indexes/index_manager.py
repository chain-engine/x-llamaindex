#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级索引管理器

提供统一的索引管理接口，支持多种索引类型
"""

import os
from typing import List, Optional, Any, Dict, Type
from enum import Enum
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.base.base_retriever import BaseRetriever

from src.indexes.base import BaseIndexManager
from src.indexes.vector_index import VectorIndexManager
from src.storage import VectorStoreManager
from src.core import get_logger

logger = get_logger(__name__)


class IndexType(str, Enum):
    """索引类型枚举"""

    VECTOR = "vector"
    # 可以扩展其他索引类型
    # TREE = "tree"
    # KEYWORD = "keyword"


class IndexManager:
    """高级索引管理器

    提供统一的索引管理接口，支持多种索引类型和自动配置

    Example:
        >>> manager = IndexManager(
        ...     index_type=IndexType.VECTOR,
        ...     persist_dir="./storage/index"
        ... )
        >>> # 构建索引
        >>> manager.build_from_documents(documents)
        >>> # 获取检索器
        >>> retriever = manager.get_retriever()
        >>> # 保存索引
        >>> manager.save()
    """

    INDEX_MANAGERS: Dict[IndexType, Type[BaseIndexManager]] = {
        IndexType.VECTOR: VectorIndexManager,
    }

    def __init__(
        self,
        index_type: IndexType = IndexType.VECTOR,
        vector_store_manager: Optional[VectorStoreManager] = None,
        embed_model: Optional[Any] = None,
        persist_dir: Optional[str] = None,
        auto_save: bool = True,
    ):
        """初始化索引管理器

        Args:
            index_type: 索引类型
            vector_store_manager: 向量存储管理器
            embed_model: 嵌入模型
            persist_dir: 持久化目录
            auto_save: 是否自动保存
        """
        self.index_type = index_type
        self.vector_store_manager = vector_store_manager
        self.embed_model = embed_model
        self.persist_dir = persist_dir or os.getenv("INDEX_PERSIST_DIR", "./storage/index")
        self.auto_save = auto_save

        # 创建具体的索引管理器
        self._manager = self._create_manager()

        logger.info(f"Initialized IndexManager with type={index_type.value}")

    def _create_manager(self) -> BaseIndexManager:
        """创建具体的索引管理器"""
        manager_class = self.INDEX_MANAGERS.get(self.index_type)

        if manager_class is None:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        if self.index_type == IndexType.VECTOR:
            return manager_class(
                vector_store_manager=self.vector_store_manager,
                embed_model=self.embed_model,
            )

        return manager_class()

    def build_from_documents(
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
        logger.info(f"Building {self.index_type.value} index from {len(documents)} documents")
        index = self._manager.build_index(documents, **kwargs)

        if self.auto_save:
            self.save()

        return index

    def build_from_nodes(
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
        logger.info(f"Building {self.index_type.value} index from {len(nodes)} nodes")
        index = self._manager.build_index_from_nodes(nodes, **kwargs)

        if self.auto_save:
            self.save()

        return index

    def insert_documents(self, documents: List[Document]) -> None:
        """插入新文档

        Args:
            documents: 文档列表
        """
        self._manager.insert_documents(documents)

        if self.auto_save:
            self.save()

    def insert_nodes(self, nodes: List[TextNode]) -> None:
        """插入新节点

        Args:
            nodes: 节点列表
        """
        self._manager.insert_nodes(nodes)

        if self.auto_save:
            self.save()

    def delete(self, doc_ids: List[str]) -> None:
        """删除文档

        Args:
            doc_ids: 文档 ID 列表
        """
        self._manager.delete(doc_ids)

        if self.auto_save:
            self.save()

    def get_retriever(self, **kwargs) -> BaseRetriever:
        """获取检索器

        Args:
            **kwargs: 检索器参数

        Returns:
            检索器实例
        """
        return self._manager.get_retriever(**kwargs)

    def save(self, path: Optional[str] = None) -> None:
        """保存索引

        Args:
            path: 保存路径，默认使用 persist_dir
        """
        save_path = path or self.persist_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self._manager.save(save_path)
        logger.info(f"Index saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """加载索引

        Args:
            path: 加载路径，默认使用 persist_dir
        """
        load_path = path or self.persist_dir
        self._manager.load(load_path)
        logger.info(f"Index loaded from {load_path}")

    @property
    def index(self) -> Any:
        """获取底层索引"""
        return self._manager.index

    @property
    def document_count(self) -> int:
        """获取文档数量"""
        return self._manager.document_count

    @property
    def node_count(self) -> int:
        """获取节点数量"""
        return self._manager.node_count

    def as_query_engine(self, **kwargs):
        """转换为查询引擎"""
        if hasattr(self._manager, "as_query_engine"):
            return self._manager.as_query_engine(**kwargs)
        raise NotImplementedError(f"Query engine not supported for {self.index_type}")

    def as_chat_engine(self, **kwargs):
        """转换为对话引擎"""
        if hasattr(self._manager, "as_chat_engine"):
            return self._manager.as_chat_engine(**kwargs)
        raise NotImplementedError(f"Chat engine not supported for {self.index_type}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "index_type": self.index_type.value,
            "persist_dir": self.persist_dir,
        }

        if hasattr(self._manager, "get_stats"):
            stats.update(self._manager.get_stats())

        return stats
