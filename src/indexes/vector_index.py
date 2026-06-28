#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量索引管理器

封装 LlamaIndex 的 VectorStoreIndex
"""

import os
from typing import List, Optional, Any

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

from src.indexes.base import BaseIndexManager
from src.storage import VectorStoreManager
from src.core import get_logger

logger = get_logger(__name__)


class VectorIndexManager(BaseIndexManager):
    """向量索引管理器

    封装 VectorStoreIndex 的创建和管理

    Example:
        >>> manager = VectorIndexManager()
        >>> # 从文档构建索引
        >>> index = manager.build_index(documents)
        >>> # 获取检索器
        >>> retriever = manager.get_retriever(similarity_top_k=5)
        >>> nodes = retriever.retrieve("query text")
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        embed_model: Optional[Any] = None,
        show_progress: bool = True,
    ):
        """初始化向量索引管理器

        Args:
            vector_store_manager: 向量存储管理器，如果为 None 则使用内存存储
            embed_model: 嵌入模型
            show_progress: 是否显示进度
        """
        super().__init__()
        self.vector_store_manager = vector_store_manager
        self.embed_model = embed_model
        self.show_progress = show_progress
        self._storage_context: Optional[StorageContext] = None

    def build_index(
        self,
        documents: List[Document],
        **kwargs,
    ) -> VectorStoreIndex:
        """从文档构建向量索引

        Args:
            documents: 文档列表
            **kwargs: 额外参数传递给 VectorStoreIndex.from_documents

        Returns:
            VectorStoreIndex 实例
        """
        logger.info(f"Building vector index from {len(documents)} documents...")

        # 获取存储上下文
        storage_context = self._get_storage_context()

        # 构建索引
        index_params = {
            "documents": documents,
            "storage_context": storage_context,
            "show_progress": self.show_progress,
        }

        if self.embed_model:
            index_params["embed_model"] = self.embed_model

        index_params.update(kwargs)

        self._index = VectorStoreIndex.from_documents(**index_params)
        self._document_count = len(documents)

        logger.info(f"Vector index built successfully with {self._document_count} documents")
        return self._index

    def build_index_from_nodes(
        self,
        nodes: List[TextNode],
        **kwargs,
    ) -> VectorStoreIndex:
        """从节点构建向量索引

        Args:
            nodes: 节点列表
            **kwargs: 额外参数传递给 VectorStoreIndex

        Returns:
            VectorStoreIndex 实例
        """
        logger.info(f"Building vector index from {len(nodes)} nodes...")

        # 获取存储上下文
        storage_context = self._get_storage_context()

        # 构建索引
        index_params = {
            "nodes": nodes,
            "storage_context": storage_context,
            "show_progress": self.show_progress,
        }

        if self.embed_model:
            index_params["embed_model"] = self.embed_model

        index_params.update(kwargs)

        self._index = VectorStoreIndex(**index_params)
        self._node_count = len(nodes)

        logger.info(f"Vector index built successfully with {self._node_count} nodes")
        return self._index

    def insert_documents(self, documents: List[Document]) -> None:
        """插入新文档到索引

        Args:
            documents: 文档列表
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        for doc in documents:
            self._index.insert(doc)

        self._document_count += len(documents)
        logger.info(f"Inserted {len(documents)} documents into index")

    def insert_nodes(self, nodes: List[TextNode]) -> None:
        """插入新节点到索引

        Args:
            nodes: 节点列表
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index_from_nodes() first.")

        for node in nodes:
            self._index.insert_nodes([node])

        self._node_count += len(nodes)
        logger.info(f"Inserted {len(nodes)} nodes into index")

    def delete(self, doc_ids: List[str]) -> None:
        """删除文档

        Args:
            doc_ids: 文档 ID 列表
        """
        if self._index is None:
            raise RuntimeError("Index not built.")

        for doc_id in doc_ids:
            self._index.delete(doc_id)

        logger.info(f"Deleted {len(doc_ids)} documents from index")

    def get_retriever(self, **kwargs) -> BaseRetriever:
        """获取检索器

        Args:
            **kwargs: 检索器参数

        Returns:
            VectorIndexRetriever 实例
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        return self._index.as_retriever(**kwargs)

    def save(self, path: str) -> None:
        """保存索引到磁盘

        Args:
            path: 保存路径
        """
        if self._index is None:
            raise RuntimeError("Index not built.")

        import os
        os.makedirs(path, exist_ok=True)

        self._index.storage_context.persist(persist_dir=path)
        logger.info(f"Index saved to {path}")

    def load(self, path: str) -> None:
        """从磁盘加载索引

        Args:
            path: 加载路径
        """
        storage_context = StorageContext.from_defaults(persist_dir=path)

        self._index = VectorStoreIndex.from_storage_context(storage_context)
        self._storage_context = storage_context
        logger.info(f"Index loaded from {path}")

    def _get_storage_context(self) -> StorageContext:
        """获取存储上下文"""
        if self.vector_store_manager:
            return self.vector_store_manager.storage_context

        # 使用内存存储
        return StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
            vector_store=SimpleVectorStore(),
        )

    def as_query_engine(self, **kwargs):
        """将索引转换为查询引擎

        Args:
            **kwargs: 查询引擎参数

        Returns:
            查询引擎实例
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        return self._index.as_query_engine(**kwargs)

    def as_chat_engine(self, **kwargs):
        """将索引转换为对话引擎

        Args:
            **kwargs: 对话引擎参数

        Returns:
            对话引擎实例
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        return self._index.as_chat_engine(**kwargs)

    def get_stats(self) -> dict:
        """获取索引统计信息

        Returns:
            统计信息字典
        """
        return {
            "document_count": self._document_count,
            "node_count": self._node_count,
            "has_vector_store": self.vector_store_manager is not None,
        }
