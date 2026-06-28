#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储管理器

封装 ChromaDB 等向量数据库的操作
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores.types import VectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core import get_logger, settings

logger = get_logger(__name__)


class VectorStoreManager:
    """向量存储管理器

    封装向量数据库的创建、连接和管理

    Example:
        >>> manager = VectorStoreManager(
        ...     persist_dir="./storage/chroma",
        ...     collection_name="my_docs"
        ... )
        >>> # 创建存储上下文
        >>> storage_context = manager.get_storage_context()
        >>> # 创建索引
        >>> index = VectorStoreIndex.from_documents(
        ...     documents,
        ...     storage_context=storage_context
        ... )
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "llamaindex_docs",
        embed_dimension: int = 1536,
    ):
        """初始化向量存储管理器

        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
            embed_dimension: 嵌入向量维度
        """
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", "./storage/chroma"
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "llamaindex_docs"
        )
        self.embed_dimension = embed_dimension

        # 确保持久化目录存在
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客户端
        self._chroma_client: Optional[chromadb.Client] = None
        self._chroma_collection: Optional[chromadb.Collection] = None
        self._vector_store: Optional[VectorStore] = None
        self._storage_context: Optional[StorageContext] = None

        self._initialize()

    def _initialize(self) -> None:
        """初始化向量存储"""
        try:
            # 创建 ChromaDB 客户端
            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # 获取或创建集合
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            # 创建 LlamaIndex 向量存储适配器
            self._vector_store = ChromaVectorStore(
                chroma_collection=self._chroma_collection
            )

            # 创建存储上下文
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store
            )

            logger.info(
                f"Initialized ChromaDB at {self.persist_dir}, "
                f"collection: {self.collection_name}, "
                f"existing docs: {self._chroma_collection.count()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    @property
    def vector_store(self) -> VectorStore:
        """获取向量存储实例"""
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized")
        return self._vector_store

    @property
    def storage_context(self) -> StorageContext:
        """获取存储上下文"""
        if self._storage_context is None:
            raise RuntimeError("Storage context not initialized")
        return self._storage_context

    @property
    def chroma_client(self) -> chromadb.Client:
        """获取 ChromaDB 客户端"""
        if self._chroma_client is None:
            raise RuntimeError("ChromaDB client not initialized")
        return self._chroma_client

    @property
    def chroma_collection(self) -> chromadb.Collection:
        """获取 ChromaDB 集合"""
        if self._chroma_collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        return self._chroma_collection

    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        return self._chroma_collection.count() if self._chroma_collection else 0

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """添加节点到向量存储

        Args:
            nodes: 节点列表
        """
        if not nodes:
            return

        self._vector_store.add(nodes)
        logger.info(f"Added {len(nodes)} nodes to vector store")

    def delete_collection(self) -> None:
        """删除集合"""
        if self._chroma_client and self._chroma_collection:
            self._chroma_client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            # 重新初始化
            self._initialize()

    def reset(self) -> None:
        """重置向量存储（清空所有数据）"""
        self.delete_collection()
        logger.warning("Vector store has been reset")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[NodeWithScore]:
        """查询向量存储

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件

        Returns:
            查询结果列表
        """
        results = self._vector_store.query(
            query=query_embedding,
            similarity_top_k=top_k,
            filters=filters,
        )
        return results.nodes if results else []

    def get_stats(self) -> Dict[str, Any]:
        """获取向量存储统计信息

        Returns:
            统计信息字典
        """
        return {
            "persist_dir": self.persist_dir,
            "collection_name": self.collection_name,
            "document_count": self.get_collection_count(),
            "embed_dimension": self.embed_dimension,
        }

    @classmethod
    def create_from_env(cls) -> "VectorStoreManager":
        """从环境变量创建实例

        Returns:
            VectorStoreManager 实例
        """
        return cls(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR"),
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "llamaindex_docs"),
        )
