#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础检索器包装器

提供检索器的通用接口和功能
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Callable

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever

from src.core import get_logger

logger = get_logger(__name__)


class BaseRetrieverWrapper(ABC):
    """基础检索器包装器抽象类

    提供检索器的标准接口和通用功能
    """

    def __init__(
        self,
        similarity_top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        """初始化检索器

        Args:
            similarity_top_k: 返回的文档数量
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤
            callbacks: 回调函数列表
        """
        self.similarity_top_k = similarity_top_k
        self.similarity_threshold = similarity_threshold
        self.callbacks = callbacks or []
        self._retriever: Optional[BaseRetriever] = None

    @abstractmethod
    def _create_retriever(self) -> BaseRetriever:
        """创建底层检索器

        Returns:
            LlamaIndex 检索器实例
        """
        pass

    def retrieve(self, query: str | QueryBundle) -> List[NodeWithScore]:
        """执行检索

        Args:
            query: 查询字符串或 QueryBundle

        Returns:
            检索结果列表
        """
        if self._retriever is None:
            self._retriever = self._create_retriever()

        # 执行检索
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query

        nodes = self._retriever.retrieve(query_bundle)

        # 应用相似度过滤
        if self.similarity_threshold is not None:
            nodes = self._filter_by_score(nodes)

        # 执行回调
        for callback in self.callbacks:
            try:
                callback(query_bundle, nodes)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")

        logger.debug(f"Retrieved {len(nodes)} nodes for query")
        return nodes

    async def aretrieve(self, query: str | QueryBundle) -> List[NodeWithScore]:
        """异步执行检索

        Args:
            query: 查询字符串或 QueryBundle

        Returns:
            检索结果列表
        """
        if self._retriever is None:
            self._retriever = self._create_retriever()

        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query

        nodes = await self._retriever.aretrieve(query_bundle)

        if self.similarity_threshold is not None:
            nodes = self._filter_by_score(nodes)

        return nodes

    def _filter_by_score(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """根据相似度分数过滤结果

        Args:
            nodes: 节点列表

        Returns:
            过滤后的节点列表
        """
        filtered = [
            node for node in nodes
            if node.score is not None and node.score >= self.similarity_threshold
        ]
        if len(filtered) < len(nodes):
            logger.debug(f"Filtered {len(nodes) - len(filtered)} nodes below threshold")
        return filtered

    @property
    def retriever(self) -> BaseRetriever:
        """获取底层检索器"""
        if self._retriever is None:
            self._retriever = self._create_retriever()
        return self._retriever
