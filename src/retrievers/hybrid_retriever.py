#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索器

结合向量检索和关键词检索的优势
"""

from typing import List, Optional, Any, Dict

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import VectorIndexRetriever

from src.retrievers.base import BaseRetrieverWrapper
from src.core import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetrieverWrapper):
    """混合检索器

    结合向量检索和 BM25 关键词检索，提高检索质量

    Example:
        >>> retriever = HybridRetriever(
        ...     index=vector_index,
        ...     similarity_top_k=10,
        ...     vector_weight=0.7,
        ...     keyword_weight=0.3
        ... )
        >>> nodes = retriever.retrieve("query text")
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_keyword: bool = True,
    ):
        """初始化混合检索器

        Args:
            index: 向量索引
            similarity_top_k: 返回的文档数量
            similarity_threshold: 相似度阈值
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            use_keyword: 是否使用关键词检索
        """
        super().__init__(
            similarity_top_k=similarity_top_k,
            similarity_threshold=similarity_threshold,
        )
        self.index = index
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.use_keyword = use_keyword

        self._vector_retriever: Optional[VectorIndexRetriever] = None
        self._keyword_retriever: Optional[Any] = None

        logger.info(
            f"Initialized HybridRetriever with vector_weight={vector_weight}, "
            f"keyword_weight={keyword_weight}"
        )

    def _create_retriever(self) -> BaseRetriever:
        """创建检索器"""
        # 创建向量检索器
        self._vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.similarity_top_k,
        )

        # 尝试创建关键词检索器
        if self.use_keyword:
            try:
                from llama_index.retrievers.bm25 import BM25Retriever

                self._keyword_retriever = BM25Retriever.from_defaults(
                    index=self.index,
                    similarity_top_k=self.similarity_top_k,
                )
                logger.info("BM25 keyword retriever initialized")
            except ImportError:
                logger.warning(
                    "BM25 retriever not available. "
                    "Install llama-index-retrievers-bm25 to enable keyword retrieval."
                )
                self._keyword_retriever = None

        return self._vector_retriever

    def retrieve(self, query: str | QueryBundle) -> List[NodeWithScore]:
        """执行混合检索

        Args:
            query: 查询字符串或 QueryBundle

        Returns:
            混合检索结果
        """
        if self._vector_retriever is None:
            self._create_retriever()

        # 向量检索
        query_bundle = query if isinstance(query, QueryBundle) else QueryBundle(query_str=query)
        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        # 如果没有关键词检索器，直接返回向量结果
        if self._keyword_retriever is None:
            if self.similarity_threshold is not None:
                vector_nodes = self._filter_by_score(vector_nodes)
            return vector_nodes[:self.similarity_top_k]

        # 关键词检索
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        # 融合结果
        merged_nodes = self._merge_results(vector_nodes, keyword_nodes)

        # 应用相似度过滤
        if self.similarity_threshold is not None:
            merged_nodes = self._filter_by_score(merged_nodes)

        return merged_nodes[:self.similarity_top_k]

    def _merge_results(
        self,
        vector_nodes: List[NodeWithScore],
        keyword_nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """融合向量检索和关键词检索结果

        使用加权 Reciprocal Rank Fusion (RRF) 算法

        Args:
            vector_nodes: 向量检索结果
            keyword_nodes: 关键词检索结果

        Returns:
            融合后的结果
        """
        # 使用 node_id 作为唯一标识
        node_scores: Dict[str, float] = {}
        node_map: Dict[str, NodeWithScore] = {}

        # 计算向量检索得分
        for rank, node in enumerate(vector_nodes):
            node_id = node.node.node_id
            # RRF 公式: 1 / (k + rank)
            rrf_score = 1.0 / (60 + rank)
            weighted_score = rrf_score * self.vector_weight
            node_scores[node_id] = node_scores.get(node_id, 0) + weighted_score
            node_map[node_id] = node

        # 计算关键词检索得分
        for rank, node in enumerate(keyword_nodes):
            node_id = node.node.node_id
            rrf_score = 1.0 / (60 + rank)
            weighted_score = rrf_score * self.keyword_weight
            node_scores[node_id] = node_scores.get(node_id, 0) + weighted_score
            if node_id not in node_map:
                node_map[node_id] = node

        # 创建带融合分数的节点列表
        merged = []
        for node_id, score in node_scores.items():
            node = node_map[node_id]
            # 创建新的 NodeWithScore，更新分数
            merged.append(NodeWithScore(
                node=node.node,
                score=score,
            ))

        # 按分数降序排序
        merged.sort(key=lambda x: x.score or 0, reverse=True)

        logger.debug(
            f"Merged {len(vector_nodes)} vector + {len(keyword_nodes)} keyword "
            f"= {len(merged)} unique nodes"
        )

        return merged


class CustomRetriever(BaseRetriever):
    """自定义检索器

    支持添加自定义检索逻辑的 LlamaIndex 检索器

    Example:
        >>> class MyRetriever(CustomRetriever):
        ...     def _retrieve(self, query_bundle):
        ...         nodes = self.vector_retriever.retrieve(query_bundle)
        ...         # 自定义过滤逻辑
        ...         return [n for n in nodes if n.score > 0.5]
    """

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        node_postprocessors: Optional[List[Any]] = None,
    ):
        """初始化自定义检索器

        Args:
            vector_retriever: 基础向量检索器
            node_postprocessors: 节点后处理器列表
        """
        self._vector_retriever = vector_retriever
        self._node_postprocessors = node_postprocessors or []
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """执行检索

        Args:
            query_bundle: 查询包

        Returns:
            检索结果
        """
        nodes = self._vector_retriever.retrieve(query_bundle)

        # 应用后处理器
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(nodes, query_bundle)

        return nodes
