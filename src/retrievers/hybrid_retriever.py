#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索器

结合向量检索和关键词检索的优势
"""

from typing import List, Optional, Any, Dict, Union

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.retrievers import VectorIndexRetriever

from src.core import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器

    结合向量检索和 BM25 关键词检索，使用加权 Reciprocal Rank Fusion (RRF) 算法融合结果

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
        index: Any,
        similarity_top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
        use_keyword: bool = True,
    ):
        """初始化混合检索器

        Args:
            index: 向量索引
            similarity_top_k: 返回的文档数量
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            rrf_k: RRF 算法中的常数参数，默认 60
            use_keyword: 是否使用关键词检索
        """
        super().__init__()

        if vector_weight + keyword_weight != 1.0:
            # 自动归一化权重
            total = vector_weight + keyword_weight
            vector_weight = vector_weight / total
            keyword_weight = keyword_weight / total
            logger.info(f"Weights normalized: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}")

        self.index = index
        self.similarity_top_k = similarity_top_k
        self.similarity_threshold = similarity_threshold
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.use_keyword = use_keyword

        self._vector_retriever: Optional[VectorIndexRetriever] = None
        self._keyword_retriever: Optional[Any] = None
        self._bm25_available: bool = False

        self._initialize_retrievers()

        logger.info(
            f"Initialized HybridRetriever with vector_weight={vector_weight:.2f}, "
            f"keyword_weight={keyword_weight:.2f}, rrf_k={rrf_k}, "
            f"bm25_available={self._bm25_available}"
        )

    def _initialize_retrievers(self) -> None:
        """初始化向量检索器和关键词检索器"""
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
                self._bm25_available = True
                logger.info("BM25 keyword retriever initialized")
            except ImportError:
                logger.warning(
                    "BM25 retriever not available. "
                    "Install llama-index-retrievers-bm25 to enable keyword retrieval. "
                    "Falling back to vector-only retrieval."
                )
                self._keyword_retriever = None
                self._bm25_available = False

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """执行混合检索（内部方法）"""
        # 向量检索
        vector_nodes = self._vector_retriever.retrieve(query_bundle)

        # 如果没有关键词检索器，直接返回向量结果
        if not self._bm25_available or self._keyword_retriever is None:
            return self._apply_threshold(vector_nodes)[:self.similarity_top_k]

        # 关键词检索
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        # 融合结果
        merged_nodes = self._fuse_results(vector_nodes, keyword_nodes)

        # 应用相似度过滤
        merged_nodes = self._apply_threshold(merged_nodes)

        return merged_nodes[:self.similarity_top_k]

    def _fuse_results(
        self,
        vector_nodes: List[NodeWithScore],
        keyword_nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """融合向量检索和关键词检索结果

        使用加权 Reciprocal Rank Fusion (RRF) 算法：
        score(node) = w_vector * RRF_vector + w_keyword * RRF_keyword
        其中 RRF = 1 / (k + rank)

        Args:
            vector_nodes: 向量检索结果
            keyword_nodes: 关键词检索结果

        Returns:
            融合后的结果（按融合分数降序排列）
        """
        node_scores: Dict[str, float] = {}
        node_map: Dict[str, NodeWithScore] = {}

        # 归一化向量检索分数到 [0, 1] 范围
        vector_max = max(n.score or 0 for n in vector_nodes) if vector_nodes else 1.0
        vector_max = max(vector_max, 1e-9)  # 避免除零

        # 归一化关键词检索分数到 [0, 1] 范围
        keyword_max = max(n.score or 0 for n in keyword_nodes) if keyword_nodes else 1.0
        keyword_max = max(keyword_max, 1e-9)  # 避免除零

        # 计算向量检索的加权 RRF 分数
        for rank, node in enumerate(vector_nodes):
            node_id = node.node.node_id
            rrf_score = 1.0 / (self.rrf_k + rank)
            # 归一化原始相似度分数并加权
            normalized_score = (node.score or 0) / vector_max
            combined_score = (
                self.vector_weight * rrf_score +
                self.vector_weight * normalized_score * 0.1  # 保留原始分数的影响
            )
            node_scores[node_id] = node_scores.get(node_id, 0) + combined_score
            node_map[node_id] = node

        # 计算关键词检索的加权 RRF 分数
        for rank, node in enumerate(keyword_nodes):
            node_id = node.node.node_id
            rrf_score = 1.0 / (self.rrf_k + rank)
            # 归一化原始 BM25 分数并加权
            normalized_score = (node.score or 0) / keyword_max
            combined_score = (
                self.keyword_weight * rrf_score +
                self.keyword_weight * normalized_score * 0.1
            )
            node_scores[node_id] = node_scores.get(node_id, 0) + combined_score
            if node_id not in node_map:
                node_map[node_id] = node

        # 创建带融合分数的节点列表
        merged = []
        for node_id, score in node_scores.items():
            node = node_map[node_id]
            merged.append(NodeWithScore(
                node=node.node,
                score=score,
            ))

        # 按分数降序排序
        merged.sort(key=lambda x: x.score or 0, reverse=True)

        logger.debug(
            f"Fused {len(vector_nodes)} vector + {len(keyword_nodes)} keyword "
            f"= {len(merged)} unique nodes"
        )

        return merged

    def _apply_threshold(
        self,
        nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """应用相似度阈值过滤"""
        if self.similarity_threshold is None:
            return nodes

        filtered = [
            node for node in nodes
            if node.score is not None and node.score >= self.similarity_threshold
        ]
        if len(filtered) < len(nodes):
            logger.debug(
                f"Filtered {len(nodes) - len(filtered)} nodes below threshold "
                f"{self.similarity_threshold}"
            )
        return filtered

    def retrieve(
        self,
        query: Union[str, QueryBundle],
    ) -> List[NodeWithScore]:
        """执行混合检索"""
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query

        return self._retrieve(query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """异步执行混合检索"""
        # 当前实现为同步，如果需要可以改为异步
        return self._retrieve(query_bundle)

    async def aretrieve(
        self,
        query: Union[str, QueryBundle],
    ) -> List[NodeWithScore]:
        """异步执行混合检索"""
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query

        return await self._aretrieve(query_bundle)

    def update_weights(
        self,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
    ) -> None:
        """更新检索权重

        Args:
            vector_weight: 新的向量检索权重
            keyword_weight: 新的关键词检索权重
        """
        if vector_weight is not None:
            self.vector_weight = vector_weight
        if keyword_weight is not None:
            self.keyword_weight = keyword_weight

        # 归一化
        total = self.vector_weight + self.keyword_weight
        if total > 0:
            self.vector_weight /= total
            self.keyword_weight /= total

        logger.info(
            f"Updated weights: vector={self.vector_weight:.2f}, "
            f"keyword={self.keyword_weight:.2f}"
        )


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
