#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重排序器

对检索结果进行重新排序以提高相关性
"""

from typing import List, Optional, Any, Dict
from enum import Enum

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from src.core import get_logger

logger = get_logger(__name__)


class RerankerModel(str, Enum):
    """重排序模型类型"""

    CROSS_ENCODER = "cross-encoder"
    COHERE = "cohere"
    LLMBASED = "llm-based"
    SIMPLE = "simple"


class Reranker:
    """重排序器

    对检索结果进行重新排序，提高最相关结果的排名

    Example:
        >>> reranker = Reranker(model_type=RerankerModel.CROSS_ENCODER)
        >>> reranked_nodes = reranker.rerank(query, nodes)
    """

    def __init__(
        self,
        model_type: RerankerModel = RerankerModel.SIMPLE,
        top_n: int = 5,
        model_name: Optional[str] = None,
    ):
        """初始化重排序器

        Args:
            model_type: 重排序模型类型
            top_n: 返回的结果数量
            model_name: 模型名称（对于某些模型类型）
        """
        self.model_type = model_type
        self.top_n = top_n
        self.model_name = model_name
        self._postprocessor: Optional[BaseNodePostprocessor] = None

        self._initialize()
        logger.info(f"Initialized Reranker with model_type={model_type.value}")

    def _initialize(self) -> None:
        """初始化重排序器"""
        try:
            if self.model_type == RerankerModel.CROSS_ENCODER:
                self._init_cross_encoder()
            elif self.model_type == RerankerModel.COHERE:
                self._init_cohere()
            elif self.model_type == RerankerModel.LLMBASED:
                self._init_llm_based()
            else:
                # Simple reranker 不需要初始化
                self._postprocessor = None

        except Exception as e:
            logger.warning(f"Failed to initialize {self.model_type.value} reranker: {e}")
            logger.info("Falling back to simple reranker")
            self._postprocessor = None

    def _init_cross_encoder(self) -> None:
        """初始化 Cross-Encoder 重排序器"""
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank

            # 这里使用 Cohere 作为示例，实际可以使用其他 Cross-Encoder
            self._postprocessor = CohereRerank(
                top_n=self.top_n,
            )
        except ImportError:
            # 尝试使用 SentenceTransformer
            try:
                from llama_index.postprocessor.sentence_transformer_rerank import (
                    SentenceTransformerRerank,
                )

                model = self.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self._postprocessor = SentenceTransformerRerank(
                    model=model,
                    top_n=self.top_n,
                )
                logger.info(f"Initialized SentenceTransformer reranker with model={model}")
            except ImportError:
                raise ImportError(
                    "No reranker backend available. "
                    "Install sentence-transformers or cohere package."
                )

    def _init_cohere(self) -> None:
        """初始化 Cohere 重排序器"""
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank

            self._postprocessor = CohereRerank(
                top_n=self.top_n,
                model_name=self.model_name or "rerank-english-v2.0",
            )
        except ImportError:
            raise ImportError("Install llama-index-postprocessor-cohere-rerank")

    def _init_llm_based(self) -> None:
        """初始化 LLM 重排序器"""
        try:
            from llama_index.postprocessor.llm_rerank import LLMRerank

            self._postprocessor = LLMRerank(
                top_n=self.top_n,
            )
        except ImportError:
            raise ImportError("Install llama-index-postprocessor-llm-rerank")

    def rerank(
        self,
        query: str | QueryBundle,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """执行重排序

        Args:
            query: 查询
            nodes: 待重排序的节点列表

        Returns:
            重排序后的节点列表
        """
        if not nodes:
            return []

        query_bundle = query if isinstance(query, QueryBundle) else QueryBundle(query_str=query)

        # 如果有后处理器，使用它
        if self._postprocessor is not None:
            try:
                reranked = self._postprocessor.postprocess_nodes(nodes, query_bundle)
                logger.debug(f"Reranked {len(nodes)} nodes -> {len(reranked)} nodes")
                return reranked[:self.top_n]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using simple reranker")

        # 使用简单重排序
        return self._simple_rerank(query_bundle, nodes)

    def _simple_rerank(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """简单重排序

        基于关键词匹配进行重排序

        Args:
            query_bundle: 查询包
            nodes: 节点列表

        Returns:
            重排序后的节点列表
        """
        query_text = query_bundle.query_str.lower()
        query_words = set(query_text.split())

        scored_nodes: List[tuple] = []

        for node in nodes:
            node_text = node.node.text.lower()
            node_words = set(node_text.split())

            # 计算关键词重叠率
            overlap = len(query_words & node_words)
            overlap_ratio = overlap / len(query_words) if query_words else 0

            # 综合原始分数和关键词匹配
            original_score = node.score or 0
            combined_score = 0.7 * original_score + 0.3 * overlap_ratio

            scored_nodes.append((node, combined_score))

        # 按综合分数排序
        scored_nodes.sort(key=lambda x: x[1], reverse=True)

        # 创建新的带分数节点
        reranked = []
        for node, score in scored_nodes[:self.top_n]:
            reranked.append(NodeWithScore(
                node=node.node,
                score=score,
            ))

        logger.debug(f"Simple rerank: {len(nodes)} nodes -> {len(reranked)} nodes")
        return reranked

    @property
    def postprocessor(self) -> Optional[BaseNodePostprocessor]:
        """获取底层后处理器"""
        return self._postprocessor


class DiversityReranker:
    """多样性重排序器

    在保持相关性的同时增加结果的多样性

    Example:
        >>> reranker = DiversityReranker(diversity_threshold=0.8)
        >>> diverse_nodes = reranker.rerank(query, nodes)
    """

    def __init__(
        self,
        diversity_threshold: float = 0.8,
        top_n: int = 5,
    ):
        """初始化多样性重排序器

        Args:
            diversity_threshold: 多样性阈值（0-1）
            top_n: 返回的结果数量
        """
        self.diversity_threshold = diversity_threshold
        self.top_n = top_n

    def rerank(
        self,
        query: str | QueryBundle,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """执行多样性重排序

        Args:
            query: 查询
            nodes: 节点列表

        Returns:
            多样性重排序后的节点列表
        """
        if len(nodes) <= self.top_n:
            return nodes

        query_bundle = query if isinstance(query, QueryBundle) else QueryBundle(query_str=query)

        # 按原始分数排序
        sorted_nodes = sorted(nodes, key=lambda x: x.score or 0, reverse=True)

        selected: List[NodeWithScore] = []
        remaining = list(sorted_nodes)

        while len(selected) < self.top_n and remaining:
            # 选择剩余中分数最高的
            best = remaining.pop(0)
            selected.append(best)

            # 过滤与已选择节点太相似的
            remaining = [
                node for node in remaining
                if not self._is_too_similar(best.node.text, node.node.text)
            ]

        logger.debug(f"Diversity rerank: {len(nodes)} nodes -> {len(selected)} nodes")
        return selected

    def _is_too_similar(self, text1: str, text2: str) -> bool:
        """检查两个文本是否太相似

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            是否太相似
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0

        return jaccard > self.diversity_threshold
