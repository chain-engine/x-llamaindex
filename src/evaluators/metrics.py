#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 评估指标

提供检索和生成质量的评估指标
"""

import time
from typing import List, Optional, Any, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum

from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore

from src.core import get_logger

logger = get_logger(__name__)


class MetricType(str, Enum):
    """评估指标类型"""

    # 检索指标
    PRECISION = "precision"
    RECALL = "recall"
    MRR = "mrr"  # Mean Reciprocal Rank
    NDCG = "ndcg"  # Normalized Discounted Cumulative Gain

    # 生成指标
    FAITHFULNESS = "faithfulness"
    RELEVANCY = "relevancy"
    COHERENCE = "coherence"

    # 性能指标
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""

    # 检索指标
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0

    # 生成指标
    faithfulness: float = 0.0
    relevancy: float = 0.0
    coherence: float = 0.0

    # 性能指标
    latency: float = 0.0
    throughput: float = 0.0

    # 统计
    total_queries: int = 0
    total_documents: int = 0

    # 额外指标
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "faithfulness": self.faithfulness,
            "relevancy": self.relevancy,
            "coherence": self.coherence,
            "latency": self.latency,
            "throughput": self.throughput,
            "total_queries": self.total_queries,
            "total_documents": self.total_documents,
            "custom_metrics": self.custom_metrics,
        }


class RAGEvaluator:
    """RAG 系统评估器

    提供检索和生成质量的评估功能

    Example:
        >>> evaluator = RAGEvaluator()
        >>> # 评估检索结果
        >>> metrics = evaluator.evaluate_retrieval(
        ...     query="What is LlamaIndex?",
        ...     retrieved_nodes=nodes,
        ...     relevant_ids=["doc1", "doc2"]
        ... )
        >>> print(metrics.to_dict())
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
    ):
        """初始化评估器

        Args:
            llm: 语言模型（用于生成指标评估）
            custom_metrics: 自定义指标函数字典
        """
        self.llm = llm
        self.custom_metrics = custom_metrics or {}
        self._metrics_history: List[EvaluationMetrics] = []

        logger.info("Initialized RAGEvaluator")

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_nodes: List[NodeWithScore],
        relevant_ids: Optional[List[str]] = None,
        k: int = 10,
    ) -> EvaluationMetrics:
        """评估检索质量

        Args:
            query: 查询字符串
            retrieved_nodes: 检索到的节点列表
            relevant_ids: 相关文档 ID 列表（如果有标注）
            k: 计算指标时的截断位置

        Returns:
            评估指标
        """
        metrics = EvaluationMetrics()

        if not retrieved_nodes:
            logger.warning("No retrieved nodes to evaluate")
            return metrics

        # 获取检索到的节点 ID
        retrieved_ids = [node.node.node_id for node in retrieved_nodes[:k]]

        if relevant_ids:
            # 计算 Precision@K
            relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
            metrics.precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0

            # 计算 Recall@K
            metrics.recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0

            # 计算 MRR
            for i, node_id in enumerate(retrieved_ids):
                if node_id in relevant_ids:
                    metrics.mrr = 1.0 / (i + 1)
                    break

            # 计算 NDCG
            metrics.ndcg = self._calculate_ndcg(retrieved_ids, relevant_ids, k)

        # 计算平均相似度分数
        scores = [node.score for node in retrieved_nodes if node.score is not None]
        if scores:
            metrics.relevancy = sum(scores) / len(scores)

        metrics.total_queries = 1
        metrics.total_documents = len(retrieved_nodes)

        self._metrics_history.append(metrics)
        return metrics

    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """计算 NDCG@K

        Args:
            retrieved_ids: 检索到的 ID 列表
            relevant_ids: 相关 ID 列表
            k: 截断位置

        Returns:
            NDCG 值
        """
        import math

        # 计算 DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)

        # 计算 IDCG（理想情况下的 DCG）
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / math.log2(i + 2)

        # 计算 NDCG
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_generation(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> EvaluationMetrics:
        """评估生成质量

        Args:
            query: 查询字符串
            response: 生成的响应
            context: 上下文（检索到的内容）
            reference: 参考答案（如果有）

        Returns:
            评估指标
        """
        metrics = EvaluationMetrics()

        if self.llm:
            # 使用 LLM 评估
            metrics.faithfulness = self._evaluate_faithfulness(response, context)
            metrics.relevancy = self._evaluate_relevancy(query, response)
            metrics.coherence = self._evaluate_coherence(response)
        else:
            # 使用简单指标
            if context:
                metrics.faithfulness = self._simple_faithfulness(response, context)
            metrics.relevancy = self._simple_relevancy(query, response)
            metrics.coherence = self._simple_coherence(response)

        metrics.total_queries = 1
        self._metrics_history.append(metrics)
        return metrics

    def _evaluate_faithfulness(self, response: str, context: Optional[str]) -> float:
        """使用 LLM 评估忠实度"""
        if not context or not self.llm:
            return 0.0

        try:
            prompt = f"""Evaluate whether the following response is faithful to the given context.
Context: {context[:1000]}
Response: {response[:500]}
Score from 0 to 1, where 1 means fully faithful.
Return only the score as a number."""

            result = self.llm.complete(prompt)
            score = float(result.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Failed to evaluate faithfulness: {e}")
            return 0.0

    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """使用 LLM 评估相关性"""
        if not self.llm:
            return self._simple_relevancy(query, response)

        try:
            prompt = f"""Evaluate whether the following response is relevant to the query.
Query: {query}
Response: {response[:500]}
Score from 0 to 1, where 1 means highly relevant.
Return only the score as a number."""

            result = self.llm.complete(prompt)
            score = float(result.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Failed to evaluate relevancy: {e}")
            return self._simple_relevancy(query, response)

    def _evaluate_coherence(self, response: str) -> float:
        """使用 LLM 评估连贯性"""
        if not self.llm:
            return self._simple_coherence(response)

        try:
            prompt = f"""Evaluate the coherence and readability of the following response.
Response: {response[:500]}
Score from 0 to 1, where 1 means highly coherent.
Return only the score as a number."""

            result = self.llm.complete(prompt)
            score = float(result.text.strip())
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Failed to evaluate coherence: {e}")
            return self._simple_coherence(response)

    def _simple_faithfulness(self, response: str, context: str) -> float:
        """简单的忠实度评估（基于词汇重叠）"""
        if not context:
            return 0.0

        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        overlap = len(response_words & context_words)
        return overlap / len(response_words) if response_words else 0.0

    def _simple_relevancy(self, query: str, response: str) -> float:
        """简单的相关性评估（基于词汇重叠）"""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        overlap = len(query_words & response_words)
        return overlap / len(query_words) if query_words else 0.0

    def _simple_coherence(self, response: str) -> float:
        """简单的连贯性评估"""
        # 基于句子长度和标点符号的简单评估
        sentences = response.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # 检查句子长度的一致性
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)

        # 惩罚过短或过长的句子
        if avg_length < 5 or avg_length > 50:
            return 0.5

        return 0.7

    def evaluate_latency(
        self,
        latency_seconds: float,
        num_queries: int = 1,
    ) -> EvaluationMetrics:
        """记录性能指标

        Args:
            latency_seconds: 延迟时间（秒）
            num_queries: 查询数量

        Returns:
            评估指标
        """
        metrics = EvaluationMetrics()
        metrics.latency = latency_seconds
        metrics.throughput = num_queries / latency_seconds if latency_seconds > 0 else 0
        metrics.total_queries = num_queries

        self._metrics_history.append(metrics)
        return metrics

    def get_aggregated_metrics(self) -> EvaluationMetrics:
        """获取聚合的评估指标

        聚合规则：
        - 检索指标（precision/recall/mrr/ndcg/relevancy）：取平均值
        - 生成指标（faithfulness/coherence）：取平均值
        - 性能指标（latency）：总和 / 总查询数 = 平均延迟
        - 统计（total_queries/total_documents）：求和

        Returns:
            聚合后的评估指标
        """
        if not self._metrics_history:
            return EvaluationMetrics()

        # 分离不同类型的指标
        retrieval_metrics = []
        generation_metrics = []
        latency_metrics = []

        for metrics in self._metrics_history:
            if metrics.total_queries > 0 and metrics.total_documents > 0:
                retrieval_metrics.append(metrics)
            if metrics.faithfulness > 0 or metrics.coherence > 0:
                generation_metrics.append(metrics)
            if metrics.latency > 0:
                latency_metrics.append(metrics)

        aggregated = EvaluationMetrics()

        # 聚合检索指标
        if retrieval_metrics:
            n_ret = len(retrieval_metrics)
            for metrics in retrieval_metrics:
                aggregated.precision += metrics.precision / n_ret
                aggregated.recall += metrics.recall / n_ret
                aggregated.mrr += metrics.mrr / n_ret
                aggregated.ndcg += metrics.ndcg / n_ret
                aggregated.relevancy += metrics.relevancy / n_ret

        # 聚合生成指标
        if generation_metrics:
            n_gen = len(generation_metrics)
            for metrics in generation_metrics:
                aggregated.faithfulness += metrics.faithfulness / n_gen
                aggregated.coherence += metrics.coherence / n_gen

        # 聚合性能指标（latency 是平均值，throughput 是总和）
        total_latency = sum(m.total_queries * m.latency for m in latency_metrics) if latency_metrics else 0
        total_throughput_queries = sum(m.total_queries for m in latency_metrics) if latency_metrics else 0
        total_throughput_time = sum(m.latency for m in latency_metrics) if latency_metrics else 0

        aggregated.latency = total_latency / total_throughput_queries if total_throughput_queries > 0 else 0
        aggregated.throughput = total_throughput_queries / total_throughput_time if total_throughput_time > 0 else 0

        # 聚合统计
        aggregated.total_queries = sum(m.total_queries for m in self._metrics_history)
        aggregated.total_documents = sum(m.total_documents for m in self._metrics_history)

        return aggregated

    def clear_history(self) -> None:
        """清除历史记录"""
        self._metrics_history.clear()
        logger.info("Evaluation metrics history cleared")

    def get_history(self) -> List[EvaluationMetrics]:
        """获取历史记录

        Returns:
            历史评估指标列表
        """
        return self._metrics_history.copy()
