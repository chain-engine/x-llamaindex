#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 查询引擎

封装 LlamaIndex 的查询引擎，提供企业级查询功能
"""

from typing import Optional, Any, Dict, List, Callable
from enum import Enum

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.base.response.schema import Response, StreamingResponse
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openai import OpenAI

from src.retrievers import HybridRetriever, Reranker
from src.core import get_logger

logger = get_logger(__name__)


class QueryMode(str, Enum):
    """查询模式"""

    DEFAULT = "default"
    COMPACT = "compact"
    REFINE = "refine"
    TREE_SUMMARIZE = "tree_summarize"
    SIMPLE = "simple"


class RAGQueryEngine:
    """RAG 查询引擎

    封装查询引擎的创建和查询执行，支持流式响应和多种查询模式

    Example:
        >>> engine = RAGQueryEngine(index=vector_index)
        >>> # 同步查询
        >>> response = engine.query("What is LlamaIndex?")
        >>> print(response.response)
        >>> # 流式查询
        >>> for text in engine.stream_query("What is RAG?"):
        ...     print(text, end="")
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: Optional[Any] = None,
        similarity_top_k: int = 5,
        response_mode: QueryMode = QueryMode.COMPACT,
        use_hybrid_retrieval: bool = False,
        use_reranker: bool = False,
        streaming: bool = False,
        verbose: bool = False,
    ):
        """初始化查询引擎

        Args:
            index: 向量索引
            llm: 语言模型
            similarity_top_k: 检索数量
            response_mode: 响应模式
            use_hybrid_retrieval: 是否使用混合检索
            use_reranker: 是否使用重排序
            streaming: 是否启用流式响应
            verbose: 是否显示详细信息
        """
        self.index = index
        self.llm = llm or self._create_default_llm()
        self.similarity_top_k = similarity_top_k
        self.response_mode = response_mode
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.use_reranker = use_reranker
        self.streaming = streaming
        self.verbose = verbose

        self._query_engine: Optional[BaseQueryEngine] = None
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None

        self._query_count = 0
        self._total_latency = 0.0

        logger.info(
            f"Initialized RAGQueryEngine with mode={response_mode.value}, "
            f"hybrid={use_hybrid_retrieval}, rerank={use_reranker}"
        )

    def _create_default_llm(self) -> OpenAI:
        """创建默认的 LLM"""
        import os
        return OpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )

    def _create_query_engine(self) -> BaseQueryEngine:
        """创建查询引擎"""
        engine_params: Dict[str, Any] = {
            "llm": self.llm,
            "similarity_top_k": self.similarity_top_k,
            "response_mode": self._get_response_mode(),
            "streaming": self.streaming,
            "verbose": self.verbose,
        }

        # 配置检索器
        if self.use_hybrid_retrieval:
            self._retriever = HybridRetriever(
                index=self.index,
                similarity_top_k=self.similarity_top_k,
            )
            engine_params["retriever"] = self._retriever.retriever

        # 配置重排序
        if self.use_reranker:
            from src.retrievers.reranker import RerankerModel
            self._reranker = Reranker(
                model_type=RerankerModel.SIMPLE,
                top_n=self.similarity_top_k,
            )
            # 添加为 node_postprocessor
            if self._reranker.postprocessor:
                engine_params["node_postprocessors"] = [self._reranker.postprocessor]

        return self.index.as_query_engine(**engine_params)

    def _get_response_mode(self) -> ResponseMode:
        """获取响应模式"""
        mode_mapping = {
            QueryMode.DEFAULT: ResponseMode.COMPACT,
            QueryMode.COMPACT: ResponseMode.COMPACT,
            QueryMode.REFINE: ResponseMode.REFINE,
            QueryMode.TREE_SUMMARIZE: ResponseMode.TREE_SUMMARIZE,
            QueryMode.SIMPLE: ResponseMode.NO_TEXT,
        }
        return mode_mapping.get(self.response_mode, ResponseMode.COMPACT)

    @property
    def query_engine(self) -> BaseQueryEngine:
        """获取查询引擎实例"""
        if self._query_engine is None:
            self._query_engine = self._create_query_engine()
        return self._query_engine

    def query(self, query_str: str) -> Response:
        """执行查询

        Args:
            query_str: 查询字符串

        Returns:
            查询响应
        """
        import time
        start_time = time.time()

        logger.info(f"Processing query: {query_str[:100]}...")

        response = self.query_engine.query(query_str)

        latency = time.time() - start_time
        self._query_count += 1
        self._total_latency += latency

        if self.verbose:
            logger.info(f"Query completed in {latency:.2f}s")

        return response

    def query_with_sources(self, query_str: str) -> Dict[str, Any]:
        """执行查询并返回来源信息

        Args:
            query_str: 查询字符串

        Returns:
            包含响应和来源信息的字典
        """
        response = self.query(query_str)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "node_id": node.node.node_id,
                "text": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                "score": node.score,
                "metadata": node.node.metadata,
            })

        return {
            "response": response.response,
            "sources": sources,
            "source_count": len(sources),
        }

    def stream_query(self, query_str: str):
        """执行流式查询

        Args:
            query_str: 查询字符串

        Yields:
            响应文本片段
        """
        if not self.streaming:
            # 如果未启用流式，创建临时流式引擎
            engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=self.similarity_top_k,
                streaming=True,
            )
        else:
            engine = self.query_engine

        logger.info(f"Processing streaming query: {query_str[:100]}...")

        response = engine.query(query_str)

        for text in response.response_gen:
            yield text

    def batch_query(self, query_strs: List[str]) -> List[Response]:
        """批量查询

        Args:
            query_strs: 查询字符串列表

        Returns:
            响应列表
        """
        responses = []
        for query_str in query_strs:
            response = self.query(query_str)
            responses.append(response)

        logger.info(f"Batch query completed: {len(query_strs)} queries")
        return responses

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        avg_latency = (
            self._total_latency / self._query_count
            if self._query_count > 0
            else 0
        )

        return {
            "query_count": self._query_count,
            "total_latency": self._total_latency,
            "average_latency": avg_latency,
            "response_mode": self.response_mode.value,
            "similarity_top_k": self.similarity_top_k,
            "streaming_enabled": self.streaming,
            "hybrid_retrieval_enabled": self.use_hybrid_retrieval,
            "reranker_enabled": self.use_reranker,
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._query_count = 0
        self._total_latency = 0.0
        logger.info("Query engine stats reset")

    def update_config(
        self,
        similarity_top_k: Optional[int] = None,
        response_mode: Optional[QueryMode] = None,
        streaming: Optional[bool] = None,
    ) -> None:
        """更新配置

        更新配置后需要重新创建查询引擎

        Args:
            similarity_top_k: 新的检索数量
            response_mode: 新的响应模式
            streaming: 是否启用流式
        """
        if similarity_top_k is not None:
            self.similarity_top_k = similarity_top_k
        if response_mode is not None:
            self.response_mode = response_mode
        if streaming is not None:
            self.streaming = streaming

        # 强制重新创建查询引擎
        self._query_engine = None
        logger.info("Query engine config updated, will recreate on next query")
