#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Service

检索增强生成业务逻辑层
"""

import time
from typing import Any, Dict, List, Optional

from src.core import get_logger

logger = get_logger(__name__)


class RAGService:
    """RAG 服务

    负责 RAG 查询和检索业务逻辑
    """

    def __init__(
        self,
        index_manager: Any,
        query_engine: Any,
    ):
        """初始化 RAG 服务

        Args:
            index_manager: 索引管理器
            query_engine: 查询引擎
        """
        self.index_manager = index_manager
        self.query_engine = query_engine

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return getattr(self.index_manager, "index", None) is not None

    def query(
        self,
        query_str: str,
        include_sources: bool = True,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """执行 RAG 查询

        Args:
            query_str: 查询字符串
            include_sources: 是否包含来源信息
            stream: 是否流式响应

        Returns:
            查询结果
        """
        if not self.is_initialized:
            raise ValueError("Index not ready. Please upload documents first.")

        start_time = time.time()

        if stream:
            return {
                "stream": True,
                "query": query_str,
            }

        result = self.query_engine.query_with_sources(query_str)

        sources = None
        if include_sources:
            from src.schemas import SourceNode
            sources = [
                SourceNode(
                    node_id=s["node_id"],
                    text=s["text"],
                    score=s["score"],
                    metadata=s["metadata"],
                )
                for s in result.get("sources", [])
            ]

        latency = time.time() - start_time
        logger.info(f"Query completed in {latency:.2f}s: {query_str[:50]}...")

        return {
            "response": result["response"],
            "sources": sources,
            "source_count": result.get("source_count"),
            "latency": latency,
        }

    def stream_query(self, query_str: str):
        """流式查询

        Args:
            query_str: 查询字符串

        Yields:
            响应文本片段
        """
        for text in self.query_engine.stream_query(query_str):
            yield text


class ChatService:
    """对话服务

    负责多轮对话业务逻辑
    """

    def __init__(self, conversation_manager: Any):
        """初始化对话服务

        Args:
            conversation_manager: 对话管理器
        """
        self.conversation_manager = conversation_manager

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """发送消息

        Args:
            message: 用户消息
            session_id: 会话ID

        Returns:
            对话结果
        """
        if not session_id:
            session_id = self.conversation_manager.create_session()
        elif session_id not in self.conversation_manager.get_session_ids():
            self.conversation_manager.create_session(session_id=session_id)

        response = self.conversation_manager.chat(session_id, message)

        message_count = len(
            self.conversation_manager._sessions[session_id].get_chat_history()
        )

        return {
            "response": response.response,
            "session_id": session_id,
            "message_count": message_count,
        }

    def stream_chat(self, message: str, session_id: Optional[str] = None):
        """流式对话

        Args:
            message: 用户消息
            session_id: 会话ID

        Yields:
            响应文本片段
        """
        if not session_id:
            session_id = self.conversation_manager.create_session()
        elif session_id not in self.conversation_manager.get_session_ids():
            self.conversation_manager.create_session(session_id=session_id)

        for text in self.conversation_manager._sessions[session_id].stream_chat(message):
            yield text

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """删除会话

        Args:
            session_id: 会话ID

        Returns:
            删除结果
        """
        self.conversation_manager.delete_session(session_id)
        return {
            "success": True,
            "message": f"Session {session_id} deleted",
        }

    def get_active_sessions(self) -> int:
        """获取活跃会话数"""
        return self.conversation_manager.get_active_session_count()