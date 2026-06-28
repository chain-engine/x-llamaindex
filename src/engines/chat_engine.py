#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 对话引擎

封装 LlamaIndex 的对话引擎，支持多轮对话
"""

from typing import Optional, Any, Dict, List
from enum import Enum

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine, ContextChatEngine
from llama_index.llms.openai import OpenAI

from src.core import get_logger

logger = get_logger(__name__)


class ChatMode(str, Enum):
    """对话模式"""

    SIMPLE = "simple"  # 简单对话，不使用上下文
    CONTEXT = "context"  # 基于上下文的对话
    CONDENSE = "condense"  # 压缩历史对话
    REACT = "react"  # ReAct 模式


class RAGChatEngine:
    """RAG 对话引擎

    封装对话引擎的创建和多轮对话功能

    Example:
        >>> engine = RAGChatEngine(index=vector_index)
        >>> # 单轮对话
        >>> response = engine.chat("What is LlamaIndex?")
        >>> # 多轮对话
        >>> engine.chat("Tell me more about RAG.")
        >>> engine.chat("What are its advantages?")
        >>> # 获取对话历史
        >>> history = engine.get_chat_history()
    """

    def __init__(
        self,
        index: Optional[VectorStoreIndex] = None,
        llm: Optional[Any] = None,
        chat_mode: ChatMode = ChatMode.CONTEXT,
        similarity_top_k: int = 5,
        system_prompt: Optional[str] = None,
        streaming: bool = False,
        verbose: bool = False,
    ):
        """初始化对话引擎

        Args:
            index: 向量索引（CONTEXT 模式需要）
            llm: 语言模型
            chat_mode: 对话模式
            similarity_top_k: 检索数量
            system_prompt: 系统提示词
            streaming: 是否启用流式响应
            verbose: 是否显示详细信息
        """
        self.index = index
        self.llm = llm or self._create_default_llm()
        self.chat_mode = chat_mode
        self.similarity_top_k = similarity_top_k
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.streaming = streaming
        self.verbose = verbose

        # 使用 Any 避免不同 llama-index 版本 BaseChatEngine 路径不一致导致导入失败
        self._chat_engine: Optional[Any] = None
        self._message_count = 0

        logger.info(f"Initialized RAGChatEngine with mode={chat_mode.value}")

    def _create_default_llm(self) -> OpenAI:
        """创建默认的 LLM"""
        import os
        return OpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
        )

    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个专业的 RAG（检索增强生成）助手。
你的任务是基于提供的知识库内容回答用户问题。
请遵循以下原则：
1. 优先使用知识库中的信息回答问题
2. 如果知识库中没有相关信息，请诚实地告知用户
3. 回答要准确、简洁、有帮助
4. 如果需要，可以引用来源信息"""

    def _create_chat_engine(self) -> Any:
        """创建对话引擎"""
        if self.chat_mode == ChatMode.SIMPLE:
            return SimpleChatEngine.from_defaults(
                llm=self.llm,
                system_prompt=self.system_prompt,
            )

        # 其他模式需要索引
        if self.index is None:
            logger.warning("Index not provided, falling back to simple chat mode")
            return SimpleChatEngine.from_defaults(
                llm=self.llm,
                system_prompt=self.system_prompt,
            )

        engine_params: Dict[str, Any] = {
            "llm": self.llm,
            "system_prompt": self.system_prompt,
            "similarity_top_k": self.similarity_top_k,
            "verbose": self.verbose,
        }

        if self.chat_mode == ChatMode.CONTEXT:
            return self.index.as_chat_engine(
                chat_mode="context",
                **engine_params,
            )
        elif self.chat_mode == ChatMode.CONDENSE:
            return self.index.as_chat_engine(
                chat_mode="condense_question",
                **engine_params,
            )
        elif self.chat_mode == ChatMode.REACT:
            return self.index.as_chat_engine(
                chat_mode="react",
                **engine_params,
            )
        else:
            return self.index.as_chat_engine(**engine_params)

    @property
    def chat_engine(self) -> Any:
        """获取对话引擎实例"""
        if self._chat_engine is None:
            self._chat_engine = self._create_chat_engine()
        return self._chat_engine

    def chat(self, message: str) -> Any:
        """发送消息

        Args:
            message: 用户消息

        Returns:
            助手响应
        """
        logger.info(f"Processing chat message: {message[:100]}...")

        response = self.chat_engine.chat(message)
        self._message_count += 1

        return response

    def stream_chat(self, message: str):
        """流式发送消息

        Args:
            message: 用户消息

        Yields:
            响应文本片段
        """
        logger.info(f"Processing streaming chat message: {message[:100]}...")

        # 确保使用流式
        if not self.streaming:
            engine = self.index.as_chat_engine(
                llm=self.llm,
                system_prompt=self.system_prompt,
                similarity_top_k=self.similarity_top_k,
                streaming=True,
            ) if self.index else SimpleChatEngine.from_defaults(
                llm=self.llm,
                system_prompt=self.system_prompt,
                streaming=True,
            )
        else:
            engine = self.chat_engine

        response = engine.chat(message)
        self._message_count += 1

        for text in response.response_gen:
            yield text

    def reset(self) -> None:
        """重置对话历史"""
        self.chat_engine.reset()
        self._message_count = 0
        logger.info("Chat history reset")

    def get_chat_history(self) -> List[Any]:
        """获取对话历史

        Returns:
            对话消息列表
        """
        # 获取对话历史
        history = self.chat_engine.chat_history if hasattr(self.chat_engine, "chat_history") else []
        return list(history) if history else []

    def get_chat_history_str(self) -> str:
        """获取对话历史的字符串表示

        Returns:
            格式化的对话历史
        """
        history = self.get_chat_history()
        formatted = []

        for msg in history:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            formatted.append(f"[{role.upper()}]: {msg.content}")

        return "\n".join(formatted)

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词

        设置后会重新创建对话引擎

        Args:
            prompt: 新的系统提示词
        """
        self.system_prompt = prompt
        self._chat_engine = None
        logger.info("System prompt updated, chat engine will be recreated")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "message_count": self._message_count,
            "chat_mode": self.chat_mode.value,
            "similarity_top_k": self.similarity_top_k,
            "streaming_enabled": self.streaming,
            "has_index": self.index is not None,
            "history_length": len(self.get_chat_history()),
        }


class ConversationManager:
    """对话管理器

    管理多个独立的对话会话

    Example:
        >>> manager = ConversationManager(index=vector_index)
        >>> # 创建新会话
        >>> session_id = manager.create_session()
        >>> # 发送消息到特定会话
        >>> response = manager.chat(session_id, "Hello!")
    """

    def __init__(
        self,
        index: Optional[VectorStoreIndex] = None,
        llm: Optional[Any] = None,
        max_sessions: int = 100,
    ):
        """初始化对话管理器

        Args:
            index: 向量索引
            llm: 语言模型
            max_sessions: 最大会话数
        """
        self.index = index
        self.llm = llm
        self.max_sessions = max_sessions
        self._sessions: Dict[str, RAGChatEngine] = {}

        logger.info(f"Initialized ConversationManager with max_sessions={max_sessions}")

    def create_session(
        self,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """创建新会话

        Args:
            session_id: 会话ID，如果不提供则自动生成
            system_prompt: 系统提示词

        Returns:
            会话ID
        """
        import uuid

        session_id = session_id or str(uuid.uuid4())

        # 检查是否超过最大会话数
        if len(self._sessions) >= self.max_sessions:
            # 删除最旧的会话
            oldest_id = next(iter(self._sessions))
            del self._sessions[oldest_id]
            logger.warning(f"Removed oldest session {oldest_id} due to max_sessions limit")

        engine = RAGChatEngine(
            index=self.index,
            llm=self.llm,
            system_prompt=system_prompt,
        )
        self._sessions[session_id] = engine

        logger.info(f"Created new chat session: {session_id}")
        return session_id

    def chat(self, session_id: str, message: str) -> Any:
        """向指定会话发送消息

        Args:
            session_id: 会话ID
            message: 用户消息

        Returns:
            助手响应
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session not found: {session_id}")

        return self._sessions[session_id].chat(message)

    def reset_session(self, session_id: str) -> None:
        """重置指定会话

        Args:
            session_id: 会话ID
        """
        if session_id in self._sessions:
            self._sessions[session_id].reset()
            logger.info(f"Reset session: {session_id}")

    def delete_session(self, session_id: str) -> None:
        """删除会话

        Args:
            session_id: 会话ID
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")

    def get_session_ids(self) -> List[str]:
        """获取所有会话ID

        Returns:
            会话ID列表
        """
        return list(self._sessions.keys())

    def get_active_session_count(self) -> int:
        """获取活跃会话数量

        Returns:
            会话数量
        """
        return len(self._sessions)
