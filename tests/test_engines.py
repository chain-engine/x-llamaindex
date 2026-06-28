#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询引擎测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.engines import RAGQueryEngine, RAGChatEngine, ConversationManager
from src.engines.query_engine import QueryMode


class TestRAGQueryEngine:
    """RAGQueryEngine 测试类"""

    def test_init(self, mock_llm):
        """测试初始化"""
        engine = RAGQueryEngine(
            index=None,
            llm=mock_llm,
            similarity_top_k=5,
        )

        assert engine.similarity_top_k == 5
        assert engine.response_mode == QueryMode.COMPACT
        assert not engine.streaming

    def test_query_with_sources(self, mock_llm, sample_documents):
        """测试带来源的查询"""
        from src.indexes import IndexManager

        # 创建索引
        index_manager = IndexManager(auto_save=False)
        index_manager.build_from_documents(sample_documents, show_progress=False)

        # 创建查询引擎
        engine = RAGQueryEngine(
            index=index_manager.index,
            llm=mock_llm,
        )

        # 由于没有真正的 LLM，这里只测试结构
        assert engine.query_engine is not None

    def test_get_stats(self, mock_llm):
        """测试获取统计信息"""
        engine = RAGQueryEngine(
            index=None,
            llm=mock_llm,
        )

        stats = engine.get_stats()

        assert "query_count" in stats
        assert "total_latency" in stats
        assert "response_mode" in stats

    def test_reset_stats(self, mock_llm):
        """测试重置统计"""
        engine = RAGQueryEngine(index=None, llm=mock_llm)
        engine._query_count = 10
        engine._total_latency = 5.0

        engine.reset_stats()

        assert engine._query_count == 0
        assert engine._total_latency == 0.0

    def test_update_config(self, mock_llm):
        """测试更新配置"""
        engine = RAGQueryEngine(
            index=None,
            llm=mock_llm,
            similarity_top_k=5,
        )

        engine.update_config(
            similarity_top_k=10,
            response_mode=QueryMode.REFINE,
        )

        assert engine.similarity_top_k == 10
        assert engine.response_mode == QueryMode.REFINE


class TestRAGChatEngine:
    """RAGChatEngine 测试类"""

    def test_init(self, mock_llm):
        """测试初始化"""
        engine = RAGChatEngine(
            index=None,
            llm=mock_llm,
        )

        assert engine.llm is not None
        assert engine.system_prompt is not None

    def test_get_default_system_prompt(self, mock_llm):
        """测试默认系统提示词"""
        engine = RAGChatEngine(index=None, llm=mock_llm)

        prompt = engine.system_prompt

        assert prompt is not None
        assert len(prompt) > 0

    def test_set_system_prompt(self, mock_llm):
        """测试设置系统提示词"""
        engine = RAGChatEngine(index=None, llm=mock_llm)

        new_prompt = "You are a helpful assistant."
        engine.set_system_prompt(new_prompt)

        assert engine.system_prompt == new_prompt

    def test_get_stats(self, mock_llm):
        """测试获取统计信息"""
        engine = RAGChatEngine(index=None, llm=mock_llm)

        stats = engine.get_stats()

        assert "message_count" in stats
        assert "chat_mode" in stats


class TestConversationManager:
    """ConversationManager 测试类"""

    def test_init(self, mock_llm):
        """测试初始化"""
        manager = ConversationManager(
            index=None,
            llm=mock_llm,
            max_sessions=10,
        )

        assert manager.max_sessions == 10
        assert manager.get_active_session_count() == 0

    def test_create_session(self, mock_llm):
        """测试创建会话"""
        manager = ConversationManager(index=None, llm=mock_llm)

        session_id = manager.create_session()

        assert session_id is not None
        assert manager.get_active_session_count() == 1

    def test_create_session_with_id(self, mock_llm):
        """测试使用指定 ID 创建会话"""
        manager = ConversationManager(index=None, llm=mock_llm)

        session_id = manager.create_session(session_id="test-session")

        assert session_id == "test-session"
        assert "test-session" in manager.get_session_ids()

    def test_delete_session(self, mock_llm):
        """测试删除会话"""
        manager = ConversationManager(index=None, llm=mock_llm)

        session_id = manager.create_session()
        assert manager.get_active_session_count() == 1

        manager.delete_session(session_id)
        assert manager.get_active_session_count() == 0

    def test_max_sessions_limit(self, mock_llm):
        """测试最大会话数限制"""
        manager = ConversationManager(
            index=None,
            llm=mock_llm,
            max_sessions=2,
        )

        manager.create_session("session1")
        manager.create_session("session2")
        manager.create_session("session3")  # 这应该触发删除最旧会话

        assert manager.get_active_session_count() == 2
        assert "session1" not in manager.get_session_ids()
