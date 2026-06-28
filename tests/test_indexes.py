#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
索引管理器测试
"""

import pytest
from unittest.mock import Mock, patch

from src.indexes import IndexManager, IndexType, VectorIndexManager


class TestVectorIndexManager:
    """VectorIndexManager 测试类"""

    def test_init(self):
        """测试初始化"""
        manager = VectorIndexManager()

        assert manager.embed_model is None
        assert manager.document_count == 0
        assert manager.node_count == 0

    def test_build_index_from_nodes(self, sample_documents):
        """测试从节点构建索引"""
        from src.processors import TextSplitter

        manager = VectorIndexManager(show_progress=False)
        splitter = TextSplitter(chunk_size=100)

        nodes = splitter.split_documents(sample_documents)
        index = manager.build_index_from_nodes(nodes)

        assert index is not None
        assert manager.node_count == len(nodes)

    def test_get_retriever_without_index(self):
        """测试未构建索引时获取检索器"""
        manager = VectorIndexManager()

        with pytest.raises(RuntimeError):
            manager.get_retriever()

    def test_get_stats(self, sample_documents):
        """测试获取统计信息"""
        from src.processors import TextSplitter

        manager = VectorIndexManager(show_progress=False)
        splitter = TextSplitter()

        nodes = splitter.split_documents(sample_documents)
        manager.build_index_from_nodes(nodes)

        stats = manager.get_stats()

        assert "document_count" in stats
        assert "node_count" in stats


class TestIndexManager:
    """IndexManager 测试类"""

    def test_init(self):
        """测试初始化"""
        manager = IndexManager(index_type=IndexType.VECTOR)

        assert manager.index_type == IndexType.VECTOR
        assert manager.persist_dir is not None

    def test_build_from_documents(self, sample_documents):
        """测试从文档构建索引"""
        manager = IndexManager(
            index_type=IndexType.VECTOR,
            auto_save=False,
        )

        index = manager.build_from_documents(sample_documents, show_progress=False)

        assert index is not None
        assert manager.document_count == len(sample_documents)

    def test_build_from_nodes(self, sample_documents):
        """测试从节点构建索引"""
        from src.processors import TextSplitter

        manager = IndexManager(
            index_type=IndexType.VECTOR,
            auto_save=False,
        )
        splitter = TextSplitter()
        nodes = splitter.split_documents(sample_documents)

        index = manager.build_from_nodes(nodes, show_progress=False)

        assert index is not None
        assert manager.node_count == len(nodes)

    def test_get_stats(self):
        """测试获取统计信息"""
        manager = IndexManager(auto_save=False)

        stats = manager.get_stats()

        assert "index_type" in stats
        assert stats["index_type"] == "vector"

    def test_unsupported_index_type(self):
        """测试不支持的索引类型"""
        with pytest.raises(ValueError):
            # 直接传入无效类型
            IndexManager.INDEX_MANAGERS[IndexType.VECTOR]  # 这应该工作
            # 但创建不存在的类型会失败
            manager = IndexManager.__new__(IndexManager)
            manager.index_type = "invalid"  # type: ignore
            manager._create_manager()  # 这应该抛出异常
