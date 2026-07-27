#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest 配置文件

定义测试 fixtures 和配置
"""

import os
import sys
import pytest
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

# 设置测试环境变量（使用 fake key，避免触发真实 API 调用）
os.environ["LLM_PROVIDER"] = "deepseek"
os.environ["DEEPSEEK_API_KEY"] = "test-fake-key-do-not-use-in-production"
os.environ["DEEPSEEK_API_BASE"] = "https://api.deepseek.com/v1"
os.environ["DEBUG"] = "true"


@pytest.fixture
def mock_llm():
    """Mock LLM fixture"""
    llm = Mock()
    llm.complete = Mock(return_value=Mock(text="This is a mock response."))
    llm.chat = Mock(return_value=Mock(message=Mock(content="This is a mock chat response.")))
    llm.stream = Mock(return_value=iter(["Mock", " stream", " response"]))
    return llm


@pytest.fixture
def mock_embed_model():
    """Mock embedding model fixture"""
    import numpy as np
    embed_model = Mock()
    embed_model.get_text_embedding = Mock(return_value=[0.1] * 1536)
    embed_model.get_text_embeddings = Mock(return_value=[[0.1] * 1536] * 5)
    return embed_model


@pytest.fixture
def sample_documents():
    """示例文档 fixture"""
    from llama_index.core import Document

    return [
        Document(
            text="LlamaIndex is a data orchestration framework for LLM applications.",
            metadata={"source": "test", "id": "doc1"},
        ),
        Document(
            text="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "test", "id": "doc2"},
        ),
        Document(
            text="Vector stores are used to store embeddings for semantic search.",
            metadata={"source": "test", "id": "doc3"},
        ),
    ]


@pytest.fixture
def sample_text():
    """示例文本 fixture"""
    return """
    LlamaIndex 是一个专为大语言模型应用设计的数据编排框架。
    它主要用于构建检索增强生成（RAG）系统。
    LlamaIndex 提供了丰富的数据连接器、索引结构和查询接口。

    核心功能包括：
    1. 数据连接：支持多种数据源
    2. 数据索引：提供多种索引结构
    3. 检索增强：通过检索增强 LLM 能力
    4. 查询接口：简单易用的查询接口
    """


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """临时目录 fixture"""
    yield tmp_path
