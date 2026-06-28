#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest 配置文件

定义测试 fixtures 和配置
"""

import os
import pytest
from pathlib import Path
from typing import Generator

# 设置测试环境变量
os.environ["OPENAI_API_KEY"] = "test-api-key"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["DEBUG"] = "true"


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


@pytest.fixture
def mock_llm():
    """模拟 LLM fixture"""
    from unittest.mock import Mock

    llm = Mock()
    llm.complete = Mock(return_value=Mock(text="This is a mock response."))
    return llm


@pytest.fixture
def mock_embed_model():
    """模拟嵌入模型 fixture"""
    from unittest.mock import Mock
    import numpy as np

    embed_model = Mock()
    embed_model.get_text_embedding = Mock(
        return_value=np.random.rand(1536).tolist()
    )
    return embed_model
