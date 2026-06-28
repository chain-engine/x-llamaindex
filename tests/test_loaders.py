#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档加载器测试
"""

import pytest
from pathlib import Path

from src.loaders import DocumentLoader, WebLoader
from src.loaders.base import BaseLoader


class TestDocumentLoader:
    """DocumentLoader 测试类"""

    def test_init(self):
        """测试初始化"""
        loader = DocumentLoader()
        assert loader.encoding == "utf-8"
        assert loader.required_exts is not None
        assert ".txt" in loader.required_exts

    def test_load_text(self):
        """测试从文本创建文档"""
        loader = DocumentLoader()
        text = "This is a test document."
        metadata = {"source": "test"}

        doc = loader.load_text(text, metadata)

        assert doc.text == text
        assert doc.metadata["source"] == "test"

    def test_load_texts(self):
        """测试批量从文本创建文档"""
        loader = DocumentLoader()
        texts = ["Text 1", "Text 2", "Text 3"]
        metadatas = [{"id": i} for i in range(3)]

        documents = loader.load_texts(texts, metadatas)

        assert len(documents) == 3
        for i, doc in enumerate(documents):
            assert doc.text == texts[i]
            assert doc.metadata["id"] == i

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        loader = DocumentLoader()
        documents = loader.load("./nonexistent_file.txt")

        assert documents == []

    def test_supported_extensions(self):
        """测试支持的扩展名"""
        extensions = DocumentLoader.get_supported_extensions()

        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".md" in extensions

    def test_load_from_directory(self, temp_dir, sample_documents):
        """测试从目录加载文档"""
        # 创建测试文件
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content for document loading.", encoding="utf-8")

        loader = DocumentLoader(required_exts=[".txt"])
        documents = loader.load(temp_dir)

        assert len(documents) >= 1
        assert "Test content" in documents[0].text


class TestWebLoader:
    """WebLoader 测试类"""

    def test_init(self):
        """测试初始化"""
        loader = WebLoader()
        assert loader.timeout == 30

    def test_is_valid_url(self):
        """测试 URL 验证"""
        loader = WebLoader()

        assert loader._is_valid_url("https://example.com")
        assert loader._is_valid_url("http://localhost:8080")
        assert not loader._is_valid_url("not-a-url")
        assert not loader._is_valid_url("")


class TestBaseLoader:
    """BaseLoader 测试类"""

    def test_loaded_count(self):
        """测试计数功能"""
        loader = DocumentLoader()
        assert loader.loaded_count == 0

        loader.load_texts(["Text 1", "Text 2"])
        assert loader.loaded_count == 2

        loader.reset_count()
        assert loader.loaded_count == 0
