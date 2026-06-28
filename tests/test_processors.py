#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理器测试
"""

import pytest

from src.processors import TextSplitter, ChunkingStrategy, MetadataExtractor


class TestTextSplitter:
    """TextSplitter 测试类"""

    def test_init_default(self):
        """测试默认初始化"""
        splitter = TextSplitter()

        assert splitter.strategy == ChunkingStrategy.SENTENCE
        assert splitter.chunk_size == 512
        assert splitter.chunk_overlap == 50

    def test_init_custom(self):
        """测试自定义初始化"""
        splitter = TextSplitter(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=1024,
            chunk_overlap=100,
        )

        assert splitter.strategy == ChunkingStrategy.PARAGRAPH
        assert splitter.chunk_size == 1024
        assert splitter.chunk_overlap == 100

    def test_split_text(self, sample_text):
        """测试文本分割"""
        splitter = TextSplitter(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=100,
            chunk_overlap=20,
        )

        nodes = splitter.split_text(sample_text)

        assert len(nodes) > 0
        for node in nodes:
            assert node.text is not None
            assert len(node.text) > 0

    def test_split_documents(self, sample_documents):
        """测试文档分割"""
        splitter = TextSplitter(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=100,
        )

        nodes = splitter.split_documents(sample_documents)

        assert len(nodes) > 0
        for node in nodes:
            assert node.metadata.get("chunk_strategy") == "sentence"

    def test_split_text_to_chunks(self, sample_text):
        """测试分割为字符串块"""
        splitter = TextSplitter(chunk_size=100)

        chunks = splitter.split_text_to_chunks(sample_text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_estimate_tokens(self):
        """测试 token 估算"""
        # 英文文本
        english_text = "This is a simple test for token estimation."
        tokens = TextSplitter.estimate_tokens(english_text)
        assert tokens > 0

        # 中文文本
        chinese_text = "这是一个用于测试 token 估算的中文文本。"
        tokens = TextSplitter.estimate_tokens(chinese_text)
        assert tokens > 0


class TestMetadataExtractor:
    """MetadataExtractor 测试类"""

    def test_init(self):
        """测试初始化"""
        extractor = MetadataExtractor()
        assert extractor.extract_keywords is True
        assert extractor.extract_language is True

    def test_extract_statistics(self, sample_documents):
        """测试统计信息提取"""
        extractor = MetadataExtractor(extract_statistics=True)
        doc = sample_documents[0]

        metadata = extractor.extract(doc)

        assert "char_count" in metadata
        assert "word_count" in metadata
        assert metadata["char_count"] > 0

    def test_detect_language(self, sample_documents):
        """测试语言检测"""
        extractor = MetadataExtractor(extract_language=True)
        doc = sample_documents[0]

        metadata = extractor.extract(doc)

        assert "language" in metadata
        assert metadata["language"] in ["en", "zh", "mixed", "unknown"]

    def test_extract_keywords(self, sample_documents):
        """测试关键词提取"""
        extractor = MetadataExtractor(extract_keywords=True)
        doc = sample_documents[0]

        metadata = extractor.extract(doc)

        assert "keywords" in metadata
        assert isinstance(metadata["keywords"], list)

    def test_enrich_documents(self, sample_documents):
        """测试批量添加元数据"""
        extractor = MetadataExtractor()

        enriched = extractor.enrich_documents(sample_documents)

        assert len(enriched) == len(sample_documents)
        for doc in enriched:
            assert "processed_at" in doc.metadata

    def test_extract_file_metadata(self, temp_dir):
        """测试文件元数据提取"""
        # 创建测试文件
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content", encoding="utf-8")

        metadata = MetadataExtractor.extract_file_metadata(test_file)

        assert metadata["file_name"] == "test.txt"
        assert metadata["file_type"] == ".txt"
        assert metadata["file_size"] > 0
