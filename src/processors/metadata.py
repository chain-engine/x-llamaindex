#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元数据提取器

自动从文档中提取元数据
"""

import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from llama_index.core import Document

from src.core import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """元数据提取器

    从文档内容和文件属性中提取元数据

    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract(document)
        >>> print(metadata)
        {'word_count': 500, 'language': 'zh', ...}
    """

    def __init__(
        self,
        extract_keywords: bool = True,
        extract_language: bool = True,
        extract_statistics: bool = True,
    ):
        """初始化元数据提取器

        Args:
            extract_keywords: 是否提取关键词
            extract_language: 是否检测语言
            extract_statistics: 是否提取统计信息
        """
        self.extract_keywords = extract_keywords
        self.extract_language = extract_language
        self.extract_statistics = extract_statistics

    def extract(self, document: Document) -> Dict[str, Any]:
        """从文档提取元数据

        Args:
            document: 文档对象

        Returns:
            提取的元数据字典
        """
        metadata = dict(document.metadata)
        text = document.text

        # 提取统计信息
        if self.extract_statistics:
            metadata.update(self._extract_statistics(text))

        # 提取语言
        if self.extract_language:
            metadata["language"] = self._detect_language(text)

        # 提取关键词（简单实现）
        if self.extract_keywords:
            metadata["keywords"] = self._extract_keywords(text)

        # 添加处理时间
        metadata["processed_at"] = datetime.now().isoformat()

        return metadata

    def extract_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """批量提取元数据

        Args:
            documents: 文档列表

        Returns:
            元数据列表
        """
        return [self.extract(doc) for doc in documents]

    def enrich_documents(self, documents: List[Document]) -> List[Document]:
        """为文档添加提取的元数据

        Args:
            documents: 文档列表

        Returns:
            添加元数据后的文档列表
        """
        for doc in documents:
            extracted = self.extract(doc)
            doc.metadata.update(extracted)

        logger.info(f"Enriched {len(documents)} documents with metadata")
        return documents

    def _extract_statistics(self, text: str) -> Dict[str, Any]:
        """提取文本统计信息

        Args:
            text: 文本内容

        Returns:
            统计信息字典
        """
        # 基础统计
        char_count = len(text)
        word_count = len(text.split())
        line_count = len(text.splitlines())
        paragraph_count = len([p for p in text.split("\n\n") if p.strip()])

        # 中文字符数
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))

        # 英文单词数
        english_words = len(re.findall(r"[a-zA-Z]+", text))

        # 数字数量
        numbers = len(re.findall(r"\d+", text))

        return {
            "char_count": char_count,
            "word_count": word_count,
            "chinese_char_count": chinese_chars,
            "english_word_count": english_words,
            "number_count": numbers,
            "line_count": line_count,
            "paragraph_count": paragraph_count,
        }

    def _detect_language(self, text: str) -> str:
        """检测文本语言

        简单的语言检测，基于中英文字符比例

        Args:
            text: 文本内容

        Returns:
            语言代码 ('zh', 'en', 'mixed', 'unknown')
        """
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total = chinese_chars + english_chars

        if total == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total
        english_ratio = english_chars / total

        if chinese_ratio > 0.7:
            return "zh"
        elif english_ratio > 0.7:
            return "en"
        else:
            return "mixed"

    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词

        简单的关键词提取，基于词频统计

        Args:
            text: 文本内容
            top_k: 返回的关键词数量

        Returns:
            关键词列表
        """
        # 停用词列表（简化版）
        stop_words = {
            "的", "是", "在", "和", "了", "有", "我", "他", "她", "它",
            "这", "那", "就", "也", "都", "而", "及", "与", "或", "等",
            "但", "如", "对", "能", "被", "把", "从", "到", "为", "以",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "and", "but",
            "if", "or", "because", "until", "while", "about", "against",
        }

        # 提取中英文词汇
        chinese_words = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        english_words = re.findall(r"[a-zA-Z]{3,}", text)

        # 合并并转小写
        words = [w.lower() for w in chinese_words + english_words]

        # 过滤停用词
        words = [w for w in words if w not in stop_words]

        # 统计词频
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # 排序并返回前 top_k 个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:top_k]]

        return keywords

    @staticmethod
    def extract_file_metadata(file_path: str | Path) -> Dict[str, Any]:
        """从文件属性提取元数据

        Args:
            file_path: 文件路径

        Returns:
            文件元数据
        """
        path = Path(file_path)

        if not path.exists():
            return {}

        stat = path.stat()

        return {
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "file_type": path.suffix.lower(),
            "file_size": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "parent_dir": str(path.parent),
        }
