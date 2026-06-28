#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分割器

提供多种分割策略，支持句子、段落、语义分割
"""

import re
from enum import Enum
from typing import List, Optional, Callable

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter,
)
from llama_index.core.schema import TextNode

from src.core import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """分割策略枚举"""

    SENTENCE = "sentence"  # 句子分割
    PARAGRAPH = "paragraph"  # 段落分割
    SEMANTIC = "semantic"  # 语义分割
    TOKEN = "token"  # Token 分割
    CUSTOM = "custom"  # 自定义分割


class TextSplitter:
    """文本分割器

    支持多种分割策略，将文档分割成适合检索的文本块

    Example:
        >>> splitter = TextSplitter(
        ...     strategy=ChunkingStrategy.SENTENCE,
        ...     chunk_size=512,
        ...     chunk_overlap=50
        ... )
        >>> nodes = splitter.split_documents(documents)
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " ",
        backup_separators: Optional[List[str]] = None,
        embed_model: Optional[Callable] = None,
    ):
        """初始化文本分割器

        Args:
            strategy: 分割策略
            chunk_size: 块大小（字符数或 token 数）
            chunk_overlap: 块重叠大小
            separator: 分隔符
            backup_separators: 备选分隔符列表
            embed_model: 嵌入模型（语义分割需要）
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.backup_separators = backup_separators or ["\n", ".", "!", "?", ";", ",", " "]
        self.embed_model = embed_model

        self._splitter = self._create_splitter()
        logger.info(
            f"Initialized TextSplitter with strategy={strategy.value}, "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def _create_splitter(self):
        """创建对应的分割器实例"""
        if self.strategy == ChunkingStrategy.SENTENCE:
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=self.separator,
                paragraph_separator="\n\n\n",
                secondary_chunking_regex="[^,.;。！？]+[,.;。！？]?",
            )

        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n\n",
                paragraph_separator="\n\n\n",
            )

        elif self.strategy == ChunkingStrategy.SEMANTIC:
            if self.embed_model is None:
                logger.warning(
                    "No embed_model provided for semantic splitting, "
                    "falling back to sentence splitting"
                )
                return SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )

            return SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embed_model,
            )

        elif self.strategy == ChunkingStrategy.TOKEN:
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=self.separator,
                backup_separators=self.backup_separators,
            )

        else:
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def split_documents(self, documents: List[Document]) -> List[TextNode]:
        """分割文档列表

        Args:
            documents: 文档列表

        Returns:
            分割后的节点列表
        """
        all_nodes: List[TextNode] = []

        for doc in documents:
            try:
                nodes = self._splitter.get_nodes_from_documents([doc])
                # 保留原始文档元数据
                for node in nodes:
                    if node.metadata is None:
                        node.metadata = {}
                    node.metadata.update(doc.metadata)
                    node.metadata["chunk_strategy"] = self.strategy.value
                    node.metadata["chunk_size"] = self.chunk_size
                    node.metadata["chunk_overlap"] = self.chunk_overlap

                all_nodes.extend(nodes)

            except Exception as e:
                logger.error(f"Error splitting document {doc.metadata.get('file_name', 'unknown')}: {e}")
                continue

        logger.info(f"Split {len(documents)} documents into {len(all_nodes)} nodes")
        return all_nodes

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[TextNode]:
        """分割纯文本

        Args:
            text: 文本内容
            metadata: 元数据

        Returns:
            分割后的节点列表
        """
        doc = Document(text=text, metadata=metadata or {})
        return self._splitter.get_nodes_from_documents([doc])

    def split_text_to_chunks(self, text: str) -> List[str]:
        """将文本分割成字符串块

        Args:
            text: 文本内容

        Returns:
            字符串块列表
        """
        nodes = self.split_text(text)
        return [node.text for node in nodes]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的 token 数量

        使用简单的启发式方法：英文约 4 字符 = 1 token，中文约 1.5 字符 = 1 token

        Args:
            text: 文本内容

        Returns:
            估算的 token 数量
        """
        # 统计中文字符数
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # 非中文字符数
        other_chars = len(text) - chinese_chars

        # 估算 token 数
        estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return max(estimated_tokens, 1)
