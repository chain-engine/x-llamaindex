#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多格式文档加载器

支持加载 TXT, PDF, DOCX, Markdown, JSON 等格式的文档
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

from src.loaders.base import BaseLoader
from src.core import get_logger, settings

logger = get_logger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".md", ".json", ".csv", ".html"]


class DocumentLoader(BaseLoader):
    """多格式文档加载器

    支持从目录或单个文件加载多种格式的文档

    Example:
        >>> loader = DocumentLoader()
        >>> # 从目录加载
        >>> docs = loader.load("./data")
        >>> # 从单个文件加载
        >>> docs = loader.load("./document.pdf")
        >>> # 批量加载
        >>> docs = loader.load_batch(["./file1.txt", "./file2.pdf"])
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        required_exts: Optional[List[str]] = None,
        exclude_hidden: bool = True,
        recursive: bool = True,
    ):
        """初始化文档加载器

        Args:
            encoding: 文件编码
            required_exts: 要求的文件扩展名列表，默认支持所有 SUPPORTED_EXTENSIONS
            exclude_hidden: 是否排除隐藏文件
            recursive: 是否递归加载子目录
        """
        super().__init__(encoding=encoding)
        self.required_exts = required_exts or SUPPORTED_EXTENSIONS
        self.exclude_hidden = exclude_hidden
        self.recursive = recursive

    def load(
        self,
        source: str | Path,
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """加载文档

        Args:
            source: 文件路径或目录路径
            extra_metadata: 额外的元数据
            **kwargs: 额外参数

        Returns:
            加载的文档列表
        """
        source_path = Path(source)
        extra_metadata = extra_metadata or {}

        if source_path.is_dir():
            return self._load_from_directory(source_path, extra_metadata, **kwargs)
        elif source_path.is_file():
            return self._load_from_file(source_path, extra_metadata, **kwargs)
        else:
            logger.error(f"Source not found: {source}")
            return []

    def _load_from_directory(
        self,
        dir_path: Path,
        extra_metadata: Dict[str, Any],
        **kwargs,
    ) -> List[Document]:
        """从目录加载所有支持的文档

        Args:
            dir_path: 目录路径
            extra_metadata: 额外元数据
            **kwargs: 额外参数

        Returns:
            文档列表
        """
        logger.info(f"Loading documents from directory: {dir_path}")

        try:
            reader = SimpleDirectoryReader(
                input_dir=str(dir_path),
                required_exts=self.required_exts,
                exclude_hidden=self.exclude_hidden,
                recursive=self.recursive,
                encoding=self.encoding,
            )

            documents = reader.load_data(**kwargs)

            # 添加额外元数据
            for doc in documents:
                doc.metadata.update(extra_metadata)
                # 确保有 file_name 元数据
                if "file_name" not in doc.metadata and "file_path" in doc.metadata:
                    doc.metadata["file_name"] = Path(doc.metadata["file_path"]).name

            logger.info(f"Loaded {len(documents)} documents from {dir_path}")
            return documents

        except Exception as e:
            logger.error(f"Error loading directory {dir_path}: {e}")
            return []

    def _load_from_file(
        self,
        file_path: Path,
        extra_metadata: Dict[str, Any],
        **kwargs,
    ) -> List[Document]:
        """从单个文件加载文档

        Args:
            file_path: 文件路径
            extra_metadata: 额外元数据
            **kwargs: 额外参数

        Returns:
            文档列表
        """
        logger.debug(f"Loading document from file: {file_path}")

        if file_path.suffix.lower() not in self.required_exts:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []

        try:
            # 使用 SimpleDirectoryReader 加载单个文件
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                encoding=self.encoding,
            )

            documents = reader.load_data(**kwargs)

            # 添加元数据
            base_metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path.absolute()),
                "file_type": file_path.suffix.lower(),
                "file_size": file_path.stat().st_size,
                **extra_metadata,
            }

            for doc in documents:
                doc.metadata.update(base_metadata)

            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []

    def load_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """从纯文本创建文档

        Args:
            text: 文本内容
            metadata: 元数据

        Returns:
            Document 对象
        """
        metadata = metadata or {}
        doc = Document(text=text, metadata=metadata)
        logger.debug("Created document from text")
        return doc

    def load_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """从多个纯文本创建文档

        Args:
            texts: 文本列表
            metadatas: 元数据列表

        Returns:
            Document 对象列表
        """
        metadatas = metadatas or [{} for _ in texts]
        documents = [
            Document(text=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        logger.info(f"Created {len(documents)} documents from texts")
        return documents

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取支持的文件扩展名列表"""
        return SUPPORTED_EXTENSIONS.copy()
