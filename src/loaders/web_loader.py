#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网页数据加载器

支持从 URL 加载网页内容
"""

from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from llama_index.core import Document

from src.loaders.base import BaseLoader
from src.core import get_logger

logger = get_logger(__name__)


class WebLoader(BaseLoader):
    """网页数据加载器

    支持从单个或多个 URL 加载网页内容

    Example:
        >>> loader = WebLoader()
        >>> # 加载单个网页
        >>> docs = loader.load("https://example.com/article")
        >>> # 批量加载
        >>> docs = loader.load_batch([
        ...     "https://example.com/article1",
        ...     "https://example.com/article2"
        ... ])
    """

    def __init__(
        self,
        timeout: int = 30,
        user_agent: Optional[str] = None,
    ):
        """初始化网页加载器

        Args:
            timeout: 请求超时时间（秒）
            user_agent: 自定义 User-Agent
        """
        super().__init__()
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

    def load(
        self,
        source: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """从 URL 加载网页内容

        Args:
            source: 网页 URL
            extra_metadata: 额外元数据
            **kwargs: 额外参数

        Returns:
            加载的文档列表
        """
        extra_metadata = extra_metadata or {}

        if not self._is_valid_url(source):
            logger.error(f"Invalid URL: {source}")
            return []

        try:
            # 尝试使用 BeautifulSoupWebReader
            try:
                from llama_index.readers.web import BeautifulSoupWebReader

                reader = BeautifulSoupWebReader()
                documents = reader.load_data(urls=[source], **kwargs)

            except ImportError:
                # 回退到 SimpleWebPageReader
                logger.warning(
                    "BeautifulSoupWebReader not available, falling back to SimpleWebPageReader"
                )
                from llama_index.readers.web import SimpleWebPageReader

                reader = SimpleWebPageReader(html_to_text=True)
                documents = reader.load_data(urls=[source], **kwargs)

            # 添加元数据
            parsed_url = urlparse(source)
            base_metadata = {
                "url": source,
                "domain": parsed_url.netloc,
                "source_type": "web",
                **extra_metadata,
            }

            for doc in documents:
                doc.metadata.update(base_metadata)

            logger.info(f"Loaded {len(documents)} documents from {source}")
            return documents

        except Exception as e:
            logger.error(f"Error loading URL {source}: {e}")
            return []

    def load_batch(
        self, sources: List[str], extra_metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[Document]:
        """批量加载网页

        Args:
            sources: URL 列表
            extra_metadata: 额外元数据
            **kwargs: 额外参数

        Returns:
            所有加载的文档列表
        """
        extra_metadata = extra_metadata or {}
        all_documents: List[Document] = []

        # 过滤有效 URL
        valid_urls = [url for url in sources if self._is_valid_url(url)]

        if not valid_urls:
            logger.warning("No valid URLs provided")
            return []

        try:
            # 批量加载
            try:
                from llama_index.readers.web import BeautifulSoupWebReader

                reader = BeautifulSoupWebReader()
                documents = reader.load_data(urls=valid_urls, **kwargs)

            except ImportError:
                from llama_index.readers.web import SimpleWebPageReader

                reader = SimpleWebPageReader(html_to_text=True)
                documents = reader.load_data(urls=valid_urls, **kwargs)

            # 添加元数据
            for i, doc in enumerate(documents):
                url = valid_urls[i] if i < len(valid_urls) else valid_urls[-1]
                parsed_url = urlparse(url)
                doc.metadata.update({
                    "url": url,
                    "domain": parsed_url.netloc,
                    "source_type": "web",
                    **extra_metadata,
                })

            all_documents.extend(documents)
            self._loaded_count += len(documents)
            logger.info(f"Loaded {len(documents)} documents from {len(valid_urls)} URLs")

        except Exception as e:
            logger.error(f"Error loading URLs: {e}")

        return all_documents

    def _is_valid_url(self, url: str) -> bool:
        """验证 URL 是否有效

        Args:
            url: 待验证的 URL

        Returns:
            是否有效
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
