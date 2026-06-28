#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档加载器模块

提供多种格式的文档加载能力
"""

from src.loaders.base import BaseLoader
from src.loaders.document_loader import DocumentLoader
from src.loaders.web_loader import WebLoader

__all__ = ["BaseLoader", "DocumentLoader", "WebLoader"]
