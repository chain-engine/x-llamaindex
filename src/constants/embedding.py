#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Constants
"""

# Embedding model config
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-ada-002"
DEFAULT_EMBEDDING_DEVICE: str = "cpu"
DEFAULT_EMBEDDING_BATCH_SIZE: int = 100
DEFAULT_EMBEDDING_CACHE_SIZE: int = 1000

__all__ = ["DEFAULT_EMBEDDING_MODEL", "DEFAULT_EMBEDDING_DEVICE", "DEFAULT_EMBEDDING_BATCH_SIZE", "DEFAULT_EMBEDDING_CACHE_SIZE"]
