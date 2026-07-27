#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Constants
"""

# Retrieval config
DEFAULT_TOP_K: int = 5
DEFAULT_SIMILARITY_THRESHOLD: float = 0.5
DEFAULT_MMR_LAMBDA: float = 0.7

# Document chunking config
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_CHUNK_OVERLAP: int = 50

__all__ = ["DEFAULT_TOP_K", "DEFAULT_SIMILARITY_THRESHOLD", "DEFAULT_MMR_LAMBDA", "DEFAULT_CHUNK_SIZE", "DEFAULT_CHUNK_OVERLAP"]
