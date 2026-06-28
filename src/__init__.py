#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
x-llamaindex 源码模块

企业级 LlamaIndex RAG 系统实现
"""

__version__ = "0.1.0"
__author__ = "John Young"

# 延迟导入，避免循环依赖
def __getattr__(name: str):
    """延迟导入模块属性"""
    _imports = {
        # Loaders
        "DocumentLoader": ("src.loaders", "DocumentLoader"),
        "WebLoader": ("src.loaders", "WebLoader"),
        "BaseLoader": ("src.loaders", "BaseLoader"),
        # Processors
        "TextSplitter": ("src.processors", "TextSplitter"),
        "ChunkingStrategy": ("src.processors", "ChunkingStrategy"),
        "MetadataExtractor": ("src.processors", "MetadataExtractor"),
        # Storage
        "VectorStoreManager": ("src.storage", "VectorStoreManager"),
        # Indexes
        "IndexManager": ("src.indexes", "IndexManager"),
        "IndexType": ("src.indexes", "IndexType"),
        "BaseIndexManager": ("src.indexes", "BaseIndexManager"),
        "VectorIndexManager": ("src.indexes", "VectorIndexManager"),
        # Retrievers
        "HybridRetriever": ("src.retrievers", "HybridRetriever"),
        "Reranker": ("src.retrievers", "Reranker"),
        "BaseRetrieverWrapper": ("src.retrievers", "BaseRetrieverWrapper"),
        # Engines
        "RAGQueryEngine": ("src.engines", "RAGQueryEngine"),
        "QueryMode": ("src.engines", "QueryMode"),
        "RAGChatEngine": ("src.engines", "RAGChatEngine"),
        "ChatMode": ("src.engines", "ChatMode"),
        "ConversationManager": ("src.engines", "ConversationManager"),
        # Evaluators
        "RAGEvaluator": ("src.evaluators", "RAGEvaluator"),
        "EvaluationMetrics": ("src.evaluators", "EvaluationMetrics"),
        # Utils
        "setup_environment": ("src.utils", "setup_environment"),
        "get_llm": ("src.utils", "get_llm"),
        "get_embedding_model": ("src.utils", "get_embedding_model"),
    }

    if name in _imports:
        import importlib
        module_path, attr_name = _imports[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Loaders
    "DocumentLoader",
    "WebLoader",
    "BaseLoader",
    # Processors
    "TextSplitter",
    "ChunkingStrategy",
    "MetadataExtractor",
    # Storage
    "VectorStoreManager",
    # Indexes
    "IndexManager",
    "IndexType",
    "BaseIndexManager",
    "VectorIndexManager",
    # Retrievers
    "HybridRetriever",
    "Reranker",
    "BaseRetrieverWrapper",
    # Engines
    "RAGQueryEngine",
    "QueryMode",
    "RAGChatEngine",
    "ChatMode",
    "ConversationManager",
    # Evaluators
    "RAGEvaluator",
    "EvaluationMetrics",
    # Utils
    "setup_environment",
    "get_llm",
    "get_embedding_model",
]
