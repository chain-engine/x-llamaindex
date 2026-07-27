#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Settings Configuration Module

全局配置中心
支持从环境变量和YAML配置文件读取配置
"""

import os
from typing import Final, Any
from pathlib import Path
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

from src.constants.rag import (
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from src.constants.generation import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
)
from src.constants.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_CACHE_SIZE,
)


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file_path: str = "./logs/rag_app.log"
    rotation: str = "1 day"
    retention: str = "7 days"


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = DEFAULT_TIMEOUT


@dataclass
class LLMProvidersConfig:
    """LLM 提供商配置"""
    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"
    # Kimi
    kimi_api_key: str = ""
    kimi_api_base: str = "https://api.moonshot.cn/v1"
    kimi_model: str = "moonshot-v1-8k"
    # GLM
    glm_api_key: str = ""
    glm_api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4-flash"


@dataclass
class EmbeddingConfig:
    """向量模型配置"""
    provider: str = "deepseek"
    model: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = 1536
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE


@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    type: str = "chroma"
    persist_dir: str = "./storage/chroma"
    collection_name: str = "llamaindex_docs"


@dataclass
class DocumentConfig:
    """文档处理配置"""
    chunking_strategy: str = "sentence"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    supported_extensions: list[str] = field(default_factory=lambda: [".txt", ".pdf", ".docx", ".md", ".json", ".csv", ".html"])


@dataclass
class RetrievalConfig:
    """检索配置"""
    similarity_top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder"
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    mmr_lambda: float = DEFAULT_MMR_LAMBDA


@dataclass
class RAGConfig:
    """RAG 配置"""
    data_dir: str = "./data"
    output_dir: str = "./reports"
    templates_dir: str = "./templates"
    max_sources: int = 20
    default_topic: str = "general"


@dataclass
class EvaluationConfig:
    """评估配置"""
    enabled: bool = False
    metrics: list[str] = field(default_factory=lambda: ["faithfulness", "relevancy", "context_similarity"])


class Settings:
    """应用配置类

    支持从环境变量和YAML配置文件读取配置
    优先级：环境变量 > YAML配置文件 > 默认配置
    """

    CONFIG_FILE_PATH: Final[str] = "config.yaml"

    def __init__(self) -> None:
        """初始化配置"""
        load_dotenv()
        self._config: dict[str, Any] = self._load_config()
        self._parse_config()

    def _get_default_config(self) -> dict[str, Any]:
        """获取默认配置"""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": True,
                "cors_origins": ["*"],
            },
            "logging": {
                "level": "INFO",
                "file_path": "./logs/rag_app.log",
                "rotation": "1 day",
                "retention": "7 days",
            },
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "timeout": DEFAULT_TIMEOUT,
            },
            "llm_providers": {
                "deepseek_api_key": "",
                "deepseek_api_base": "https://api.deepseek.com/v1",
                "deepseek_model": "deepseek-v4-pro",
                "kimi_api_key": "",
                "kimi_api_base": "https://api.moonshot.cn/v1",
                "kimi_model": "moonshot-v1-8k",
                "glm_api_key": "",
                "glm_api_base": "https://open.bigmodel.cn/api/paas/v4",
                "glm_model": "glm-4-flash",
            },
            "embedding": {
                "provider": "deepseek",
                "model": DEFAULT_EMBEDDING_MODEL,
                "dimension": 1536,
                "batch_size": DEFAULT_EMBEDDING_BATCH_SIZE,
            },
            "vector_store": {
                "type": "chroma",
                "persist_dir": "./storage/chroma",
                "collection_name": "llamaindex_docs",
            },
            "document": {
                "chunking_strategy": "sentence",
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                "supported_extensions": [".txt", ".pdf", ".docx", ".md", ".json", ".csv", ".html"],
            },
            "retrieval": {
                "similarity_top_k": DEFAULT_TOP_K,
                "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
                "rerank_enabled": False,
                "rerank_model": "cross-encoder",
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "mmr_lambda": DEFAULT_MMR_LAMBDA,
            },
            "rag": {
                "data_dir": "./data",
                "output_dir": "./reports",
                "templates_dir": "./templates",
                "max_sources": 20,
                "default_topic": "general",
            },
            "evaluation": {
                "enabled": False,
                "metrics": ["faithfulness", "relevancy", "context_similarity"],
            },
        }

    def _load_config(self) -> dict[str, Any]:
        """加载配置"""
        config: dict[str, Any] = self._get_default_config()
        self._load_from_file(config)
        self._load_from_env(config)
        return config

    def _load_from_file(self, config: dict[str, Any]) -> None:
        """从YAML文件加载配置"""
        config_file: Path = Path(self.CONFIG_FILE_PATH)
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    file_config: dict[str, Any] = yaml.safe_load(f) or {}
                self._merge_config(config, file_config)
            except Exception as e:
                print(f"Warning: Cannot load config file {self.CONFIG_FILE_PATH}: {e}")

    def _merge_config(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """递归合并配置"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _load_from_env(self, config: dict[str, Any]) -> None:
        """从环境变量加载配置"""
        env_mappings = {
            # Server
            "DEBUG": ("server", "debug", lambda v: v.lower() == "true"),
            "PORT": ("server", "port", int),
            "SERVER_PORT": ("server", "port", int),
            "SERVER_HOST": ("server", "host", str),
            # Logging
            "LOG_LEVEL": ("logging", "level", str),
            "LOG_FILE_PATH": ("logging", "file_path", str),
            "LOG_ROTATION": ("logging", "rotation", str),
            "LOG_RETENTION": ("logging", "retention", str),
            # LLM
            "LLM_PROVIDER": ("llm", "provider", str),
            "TEMPERATURE": ("llm", "temperature", float),
            "MAX_TOKENS": ("llm", "max_tokens", int),
            "GENERATION_TIMEOUT": ("llm", "timeout", int),
            # DeepSeek
            "DEEPSEEK_API_KEY": ("llm_providers", "deepseek_api_key", str),
            "DEEPSEEK_API_BASE": ("llm_providers", "deepseek_api_base", str),
            "DEEPSEEK_MODEL": ("llm_providers", "deepseek_model", str),
            # Kimi
            "KIMI_API_KEY": ("llm_providers", "kimi_api_key", str),
            "KIMI_API_BASE": ("llm_providers", "kimi_api_base", str),
            "KIMI_MODEL": ("llm_providers", "kimi_model", str),
            # GLM
            "GLM_API_KEY": ("llm_providers", "glm_api_key", str),
            "GLM_API_BASE": ("llm_providers", "glm_api_base", str),
            "GLM_MODEL": ("llm_providers", "glm_model", str),
            # Embedding
            "EMBEDDING_PROVIDER": ("embedding", "provider", str),
            "EMBEDDING_MODEL": ("embedding", "model", str),
            "EMBEDDING_DIMENSION": ("embedding", "dimension", int),
            "EMBEDDING_BATCH_SIZE": ("embedding", "batch_size", int),
            # Vector Store
            "VECTOR_STORE_TYPE": ("vector_store", "type", str),
            "CHROMA_PERSIST_DIR": ("vector_store", "persist_dir", str),
            "CHROMA_COLLECTION_NAME": ("vector_store", "collection_name", str),
            # Document
            "CHUNK_SIZE": ("document", "chunk_size", int),
            "CHUNK_OVERLAP": ("document", "chunk_overlap", int),
            # Retrieval
            "SIMILARITY_TOP_K": ("retrieval", "similarity_top_k", int),
            "SIMILARITY_THRESHOLD": ("retrieval", "similarity_threshold", float),
            # RAG
            "DATA_DIR": ("rag", "data_dir", str),
        }

        for env_key, (section, field_key, converter) in env_mappings.items():
            env_value = os.environ.get(env_key)
            if env_value is not None:
                try:
                    config[section][field_key] = converter(env_value)
                except (ValueError, TypeError):
                    pass

    def _parse_config(self) -> None:
        """解析配置到具体配置对象"""
        self.server = ServerConfig(**self._config["server"])
        self.logging = LoggingConfig(**self._config["logging"])
        self.llm = LLMConfig(**self._config["llm"])
        self.llm_providers = LLMProvidersConfig(**self._config["llm_providers"])
        self.embedding = EmbeddingConfig(**self._config["embedding"])
        self.vector_store = VectorStoreConfig(**self._config["vector_store"])
        self.document = DocumentConfig(**self._config["document"])
        self.retrieval = RetrievalConfig(**self._config["retrieval"])
        self.rag = RAGConfig(**self._config["rag"])
        self.evaluation = EvaluationConfig(**self._config["evaluation"])

    def reload(self) -> None:
        """重新加载配置"""
        self._config = self._load_config()
        self._parse_config()

    # === 属性快捷方式 ===
    @property
    def DEBUG(self) -> bool:
        return self.server.debug

    @property
    def PORT(self) -> int:
        return self.server.port

    @property
    def HOST(self) -> str:
        return self.server.host

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return self.server.cors_origins

    @property
    def LOG_LEVEL(self) -> str:
        return self.logging.level

    @property
    def LOG_FILE_PATH(self) -> str:
        return self.logging.file_path

    @property
    def LOG_ROTATION(self) -> str:
        return self.logging.rotation

    @property
    def LOG_RETENTION(self) -> str:
        return self.logging.retention

    @property
    def LLM_PROVIDER(self) -> str:
        return self.llm.provider

    @property
    def TEMPERATURE(self) -> float:
        return self.llm.temperature

    @property
    def MAX_TOKENS(self) -> int:
        return self.llm.max_tokens

    @property
    def GENERATION_TIMEOUT(self) -> int:
        return self.llm.timeout

    # DeepSeek
    @property
    def DEEPSEEK_API_KEY(self) -> str:
        return self.llm_providers.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")

    @property
    def DEEPSEEK_API_BASE(self) -> str:
        return self.llm_providers.deepseek_api_base

    @property
    def DEEPSEEK_MODEL(self) -> str:
        return self.llm_providers.deepseek_model

    # Kimi
    @property
    def KIMI_API_KEY(self) -> str:
        return self.llm_providers.kimi_api_key or os.environ.get("KIMI_API_KEY", "")

    @property
    def KIMI_API_BASE(self) -> str:
        return self.llm_providers.kimi_api_base

    @property
    def KIMI_MODEL(self) -> str:
        return self.llm_providers.kimi_model

    # GLM
    @property
    def GLM_API_KEY(self) -> str:
        return self.llm_providers.glm_api_key or os.environ.get("GLM_API_KEY", "")

    @property
    def GLM_API_BASE(self) -> str:
        return self.llm_providers.glm_api_base

    @property
    def GLM_MODEL(self) -> str:
        return self.llm_providers.glm_model

    # Embedding
    @property
    def EMBEDDING_PROVIDER(self) -> str:
        return self.embedding.provider

    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.embedding.model

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        return self.embedding.dimension

    @property
    def EMBEDDING_BATCH_SIZE(self) -> int:
        return self.embedding.batch_size

    # Vector Store
    @property
    def VECTOR_STORE_TYPE(self) -> str:
        return self.vector_store.type

    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        return self.vector_store.persist_dir

    @property
    def CHROMA_COLLECTION_NAME(self) -> str:
        return self.vector_store.collection_name

    # Document
    @property
    def CHUNK_SIZE(self) -> int:
        return self.document.chunk_size

    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.document.chunk_overlap

    @property
    def CHUNKING_STRATEGY(self) -> str:
        return self.document.chunking_strategy

    @property
    def SUPPORTED_EXTENSIONS(self) -> list[str]:
        return self.document.supported_extensions

    # Retrieval
    @property
    def SIMILARITY_TOP_K(self) -> int:
        return self.retrieval.similarity_top_k

    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        return self.retrieval.similarity_threshold

    @property
    def RERANK_ENABLED(self) -> bool:
        return self.retrieval.rerank_enabled

    @property
    def RERANK_MODEL(self) -> str:
        return self.retrieval.rerank_model

    @property
    def VECTOR_WEIGHT(self) -> float:
        return self.retrieval.vector_weight

    @property
    def KEYWORD_WEIGHT(self) -> float:
        return self.retrieval.keyword_weight

    # RAG
    @property
    def DATA_DIR(self) -> str:
        return self.rag.data_dir

    @property
    def OUTPUT_DIR(self) -> str:
        return self.rag.output_dir

    @property
    def MAX_SOURCES(self) -> int:
        return self.rag.max_sources

    # Evaluation
    @property
    def EVALUATION_ENABLED(self) -> bool:
        return self.evaluation.enabled

    # ===== 通用方法 =====
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号路径访问"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_llm_config(self, provider: str | None = None) -> dict[str, Any]:
        """获取指定 provider 的 LLM 配置"""
        provider = provider or self.LLM_PROVIDER
        base_config = {
            "provider": provider,
            "temperature": self.TEMPERATURE,
            "max_tokens": self.MAX_TOKENS,
        }

        if provider == "deepseek":
            base_config.update({
                "api_key": self.DEEPSEEK_API_KEY,
                "api_base": self.DEEPSEEK_API_BASE,
                "model": self.DEEPSEEK_MODEL,
            })
        elif provider == "kimi":
            base_config.update({
                "api_key": self.KIMI_API_KEY,
                "api_base": self.KIMI_API_BASE,
                "model": self.KIMI_MODEL,
            })
        elif provider == "glm":
            base_config.update({
                "api_key": self.GLM_API_KEY,
                "api_base": self.GLM_API_BASE,
                "model": self.GLM_MODEL,
            })

        return base_config

    def get_embedding_config(self) -> dict[str, Any]:
        """获取 embedding 配置"""
        return {
            "provider": self.EMBEDDING_PROVIDER,
            "model": self.EMBEDDING_MODEL,
            "dimension": self.EMBEDDING_DIMENSION,
        }

    def get_rag_config(self) -> dict[str, Any]:
        """获取 RAG 配置"""
        return {
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP,
            "chunking_strategy": self.CHUNKING_STRATEGY,
            "similarity_top_k": self.SIMILARITY_TOP_K,
            "similarity_threshold": self.SIMILARITY_THRESHOLD,
            "rerank_enabled": self.RERANK_ENABLED,
            "vector_weight": self.VECTOR_WEIGHT,
            "keyword_weight": self.KEYWORD_WEIGHT,
        }


# 创建全局配置实例
settings: Final[Settings] = Settings()

__all__ = [
    "ServerConfig",
    "LoggingConfig",
    "LLMConfig",
    "LLMProvidersConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "DocumentConfig",
    "RetrievalConfig",
    "RAGConfig",
    "EvaluationConfig",
    "Settings",
    "settings",
]
