#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用配置管理
支持从环境变量和YAML配置文件读取配置
"""

import os
from typing import Final, Any
from pathlib import Path
import yaml


class Settings:
    """应用配置类"""

    # 配置文件路径
    CONFIG_FILE_PATH: Final[str] = 'config.yaml'

    # 默认配置
    _DEFAULT_CONFIG: Final[dict[str, Any]] = {
        'server': {
            'debug': True,
            'port': 8000
        },
        'logging': {
            'level': 'INFO',
            'file_path': 'logs/app.log',
            'rotation': '1 day',
            'retention': '7 days'
        },
        'agent': {
            'main_agent': {
                'max_iterations': 50,
                'timeout': 300,
                'verbose': True
            },
            'subagents': {
                'researcher': {'max_results': 10, 'timeout': 120},
                'analyst': {'timeout': 180},
                'writer': {'timeout': 240}
            }
        },
        'research': {
            'output_dir': 'reports',
            'templates_dir': 'templates',
            'max_sources': 20,
            'default_topic': 'general'
        }
    }

    def __init__(self) -> None:
        """初始化配置"""
        self._config: dict[str, Any] = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """加载配置

        优先级：环境变量 > YAML配置文件 > 默认配置

        Returns:
            dict[str, Any]: 合并后的配置
        """
        config: dict[str, Any] = self._DEFAULT_CONFIG.copy()

        # 从YAML文件加载配置
        config_file: Path = Path(self.CONFIG_FILE_PATH)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config: dict[str, Any] = yaml.safe_load(f) or {}
                # 合并YAML配置
                self._merge_config(config, file_config)
            except Exception as e:
                print(f"警告: 无法加载配置文件 {self.CONFIG_FILE_PATH}: {e}")

        # 从环境变量加载配置（优先级最高）
        self._load_from_env(config)

        return config

    def _merge_config(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """递归合并配置

        Args:
            base: 基础配置字典
            override: 要覆盖的配置字典
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _load_from_env(self, config: dict[str, Any]) -> None:
        """从环境变量加载配置

        Args:
            config: 配置字典
        """
        # 服务配置
        if os.environ.get('DEBUG'):
            config['server']['debug'] = os.environ.get('DEBUG').lower() == 'true'
        if os.environ.get('PORT'):
            config['server']['port'] = int(os.environ.get('PORT'))

        # LLM 配置
        if os.environ.get('TEMPERATURE'):
            try:
                config.setdefault('llm', {})['temperature'] = float(os.environ.get('TEMPERATURE'))
            except ValueError:
                pass

    # 服务配置
    @property
    def DEBUG(self) -> bool:
        return self._config['server']['debug']

    @property
    def PORT(self) -> int:
        return self._config['server']['port']

    # 日志配置
    @property
    def LOG_LEVEL(self) -> str:
        return self._config['logging']['level']

    @property
    def LOG_FILE_PATH(self) -> str:
        return self._config['logging']['file_path']

    @property
    def LOG_ROTATION(self) -> str:
        return self._config['logging']['rotation']

    @property
    def LOG_RETENTION(self) -> str:
        return self._config['logging']['retention']

    # Agent 配置
    @property
    def AGENT_CONFIG(self) -> dict[str, Any]:
        return self._config.get('agent', {})

    @property
    def MAIN_AGENT_CONFIG(self) -> dict[str, Any]:
        return self.AGENT_CONFIG.get('main_agent', {})

    @property
    def SUBAGENTS_CONFIG(self) -> dict[str, Any]:
        return self.AGENT_CONFIG.get('subagents', {})

    # 研究配置
    @property
    def RESEARCH_CONFIG(self) -> dict[str, Any]:
        return self._config.get('research', {})

    @property
    def OUTPUT_DIR(self) -> str:
        return self.RESEARCH_CONFIG.get('output_dir', 'reports')

    @property
    def MAX_SOURCES(self) -> int:
        return self.RESEARCH_CONFIG.get('max_sources', 20)

    # LLM 配置
    @property
    def LLM_PROVIDER(self) -> str:
        """获取 LLM 提供商

        支持的提供商: deepseek (默认), kimi, glm, openai (暂不支持)
        """
        return os.environ.get('LLM_PROVIDER', 'deepseek')

    @property
    def TEMPERATURE(self) -> float:
        return float(os.environ.get('TEMPERATURE', '0.1'))

    @property
    def MAX_TOKENS(self) -> int:
        return int(os.environ.get('MAX_TOKENS', '2048'))

    # DeepSeek 配置
    @property
    def DEEPSEEK_API_KEY(self) -> str:
        return os.environ.get('DEEPSEEK_API_KEY', '')

    @property
    def DEEPSEEK_API_BASE(self) -> str:
        return os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')

    @property
    def DEEPSEEK_MODEL(self) -> str:
        return os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')

    # Kimi 配置
    @property
    def KIMI_API_KEY(self) -> str:
        return os.environ.get('KIMI_API_KEY', '')

    @property
    def KIMI_API_BASE(self) -> str:
        return os.environ.get('KIMI_API_BASE', 'https://api.moonshot.cn/v1')

    @property
    def KIMI_MODEL(self) -> str:
        return os.environ.get('KIMI_MODEL', 'moonshot-v1-8k')

    # GLM 配置
    @property
    def GLM_API_KEY(self) -> str:
        return os.environ.get('GLM_API_KEY', '')

    @property
    def GLM_API_BASE(self) -> str:
        return os.environ.get('GLM_API_BASE', 'https://open.bigmodel.cn/api/paas/v4')

    @property
    def GLM_MODEL(self) -> str:
        return os.environ.get('GLM_MODEL', 'glm-4-flash')

    # Embedding 配置
    @property
    def EMBEDDING_PROVIDER(self) -> str:
        return os.environ.get('EMBEDDING_PROVIDER', 'deepseek')

    @property
    def EMBEDDING_MODEL(self) -> str:
        return os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002')

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        return int(os.environ.get('EMBEDDING_DIMENSION', '1536'))

    # Vector Store 配置
    @property
    def VECTOR_STORE_TYPE(self) -> str:
        return os.environ.get('VECTOR_STORE_TYPE', 'chroma')

    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        return os.environ.get('CHROMA_PERSIST_DIR', './storage/chroma')

    @property
    def CHROMA_COLLECTION_NAME(self) -> str:
        return os.environ.get('CHROMA_COLLECTION_NAME', 'llamaindex_docs')

    # RAG 配置
    @property
    def CHUNK_SIZE(self) -> int:
        return int(os.environ.get('CHUNK_SIZE', '512'))

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(os.environ.get('CHUNK_OVERLAP', '50'))

    @property
    def SIMILARITY_TOP_K(self) -> int:
        return int(os.environ.get('SIMILARITY_TOP_K', '5'))

    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        return float(os.environ.get('SIMILARITY_THRESHOLD', '0.5'))

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


# 创建全局配置实例
settings: Final[Settings] = Settings()
