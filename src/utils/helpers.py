#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数

提供通用的辅助函数
"""

import os
import time
import functools
from typing import Optional, Any, Dict, List, Callable, TypeVar
from pathlib import Path

from dotenv import load_dotenv

from src.core import get_logger, settings

logger = get_logger(__name__)

T = TypeVar("T")


def setup_environment(env_file: Optional[str] = None) -> None:
    """设置环境变量

    Args:
        env_file: .env 文件路径，默认为当前目录下的 .env
    """
    env_path = Path(env_file) if env_file else Path.cwd() / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"Environment file not found: {env_path}")

    # 验证必要的环境变量（根据选择的提供商）
    provider = os.getenv("LLM_PROVIDER", "deepseek")
    provider_keys = {
        "deepseek": "DEEPSEEK_API_KEY",
        "kimi": "KIMI_API_KEY",
        "glm": "GLM_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    required_key = provider_keys.get(provider)
    if required_key and not os.getenv(required_key):
        logger.warning(f"Missing required environment variable: {required_key} for provider: {provider}")


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> Any:
    """获取 LLM 实例

    Args:
        provider: LLM 提供商 (deepseek, kimi, glm, openai)
                  默认使用 deepseek
        model: 模型名称
        temperature: 温度参数
        **kwargs: 额外参数

    Returns:
        LLM 实例

    Supported Providers:
        - deepseek: DeepSeek 模型 (默认)
        - kimi: Moonshot Kimi 模型
        - glm: 智谱 GLM 模型
        - openai: OpenAI 模型 (暂不支持)
    """
    provider = provider or os.getenv("LLM_PROVIDER", "deepseek")
    temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", "0.1"))

    # DeepSeek 提供商 (默认)
    if provider == "deepseek":
        from llama_index.llms.openai import OpenAI

        default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return OpenAI(
            model=model or default_model,
            temperature=temperature,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            **kwargs,
        )

    # Kimi (Moonshot) 提供商
    elif provider == "kimi":
        from llama_index.llms.openai import OpenAI

        default_model = os.getenv("KIMI_MODEL", "moonshot-v1-8k")
        return OpenAI(
            model=model or default_model,
            temperature=temperature,
            api_key=os.getenv("KIMI_API_KEY"),
            api_base=os.getenv("KIMI_API_BASE", "https://api.moonshot.cn/v1"),
            **kwargs,
        )

    # GLM (智谱) 提供商
    elif provider == "glm":
        from llama_index.llms.openai import OpenAI

        default_model = os.getenv("GLM_MODEL", "glm-4-flash")
        return OpenAI(
            model=model or default_model,
            temperature=temperature,
            api_key=os.getenv("GLM_API_KEY"),
            api_base=os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
            **kwargs,
        )

    # OpenAI 提供商 (暂不支持)
    elif provider == "openai":
        logger.warning("OpenAI provider is currently not supported. Please use deepseek, kimi, or glm instead.")
        raise ValueError(
            "OpenAI provider is currently not supported. "
            "Please use one of the supported providers: deepseek, kimi, glm"
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: deepseek, kimi, glm"
        )


def get_embedding_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dimension: Optional[int] = None,
    **kwargs,
) -> Any:
    """获取嵌入模型实例

    Args:
        provider: 嵌入模型提供商 (deepseek, kimi, glm, local)
                  默认使用 deepseek
        model: 模型名称
        dimension: 嵌入维度
        **kwargs: 额外参数

    Returns:
        嵌入模型实例

    Supported Providers:
        - deepseek: DeepSeek 嵌入模型 (默认)
        - kimi: Moonshot Kimi 嵌入模型
        - glm: 智谱 GLM 嵌入模型
        - local: 本地嵌入模型 (使用 HuggingFace)
    """
    from llama_index.embeddings.openai import OpenAIEmbedding

    provider = provider or os.getenv("EMBEDDING_PROVIDER", "deepseek")
    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    dimension = dimension or int(os.getenv("EMBEDDING_DIMENSION", "1536"))

    # DeepSeek 嵌入模型 (默认)
    if provider == "deepseek":
        return OpenAIEmbedding(
            model=model,
            dimensions=dimension,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
            **kwargs,
        )

    # Kimi (Moonshot) 嵌入模型
    elif provider == "kimi":
        return OpenAIEmbedding(
            model=model,
            dimensions=dimension,
            api_key=os.getenv("KIMI_API_KEY"),
            api_base=os.getenv("KIMI_API_BASE", "https://api.moonshot.cn/v1"),
            **kwargs,
        )

    # GLM (智谱) 嵌入模型
    elif provider == "glm":
        return OpenAIEmbedding(
            model=model,
            dimensions=dimension,
            api_key=os.getenv("GLM_API_KEY"),
            api_base=os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"),
            **kwargs,
        )

    # 本地嵌入模型 (使用 HuggingFace)
    elif provider == "local":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            return HuggingFaceEmbedding(
                model_name=model or "BAAI/bge-small-zh-v1.5",
                **kwargs,
            )
        except ImportError:
            logger.warning("HuggingFace embedding not available. Install with: pip install llama-index-embeddings-huggingface")
            raise ValueError(
                "HuggingFace embedding requires llama-index-embeddings-huggingface package. "
                "Install it with: pip install llama-index-embeddings-huggingface"
            )

    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: deepseek, kimi, glm, local"
        )


def format_response(
    response: Any,
    include_sources: bool = True,
    max_source_length: int = 200,
) -> Dict[str, Any]:
    """格式化响应

    Args:
        response: 查询响应
        include_sources: 是否包含来源信息
        max_source_length: 来源文本最大长度

    Returns:
        格式化的响应字典
    """
    result = {
        "response": response.response if hasattr(response, "response") else str(response),
    }

    if include_sources and hasattr(response, "source_nodes"):
        sources = []
        for node in response.source_nodes:
            text = node.node.text
            if len(text) > max_source_length:
                text = text[:max_source_length] + "..."

            sources.append({
                "text": text,
                "score": node.score,
                "metadata": node.node.metadata,
            })

        result["sources"] = sources
        result["source_count"] = len(sources)

    return result


def timer(func: Callable[..., T]) -> Callable[..., T]:
    """计时装饰器

    Args:
        func: 被装饰的函数

    Returns:
        装饰后的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
        return result

    return wrapper


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def batch_iterator(items: List[Any], batch_size: int):
    """批量迭代器

    Args:
        items: 项目列表
        batch_size: 批次大小

    Yields:
        批次项目
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-3.5-turbo",
) -> float:
    """计算 API 调用成本

    Args:
        input_tokens: 输入 token 数
        output_tokens: 输出 token 数
        model: 模型名称

    Returns:
        成本（美元）
    """
    # 价格表（每 1K tokens，美元）
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0},
    }

    model_pricing = pricing.get(model, {"input": 0, "output": 0})

    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]

    return input_cost + output_cost


class Singleton(type):
    """单例元类"""

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
