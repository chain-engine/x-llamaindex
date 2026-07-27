#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 数据模型

定义 FastAPI 的请求和响应模型
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求"""

    query: str = Field(..., description="查询字符串", min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=5, description="返回的文档数量", ge=1, le=20)
    include_sources: Optional[bool] = Field(default=True, description="是否包含来源信息")
    stream: Optional[bool] = Field(default=False, description="是否使用流式响应")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is LlamaIndex?",
                    "top_k": 5,
                    "include_sources": True,
                    "stream": False,
                }
            ]
        }
    }


class SourceNode(BaseModel):
    """来源节点"""

    node_id: str = Field(..., description="节点 ID")
    text: str = Field(..., description="节点文本")
    score: Optional[float] = Field(default=None, description="相似度分数")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class QueryResponse(BaseModel):
    """查询响应"""

    response: str = Field(..., description="生成的响应")
    sources: Optional[List[SourceNode]] = Field(default=None, description="来源节点")
    source_count: Optional[int] = Field(default=None, description="来源数量")
    latency: Optional[float] = Field(default=None, description="响应延迟（秒）")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class ChatRequest(BaseModel):
    """对话请求"""

    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None, description="会话 ID")
    stream: Optional[bool] = Field(default=False, description="是否使用流式响应")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Tell me about RAG systems.",
                    "session_id": "user-123",
                    "stream": False,
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """对话响应"""

    response: str = Field(..., description="助手响应")
    session_id: str = Field(..., description="会话 ID")
    message_count: Optional[int] = Field(default=None, description="消息数量")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""

    texts: List[str] = Field(..., description="文档文本列表", min_length=1)
    metadatas: Optional[List[Dict[str, Any]]] = Field(default=None, description="元数据列表")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "texts": [
                        "LlamaIndex is a data orchestration framework for LLM applications.",
                        "RAG combines retrieval and generation for better responses.",
                    ],
                    "metadatas": [
                        {"source": "doc1", "category": "framework"},
                        {"source": "doc2", "category": "technique"},
                    ],
                }
            ]
        }
    }


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""

    success: bool = Field(..., description="是否成功")
    document_count: int = Field(..., description="上传的文档数量")
    message: Optional[str] = Field(default=None, description="消息")


class IndexStatusResponse(BaseModel):
    """索引状态响应"""

    status: str = Field(..., description="状态")
    document_count: int = Field(default=0, description="文档数量")
    node_count: int = Field(default=0, description="节点数量")
    index_type: str = Field(default="vector", description="索引类型")
    vector_store_type: str = Field(default="chroma", description="向量存储类型")
    message: Optional[str] = Field(default=None, description="消息")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本号")
    uptime: Optional[float] = Field(default=None, description="运行时间（秒）")
    components: Dict[str, str] = Field(default_factory=dict, description="组件状态")


class ErrorResponse(BaseModel):
    """错误响应"""

    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    detail: Optional[str] = Field(default=None, description="详细信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
