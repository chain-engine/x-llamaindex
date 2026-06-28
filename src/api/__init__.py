#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 模块

提供 FastAPI RESTful API 服务
"""

from src.api.app import create_app, get_app
from src.api.routes import router
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    ChatRequest,
    ChatResponse,
    DocumentUploadRequest,
    IndexStatusResponse,
    HealthResponse,
)

__all__ = [
    "create_app",
    "get_app",
    "router",
    "QueryRequest",
    "QueryResponse",
    "ChatRequest",
    "ChatResponse",
    "DocumentUploadRequest",
    "IndexStatusResponse",
    "HealthResponse",
]
