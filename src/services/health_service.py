#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health Service

健康检查业务逻辑层
"""

from typing import Any, Dict

from src.schemas import HealthResponse


class HealthService:
    """健康检查服务"""

    def __init__(self, rag_service: Any):
        """初始化健康检查服务

        Args:
            rag_service: RAG 服务实例
        """
        self.rag_service = rag_service

    def check(self) -> HealthResponse:
        """执行健康检查

        Returns:
            健康检查响应
        """
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            components={
                "api": "ok",
                "index": "ok" if self.rag_service.is_initialized else "not_initialized",
            },
        )