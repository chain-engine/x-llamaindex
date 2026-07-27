#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Health API Routes

健康检查相关路由
"""

from fastapi import APIRouter

from src.schemas import HealthResponse

router = APIRouter()


@router.get("", response_model=HealthResponse, tags=["健康检查"])
async def health_check(
    health_service = Depends(lambda request: request.app.state.health_service),
) -> HealthResponse:
    """健康检查端点"""
    return health_service.check()
