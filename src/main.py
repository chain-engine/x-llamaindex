#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point

FastAPI 应用入口
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.resolve()
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.logger import logger
from core.config import settings

from services import DocumentService, RAGService, ChatService, HealthService
from indexes.index_manager import IndexManager
from engines.query_engine import RAGQueryEngine
from engines.chat_engine import ConversationManager


# ============================================================================
# 全局实例（应用级别单例）
# ============================================================================
_index_manager: IndexManager | None = None
_document_service: DocumentService | None = None
_rag_service: RAGService | None = None
_chat_service: ChatService | None = None
_health_service: HealthService | None = None


def _get_index_manager() -> IndexManager:
    """获取索引管理器实例"""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager


def _get_document_service(
    request: Request,
) -> DocumentService:
    """依赖注入：获取文档服务"""
    global _document_service
    if _document_service is None:
        index_manager = _get_index_manager()
        _document_service = DocumentService(index_manager=index_manager)
    return _document_service


def _get_rag_service(
    request: Request,
) -> RAGService:
    """依赖注入：获取 RAG 服务"""
    global _rag_service
    if _rag_service is None:
        index_manager = _get_index_manager()
        # 创建查询引擎
        query_engine = RAGQueryEngine(
            index=index_manager.index,
            streaming=False,
        )
        _rag_service = RAGService(
            index_manager=index_manager,
            query_engine=query_engine,
        )
    return _rag_service


def _get_chat_service(
    request: Request,
) -> ChatService:
    """依赖注入：获取对话服务"""
    global _chat_service
    if _chat_service is None:
        index_manager = _get_index_manager()
        conversation_manager = ConversationManager(index=index_manager.index)
        _chat_service = ChatService(conversation_manager=conversation_manager)
    return _chat_service


def _get_health_service(
    request: Request,
) -> HealthService:
    """依赖注入：获取健康检查服务"""
    global _health_service
    if _health_service is None:
        rag_service = _get_rag_service(request)
        _health_service = HealthService(rag_service=rag_service)
    return _health_service


# 类型别名
DocumentServiceDep = Annotated[DocumentService, Depends(_get_document_service)]
RAGServiceDep = Annotated[RAGService, Depends(_get_rag_service)]
ChatServiceDep = Annotated[ChatService, Depends(_get_chat_service)]
HealthServiceDep = Annotated[HealthService, Depends(_get_health_service)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting x-LlamaIndex application...")

    try:
        index_manager = _get_index_manager()
        logger.info(f"IndexManager initialized: {getattr(index_manager, 'index', None) is not None}")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Shutting down x-LlamaIndex application...")
        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    from api.v1 import health, documents, rag

    app = FastAPI(
        title="x-LlamaIndex API",
        description="A production-ready LlamaIndex RAG learning and training project",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 配置中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(health.router, prefix="/api/v1/health", tags=["健康检查"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["文档管理"])
    app.include_router(rag.router, prefix="/api/v1/rag", tags=["检索增强生成"])

    # 根路径
    @app.get("/", tags=["根路径"])
    async def root():
        """根路径"""
        return {
            "name": "x-LlamaIndex",
            "version": "0.1.0",
            "description": "A production-ready LlamaIndex learning and training project",
            "docs": "/docs",
            "redoc": "/redoc",
        }

    # 全局异常处理
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        """请求验证异常处理器"""
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=400,
            content={
                "code": 400,
                "message": "Validation failed",
                "errors": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """通用异常处理器"""
        logger.opt(exception=True).error("Unhandled exception occurred")
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "message": "Internal server error",
                "detail": str(exc) if settings.DEBUG else None,
            },
        )

    return app


app = create_app()


def run():
    """CLI 入口点"""
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug",
    )


if __name__ == "__main__":
    run()
