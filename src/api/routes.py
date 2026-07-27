#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 路由定义

定义 FastAPI 的路由端点
"""

import time
from typing import Optional, AsyncIterator

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

from src.schemas import (
    QueryRequest,
    QueryResponse,
    ChatRequest,
    ChatResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    IndexStatusResponse,
    HealthResponse,
    ErrorResponse,
    SourceNode,
)
from src.core import get_logger

logger = get_logger(__name__)

router = APIRouter()

# 全局 RAG 系统实例（通过 app.state 共享）
_rag_system = None


def get_rag_system():
    """获取 RAG 系统实例"""
    global _rag_system
    if _rag_system is None:
        from src.engines.query_engine import QueryEngine
        from src.engines.chat_engine import ChatEngine
        from src.indexes.index_manager import IndexManager
        from src.core.container import container

        # 注册服务到容器
        if not container.is_registered(IndexManager):
            container.register(IndexManager, lambda: IndexManager())
        if not container.is_registered(QueryEngine):
            container.register(QueryEngine, lambda: QueryEngine(
                index_manager=container.resolve(IndexManager)
            ))
        if not container.is_registered(ChatEngine):
            container.register(ChatEngine, lambda: ChatEngine(
                index_manager=container.resolve(IndexManager)
            ))

        # 简单的 RAG 系统封装
        class SimpleRAGSystem:
            def __init__(self):
                self.index_manager = container.resolve(IndexManager)
                self.query_engine = container.resolve(QueryEngine)
                self.conversation_manager = container.resolve(ChatEngine)

            @property
            def is_initialized(self):
                return getattr(self.index_manager, 'index', None) is not None

        _rag_system = SimpleRAGSystem()

    return _rag_system


@router.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查端点"""
    rag = get_rag_system()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components={
            "api": "ok",
            "index": "ok" if rag.is_initialized else "not_initialized",
        },
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["查询"],
)
async def query(request: QueryRequest):
    """执行 RAG 查询

    根据查询字符串从知识库中检索相关信息并生成响应
    """
    start_time = time.time()

    try:
        rag = get_rag_system()

        if getattr(rag.index_manager, "index", None) is None:
            raise HTTPException(
                status_code=503,
                detail="Index not ready. Provide valid API keys (and restart) or upload documents first.",
            )

        if request.stream:
            # 流式响应
            async def stream_generator():
                for text in rag.query_engine.stream_query(request.query):
                    yield f"data: {text}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )

        # 同步查询
        result = rag.query_engine.query_with_sources(request.query)

        # 构建响应
        sources = None
        if request.include_sources:
            sources = [
                SourceNode(
                    node_id=s["node_id"],
                    text=s["text"],
                    score=s["score"],
                    metadata=s["metadata"],
                )
                for s in result.get("sources", [])
            ]

        latency = time.time() - start_time
        logger.info(f"Query completed in {latency:.2f}s: {request.query[:50]}...")

        return QueryResponse(
            response=result["response"],
            sources=sources,
            source_count=result.get("source_count"),
            latency=latency,
        )

    except Exception as e:
        # 如果已经是 HTTPException（例如 Index not ready 503），不要再包装成 500
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["对话"],
)
async def chat(request: ChatRequest):
    """执行多轮对话

    支持基于会话的多轮对话
    """
    try:

        # 获取或创建会话
        session_id = request.session_id
        if not session_id:
            session_id = rag.conversation_manager.create_session()
        elif session_id not in rag.conversation_manager.get_session_ids():
            rag.conversation_manager.create_session(session_id=session_id)

        # 发送消息
        response = rag.conversation_manager.chat(session_id, request.message)

        # 获取消息数量
        message_count = len(
            rag.conversation_manager._sessions[session_id].get_chat_history()
        )

        return ChatResponse(
            response=response.response,
            session_id=session_id,
            message_count=message_count,
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/stream",
    responses={500: {"model": ErrorResponse}},
    tags=["对话"],
)
async def chat_stream(request: ChatRequest):
    """流式对话

    返回流式响应
    """
    try:
        rag = get_rag_system()

        # 获取或创建会话
        session_id = request.session_id
        if not session_id:
            session_id = rag.conversation_manager.create_session()
        elif session_id not in rag.conversation_manager.get_session_ids():
            rag.conversation_manager.create_session(session_id=session_id)

        async def stream_generator():
            for text in rag.conversation_manager._sessions[session_id].stream_chat(
                request.message
            ):
                yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error(f"Chat stream failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/documents",
    response_model=DocumentUploadResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["文档管理"],
)
async def upload_documents(request: DocumentUploadRequest):
    """上传文档到知识库

    将文档添加到向量索引中
    """
    try:
        rag = get_rag_system()

        # 创建文档
        from llama_index.core import Document

        documents = []
        for i, text in enumerate(request.texts):
            metadata = request.metadatas[i] if request.metadatas and i < len(request.metadatas) else {}
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)

        # 添加到索引
        rag.index_manager.insert_documents(documents)

        logger.info(f"Uploaded {len(documents)} documents")

        return DocumentUploadResponse(
            success=True,
            document_count=len(documents),
            message=f"Successfully uploaded {len(documents)} documents",
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/index/status",
    response_model=IndexStatusResponse,
    tags=["文档管理"],
)
async def get_index_status():
    """获取索引状态

    返回当前索引的统计信息
    """
    try:
        rag = get_rag_system()
        stats = rag.index_manager.get_stats()

        return IndexStatusResponse(
            status="ready",
            document_count=stats.get("document_count", 0),
            node_count=stats.get("node_count", 0),
            index_type=stats.get("index_type", "vector"),
            vector_store_type="chroma" if stats.get("has_vector_store") else "memory",
        )

    except Exception as e:
        logger.error(f"Failed to get index status: {e}")
        return IndexStatusResponse(
            status="error",
            message=str(e),
        )


@router.delete(
    "/chat/{session_id}",
    tags=["对话"],
)
async def delete_session(session_id: str):
    """删除对话会话

    清除指定会话的历史记录
    """
    try:
        rag = get_rag_system()
        rag.conversation_manager.delete_session(session_id)

        return {"success": True, "message": f"Session {session_id} deleted"}

    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats",
    tags=["系统"],
)
async def get_stats():
    """获取系统统计信息

    返回查询和对话的统计信息
    """
    try:
        rag = get_rag_system()

        query_engine_stats = None
        if getattr(rag.index_manager, "index", None) is not None:
            query_engine_stats = rag.query_engine.get_stats()

        return {
            "query_engine": query_engine_stats,
            "chat_engine": {
                "active_sessions": rag.conversation_manager.get_active_session_count(),
            },
            "index": rag.index_manager.get_stats(),
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
