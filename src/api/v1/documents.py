#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documents API Routes

文档管理相关路由
"""

from fastapi import APIRouter, HTTPException, Depends

from src.schemas import DocumentUploadRequest, DocumentUploadResponse, IndexStatusResponse
from src.core import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=DocumentUploadResponse,
    tags=["文档管理"],
)
async def upload_documents(
    request: DocumentUploadRequest,
    document_service = Depends(lambda request: request.app.state.document_service),
):
    """上传文档到知识库

    将文档添加到向量索引中
    """
    try:
        result = document_service.upload_documents(
            texts=request.texts,
            metadatas=request.metadatas,
        )
        return DocumentUploadResponse(**result)

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/status",
    response_model=IndexStatusResponse,
    tags=["文档管理"],
)
async def get_index_status(
    document_service = Depends(lambda request: request.app.state.document_service),
):
    """获取索引状态

    返回当前索引的统计信息
    """
    try:
        result = document_service.get_index_status()
        return IndexStatusResponse(**result)

    except Exception as e:
        logger.error(f"Failed to get index status: {e}")
        return IndexStatusResponse(status="error", message=str(e))
