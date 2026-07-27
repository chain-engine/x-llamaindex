#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Service

文档管理业务逻辑层
"""

from typing import Any, Dict, List, Optional

from llama_index.core import Document

from src.core import get_logger

logger = get_logger(__name__)


class DocumentService:
    """文档服务

    负责文档的添加、查询、删除等业务逻辑
    """

    def __init__(self, index_manager: Any):
        """初始化文档服务

        Args:
            index_manager: 索引管理器实例
        """
        self.index_manager = index_manager

    def upload_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """上传文档到知识库

        Args:
            texts: 文档文本列表
            metadatas: 元数据列表

        Returns:
            上传结果
        """
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)

        self.index_manager.insert_documents(documents)

        logger.info(f"Uploaded {len(documents)} documents")

        return {
            "success": True,
            "document_count": len(documents),
            "message": f"Successfully uploaded {len(documents)} documents",
        }

    def get_index_status(self) -> Dict[str, Any]:
        """获取索引状态

        Returns:
            索引状态信息
        """
        stats = self.index_manager.get_stats()

        return {
            "status": "ready",
            "document_count": stats.get("document_count", 0),
            "node_count": stats.get("node_count", 0),
            "index_type": stats.get("index_type", "vector"),
            "vector_store_type": "chroma" if stats.get("has_vector_store") else "memory",
        }

    def delete_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """删除文档

        Args:
            doc_ids: 文档ID列表

        Returns:
            删除结果
        """
        self.index_manager.delete(doc_ids)
        logger.info(f"Deleted {len(doc_ids)} documents")

        return {
            "success": True,
            "deleted_count": len(doc_ids),
            "message": f"Successfully deleted {len(doc_ids)} documents",
        }
