#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI 应用

创建和配置 FastAPI 应用实例
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils import setup_environment, get_llm, get_embedding_model
from src.core import get_logger, settings

logger = get_logger(__name__)

# 应用启动时间
_start_time: float = 0


class RAGSystem:
    """RAG 系统封装类"""

    def __init__(self):
        """初始化 RAG 系统"""
        self.index_manager = None
        self.query_engine = None
        self.conversation_manager = None
        self.llm = None
        self.embed_model = None
        self.vector_store_manager = None

    def initialize(
        self,
        data_dir: Optional[str] = None,
        use_vector_store: bool = True,
    ) -> None:
        """初始化 RAG 系统组件

        Args:
            data_dir: 数据目录
            use_vector_store: 是否使用向量存储
        """
        logger.info("Initializing RAG system...")

        # 设置环境
        setup_environment()

        # 创建 LLM 和嵌入模型
        self.llm = get_llm()
        self.embed_model = get_embedding_model()

        # 创建向量存储管理器
        if use_vector_store:
            from src.storage import VectorStoreManager

            self.vector_store_manager = VectorStoreManager()

        # 创建索引管理器
        from src.indexes import IndexManager, IndexType

        self.index_manager = IndexManager(
            index_type=IndexType.VECTOR,
            vector_store_manager=self.vector_store_manager,
            embed_model=self.embed_model,
        )

        # 如果有数据目录，加载文档并构建索引
        if data_dir and os.path.exists(data_dir):
            self._load_and_index(data_dir)

        # 创建查询引擎
        from src.engines import RAGQueryEngine

        self.query_engine = RAGQueryEngine(
            index=self.index_manager.index,
            llm=self.llm,
        )

        # 创建对话管理器
        from src.engines import ConversationManager

        self.conversation_manager = ConversationManager(
            index=self.index_manager.index,
            llm=self.llm,
        )

        logger.info("RAG system initialized successfully")

    def _load_and_index(self, data_dir: str) -> None:
        """加载文档并构建索引

        Args:
            data_dir: 数据目录
        """
        from src.loaders import DocumentLoader
        from src.processors import TextSplitter, ChunkingStrategy

        logger.info(f"Loading documents from {data_dir}...")

        # 加载文档
        loader = DocumentLoader()
        documents = loader.load(data_dir)

        if not documents:
            logger.warning(f"No documents found in {data_dir}")
            return

        # 分割文档
        splitter = TextSplitter(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=512,
            chunk_overlap=50,
        )
        nodes = splitter.split_documents(documents)

        # 构建索引
        try:
            self.index_manager.build_from_nodes(nodes)
        except Exception as e:
            # 例如 embeddings API key 缺失/失效时，这里会抛出 AuthenticationError
            # 为了让服务可以启动，失败时跳过索引构建，后续查询接口会返回 Index not ready。
            logger.error(f"Failed to build index from nodes: {e}")
            return

        logger.info(f"Indexed {len(nodes)} nodes from {len(documents)} documents")


# 全局 RAG 系统实例
_rag_system: Optional[RAGSystem] = None


def init_rag_system() -> RAGSystem:
    """初始化并返回 RAG 系统实例"""
    global _rag_system

    if _rag_system is None:
        _rag_system = RAGSystem()
        data_dir = os.getenv("DATA_DIR", "./data")
        _rag_system.initialize(data_dir=data_dir)

    return _rag_system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _start_time
    _start_time = time.time()

    # 启动时初始化
    logger.info("Starting RAG API server...")
    init_rag_system()

    yield

    # 关闭时清理
    logger.info("Shutting down RAG API server...")


def create_app(
    title: str = "RAG API",
    description: str = "Enterprise-grade RAG System API powered by LlamaIndex",
    version: str = "0.1.0",
) -> FastAPI:
    """创建 FastAPI 应用实例

    Args:
        title: 应用标题
        description: 应用描述
        version: 版本号

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get("server.cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(router, prefix="/api/v1")

    # 根路径
    @app.get("/")
    async def root():
        return {
            "name": title,
            "version": version,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    logger.info(f"FastAPI app created: {title} v{version}")
    return app


def get_app() -> FastAPI:
    """获取应用实例（用于 uvicorn）"""
    return create_app()


# 用于 uvicorn 直接运行
app = get_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.get("server.host", "0.0.0.0"),
        port=settings.PORT,
        reload=settings.DEBUG,
    )
