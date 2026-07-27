#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业级 RAG 系统示例

本示例展示了如何使用项目模块构建一个完整的企业级 RAG 系统。

功能包括：
1. 多格式文档加载
2. 智能文本分割
3. 向量索引管理
4. 混合检索和重排序
5. 查询和对话引擎
6. 系统评估
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.core import get_logger, settings
from src.utils import setup_environment, get_llm, get_embedding_model
from src.loaders import DocumentLoader
from src.processors import TextSplitter, ChunkingStrategy, MetadataExtractor
from src.storage import VectorStoreManager
from src.indexes import IndexManager, IndexType
from src.retrievers import HybridRetriever, Reranker
from src.engines import RAGQueryEngine, RAGChatEngine, ConversationManager
from src.evaluators import RAGEvaluator

logger = get_logger(__name__)


class EnterpriseRAGSystem:
    """企业级 RAG 系统

    整合所有组件，提供完整的 RAG 功能

    Example:
        >>> rag = EnterpriseRAGSystem(data_dir="./data")
        >>> # 查询
        >>> response = rag.query("What is LlamaIndex?")
        >>> # 对话
        >>> rag.chat("Tell me more about RAG.")
    """

    def __init__(
        self,
        data_dir: str = "./data",
        persist_dir: str = "./storage",
        use_vector_store: bool = True,
        use_hybrid_retrieval: bool = True,
        use_reranker: bool = False,
    ):
        """初始化企业级 RAG 系统

        Args:
            data_dir: 数据目录
            persist_dir: 持久化目录
            use_vector_store: 是否使用持久化向量存储
            use_hybrid_retrieval: 是否使用混合检索
            use_reranker: 是否使用重排序
        """
        self.data_dir = data_dir
        self.persist_dir = persist_dir

        # 设置环境
        setup_environment()

        # 初始化组件
        self.llm = None
        self.embed_model = None
        self.vector_store_manager = None
        self.index_manager = None
        self.query_engine = None
        self.chat_engine = None
        self.conversation_manager = None
        self.evaluator = None

        # 配置选项
        self.use_vector_store = use_vector_store
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.use_reranker = use_reranker

        # 初始化系统
        self._initialize()

    def _initialize(self) -> None:
        """初始化所有组件"""
        logger.info("Initializing Enterprise RAG System...")

        # 1. 创建 LLM 和嵌入模型
        self.llm = get_llm()
        self.embed_model = get_embedding_model()
        logger.info("LLM and embedding model initialized")

        # 2. 创建向量存储管理器
        if self.use_vector_store:
            self.vector_store_manager = VectorStoreManager(
                persist_dir=os.path.join(self.persist_dir, "chroma"),
            )
            logger.info("Vector store manager initialized")

        # 3. 创建索引管理器
        self.index_manager = IndexManager(
            index_type=IndexType.VECTOR,
            vector_store_manager=self.vector_store_manager,
            embed_model=self.embed_model,
            persist_dir=os.path.join(self.persist_dir, "index"),
        )
        logger.info("Index manager initialized")

        # 4. 加载文档并构建索引
        self._load_and_index_documents()

        # 5. 创建查询引擎
        self.query_engine = RAGQueryEngine(
            index=self.index_manager.index,
            llm=self.llm,
            use_hybrid_retrieval=self.use_hybrid_retrieval,
            use_reranker=self.use_reranker,
            verbose=True,
        )
        logger.info("Query engine initialized")

        # 6. 创建对话管理器
        self.conversation_manager = ConversationManager(
            index=self.index_manager.index,
            llm=self.llm,
        )
        logger.info("Conversation manager initialized")

        # 7. 创建评估器
        self.evaluator = RAGEvaluator(llm=self.llm)
        logger.info("Evaluator initialized")

        logger.info("Enterprise RAG System initialized successfully")

    def _load_and_index_documents(self) -> None:
        """加载文档并构建索引"""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return

        # 检查索引是否已存在
        if self.index_manager.document_count > 0:
            logger.info(f"Index already exists with {self.index_manager.document_count} documents")
            return

        logger.info(f"Loading documents from {self.data_dir}...")

        # 1. 加载文档
        loader = DocumentLoader()
        documents = loader.load(self.data_dir)

        if not documents:
            logger.warning("No documents found")
            return

        logger.info(f"Loaded {len(documents)} documents")

        # 2. 提取元数据
        extractor = MetadataExtractor()
        documents = extractor.enrich_documents(documents)

        # 3. 分割文档
        splitter = TextSplitter(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=512,
            chunk_overlap=50,
        )
        nodes = splitter.split_documents(documents)

        logger.info(f"Split into {len(nodes)} nodes")

        # 4. 构建索引
        self.index_manager.build_from_nodes(nodes)

        logger.info(f"Index built with {self.index_manager.node_count} nodes")

    def query(self, query_str: str, include_sources: bool = True) -> dict:
        """执行查询

        Args:
            query_str: 查询字符串
            include_sources: 是否包含来源信息

        Returns:
            查询结果字典
        """
        if include_sources:
            return self.query_engine.query_with_sources(query_str)

        response = self.query_engine.query(query_str)
        return {"response": response.response}

    def chat(self, message: str, session_id: str = None) -> dict:
        """执行对话

        Args:
            message: 用户消息
            session_id: 会话 ID（可选）

        Returns:
            对话结果
        """
        # 获取或创建会话
        if session_id is None:
            session_id = self.conversation_manager.create_session()
        elif session_id not in self.conversation_manager.get_session_ids():
            self.conversation_manager.create_session(session_id=session_id)

        response = self.conversation_manager.chat(session_id, message)

        return {
            "response": response.response,
            "session_id": session_id,
        }

    def add_documents(self, texts: list, metadatas: list = None) -> int:
        """添加新文档

        Args:
            texts: 文本列表
            metadatas: 元数据列表

        Returns:
            添加的节点数量
        """
        from llama_index.core import Document

        # 创建文档
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(text=text, metadata=metadata)
            documents.append(doc)

        # 分割并添加到索引
        splitter = TextSplitter(strategy=ChunkingStrategy.SENTENCE)
        nodes = splitter.split_documents(documents)

        self.index_manager.insert_nodes(nodes)

        logger.info(f"Added {len(nodes)} nodes to index")
        return len(nodes)

    def get_stats(self) -> dict:
        """获取系统统计信息"""
        return {
            "index": self.index_manager.get_stats(),
            "query_engine": self.query_engine.get_stats(),
            "chat": {
                "active_sessions": self.conversation_manager.get_active_session_count(),
            },
        }


def demo_query(rag: EnterpriseRAGSystem):
    """演示查询功能"""
    print("\n" + "=" * 60)
    print("查询演示")
    print("=" * 60)

    queries = [
        "什么是 LlamaIndex？",
        "LlamaIndex 的核心功能有哪些？",
        "LlamaIndex 和 LangChain 有什么区别？",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        print("-" * 40)

        result = rag.query(query)

        print(f"回答: {result['response']}")

        if "sources" in result and result["sources"]:
            print("\n参考来源:")
            for i, source in enumerate(result["sources"][:3], 1):
                print(f"  {i}. [{source.score:.4f}] {source.metadata.get('file_name', 'Unknown')}")

        print()


def demo_chat(rag: EnterpriseRAGSystem):
    """演示对话功能"""
    print("\n" + "=" * 60)
    print("对话演示")
    print("=" * 60)

    session_id = None

    messages = [
        "你好，请介绍一下 LlamaIndex。",
        "它主要用于什么场景？",
        "如何开始使用它？",
    ]

    for message in messages:
        print(f"\n用户: {message}")

        result = rag.chat(message, session_id)
        session_id = result["session_id"]

        print(f"助手: {result['response']}")

    print(f"\n会话 ID: {session_id}")


def demo_evaluation(rag: EnterpriseRAGSystem):
    """演示评估功能"""
    print("\n" + "=" * 60)
    print("评估演示")
    print("=" * 60)

    # 评估查询
    query = "What is RAG?"
    result = rag.query(query, include_sources=False)

    # 评估生成质量
    metrics = rag.evaluator.evaluate_generation(
        query=query,
        response=result["response"],
    )

    print(f"查询: {query}")
    print(f"响应: {result['response'][:200]}...")
    print("\n评估指标:")
    print(f"  相关性: {metrics.relevancy:.4f}")
    print(f"  连贯性: {metrics.coherence:.4f}")

    # 获取聚合指标
    aggregated = rag.evaluator.get_aggregated_metrics()
    print("\n聚合指标:")
    print(f"  总查询数: {aggregated.total_queries}")


def main():
    """主函数"""
    print("=" * 60)
    print("企业级 RAG 系统示例")
    print("=" * 60)

    # 加载环境变量
    load_dotenv()
    setup_environment()

    # 检查 API 密钥（使用项目配置系统）
    llm = get_llm()
    if not llm:
        print("\n警告: 无法创建 LLM 实例")
        print("请在 .env 中配置 LLM_PROVIDER 和对应的 API Key")
        print("\n演示将使用模拟数据...")
        demo_with_mock_data()
        return

    # 创建 RAG 系统
    print("\n初始化 RAG 系统...")
    rag = EnterpriseRAGSystem(
        data_dir="./data",
        persist_dir="./storage",
        use_vector_store=True,
        use_hybrid_retrieval=True,
    )

    # 获取系统统计
    print("\n系统统计:")
    stats = rag.get_stats()
    print(f"  索引文档数: {stats['index'].get('document_count', 0)}")
    print(f"  索引节点数: {stats['index'].get('node_count', 0)}")

    # 演示查询
    demo_query(rag)

    # 演示对话
    demo_chat(rag)

    # 演示评估
    demo_evaluation(rag)

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


def demo_with_mock_data():
    """使用模拟数据进行演示"""
    print("\n使用模拟数据演示系统架构...")

    # 显示系统架构
    print("""
    企业级 RAG 系统架构:

    ┌─────────────────────────────────────────────────────────────┐
    │                      用户接口层                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
    │  │  CLI 接口    │  │  REST API    │  │  Web UI      │       │
    │  └──────────────┘  └──────────────┘  └──────────────┘       │
    └─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                      应用层                                  │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
    │  │  Query Engine│  │ Chat Engine  │  │  Evaluator   │       │
    │  └──────────────┘  └──────────────┘  └──────────────┘       │
    └─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                      检索层                                  │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
    │  │ Hybrid Retr. │  │   Reranker   │  │   Index Mgr  │       │
    │  └──────────────┘  └──────────────┘  └──────────────┘       │
    └─────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────────────────────────────────────────┐
    │                      数据层                                  │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
    │  │ Document     │  │   Vector     │  │   Text       │       │
    │  │ Loader       │  │   Store      │  │   Splitter   │       │
    │  └──────────────┘  └──────────────┘  └──────────────┘       │
    └─────────────────────────────────────────────────────────────┘
    """)

    print("\n核心模块说明:")
    print("  - loaders:      多格式文档加载（TXT, PDF, DOCX, MD, JSON）")
    print("  - processors:   文本分割和元数据提取")
    print("  - storage:      向量存储管理（ChromaDB）")
    print("  - indexes:      索引创建和管理")
    print("  - retrievers:   混合检索和重排序")
    print("  - engines:      查询引擎和对话引擎")
    print("  - evaluators:   RAG 系统评估")
    print("  - api:          FastAPI RESTful 服务")
    print("  - utils:        通用工具函数")


if __name__ == "__main__":
    main()
