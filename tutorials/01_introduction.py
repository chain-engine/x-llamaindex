"""
LlamaIndex 入门教程

本教程介绍了 LlamaIndex 的核心概念和基本使用方法，包括：
1. LlamaIndex 简介
2. 核心组件
3. 基本工作流程
4. 实践示例
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, 
    Document, ServiceContext
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 加载环境变量
load_dotenv()

# 配置 OpenAI API 密钥
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("警告：未设置 OPENAI_API_KEY 环境变量")
    print("本教程将使用模拟数据进行演示")

# 打印教程标题
print("=" * 60)
print("LlamaIndex 入门教程")
print("=" * 60)

# 1. LlamaIndex 简介
print("\n1. LlamaIndex 简介")
print("-" * 40)
print("LlamaIndex 是一个专为大语言模型（LLM）应用设计的数据编排框架，")
print("主要用于构建检索增强生成（RAG）系统。它提供了一套工具和接口，")
print("帮助开发者更有效地处理、索引和查询外部数据，从而增强 LLM 的能力。")

# 2. 核心组件
print("\n2. 核心组件")
print("-" * 40)
print("LlamaIndex 的核心组件包括：")
print("  - Documents：表示原始文本数据")
print("  - Nodes：表示文本的段落或片段")
print("  - Indexes：用于组织和索引 Nodes")
print("  - Retrievers：用于从 Indexes 中检索相关 Nodes")
print("  - Query Engines：用于执行查询并生成回答")
print("  - Chat Engines：用于构建对话式应用")

# 3. 基本工作流程
print("\n3. 基本工作流程")
print("-" * 40)
print("LlamaIndex 的基本工作流程：")
print("  1. 加载数据（Documents）")
print("  2. 分割数据为 Nodes")
print("  3. 构建 Index")
print("  4. 创建 Query Engine")
print("  5. 执行查询")

# 4. 实践示例
print("\n4. 实践示例")
print("-" * 40)
print("现在我们将通过一个简单的示例来演示 LlamaIndex 的使用方法。")

# 创建模拟数据
print("\n4.1 创建模拟数据")
sample_documents = [
    Document(
        text="LlamaIndex 是一个数据编排框架，专为大语言模型应用设计。它提供了一套工具和接口，帮助开发者更有效地处理、索引和查询外部数据。",
        metadata={"source": "introduction", "section": "overview"}
    ),
    Document(
        text="RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的技术，通过从外部知识库检索相关信息来增强大语言模型的生成能力。",
        metadata={"source": "introduction", "section": "rag"}
    ),
    Document(
        text="LlamaIndex 的核心组件包括：Documents、Nodes、Indexes、Retrievers、Query Engines 和 Chat Engines。",
        metadata={"source": "introduction", "section": "components"}
    ),
    Document(
        text="使用 LlamaIndex 构建 RAG 系统的基本步骤：加载数据、分割数据、构建索引、创建查询引擎、执行查询。",
        metadata={"source": "introduction", "section": "workflow"}
    )
]

print("成功创建 4 个示例文档")

# 检查是否设置了 API 密钥
if openai_api_key:
    # 4.2 构建索引
    print("\n4.2 构建索引")
    print("正在构建向量索引...")
    
    # 创建服务上下文
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo"),
        embed_model=OpenAIEmbedding(model="text-embedding-ada-002")
    )
    
    # 构建索引
    index = VectorStoreIndex.from_documents(
        sample_documents,
        service_context=service_context
    )
    
    print("索引构建完成")
    
    # 4.3 创建查询引擎
    print("\n4.3 创建查询引擎")
    query_engine = index.as_query_engine()
    print("查询引擎创建完成")
    
    # 4.4 执行查询
    print("\n4.4 执行查询")
    test_queries = [
        "什么是 LlamaIndex？",
        "RAG 是什么？",
        "LlamaIndex 的核心组件有哪些？",
        "如何使用 LlamaIndex 构建 RAG 系统？"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n查询 {i+1}: {query}")
        response = query_engine.query(query)
        print(f"回答: {response.response}")
        print(f"参考来源: {response.source_nodes[0].node.metadata.get('section', 'Unknown')}")
else:
    print("\n警告：由于未设置 OPENAI_API_KEY，")
    print("我们将跳过实际的索引构建和查询步骤。")
    print("要运行完整示例，请设置 OPENAI_API_KEY 环境变量。")

# 5. 总结
print("\n5. 总结")
print("-" * 40)
print("本教程介绍了 LlamaIndex 的基本概念和使用方法，包括：")
print("  - LlamaIndex 的核心功能和定位")
print("  - 主要组件及其作用")
print("  - 基本工作流程")
print("  - 简单的实践示例")
print("\n要深入学习 LlamaIndex，建议参考以下资源：")
print("  - LlamaIndex 官方文档：https://docs.llamaindex.ai/")
print("  - LlamaIndex GitHub 仓库：https://github.com/run-llama/llama_index")

print("\n" + "=" * 60)
print("教程结束")
print("=" * 60)
