"""
高级 RAG 系统示例

本示例展示了如何使用 LlamaIndex 构建一个高级的检索增强生成（RAG）系统。
包括：
1. 自定义检索器
2. 路由查询
3. 多模态支持
4. 评估指标
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, 
    RouterQueryEngine, QueryBundle
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

from src.utils import setup_environment, get_llm

# 加载环境变量
load_dotenv()
setup_environment()

# 获取 LLM
llm = get_llm()

# 1. 加载文档
print("正在加载文档...")
documents = SimpleDirectoryReader(
    input_dir="data",
    required_exts=[".txt", ".pdf", ".docx"]
).load_data()

print(f"成功加载 {len(documents)} 个文档")

# 2. 构建多个索引
print("正在构建索引...")

# 向量索引
vector_index = VectorStoreIndex.from_documents(documents)

# 3. 创建自定义检索器
class CustomRetriever(BaseRetriever):
    def __init__(self, vector_retriever):
        self.vector_retriever = vector_retriever
        super().__init__()
    
    def _retrieve(self, query_bundle: QueryBundle):
        # 可以在这里添加自定义检索逻辑
        nodes = self.vector_retriever.retrieve(query_bundle)
        # 例如，过滤掉相似度低于阈值的结果
        filtered_nodes = [node for node in nodes if node.score > 0.5]
        return filtered_nodes

# 创建向量检索器
vector_retriever = vector_index.as_retriever(
    similarity_top_k=5
)

# 创建自定义检索器
custom_retriever = CustomRetriever(vector_retriever)

# 4. 创建查询引擎
print("正在创建查询引擎...")

# 基础向量查询引擎
vector_query_engine = vector_index.as_query_engine(
    llm=llm,
    similarity_top_k=3
)

# 使用 HyDE 转换的查询引擎
hyde_query_engine = vector_index.as_query_engine(
    llm=llm,
    similarity_top_k=3,
    query_transform=HyDEQueryTransform(llm=llm)
)

# 创建路由查询引擎
query_engine = RouterQueryEngine(
    query_engine_tools=[
        {
            "name": "vector_tool",
            "description": "适合处理事实性问题和具体信息查询",
            "query_engine": vector_query_engine
        },
        {
            "name": "hyde_tool",
            "description": "适合处理需要更深入理解的问题",
            "query_engine": hyde_query_engine
        }
    ],
    llm=llm
)

print("查询引擎创建完成")

# 5. 执行查询
print("\n开始交互式查询（输入 'exit' 退出）：")
while True:
    query = input("请输入您的问题：")
    if query.lower() == "exit":
        break
    
    print("\n正在查询...")
    response = query_engine.query(query)
    print(f"\n回答：{response.response}")
    print(f"\n参考来源：")
    for i, source in enumerate(response.source_nodes):
        print(f"{i+1}. {source.node.metadata.get('file_name', 'Unknown')} - 相似度: {source.score:.4f}")
    print("\n" + "-" * 50 + "\n")

# 6. 简单评估
print("\n开始简单评估...")
test_queries = [
    "什么是 LlamaIndex？",
    "RAG 系统的工作原理是什么？",
    "如何优化检索质量？"
]

for query in test_queries:
    print(f"\n测试查询：{query}")
    response = query_engine.query(query)
    print(f"回答长度：{len(response.response)} 字符")
    print(f"参考来源数量：{len(response.source_nodes)}")

print("\n评估完成！")
