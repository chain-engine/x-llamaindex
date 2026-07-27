"""
基础 RAG 系统示例

本示例展示了如何使用 LlamaIndex 构建一个基础的检索增强生成（RAG）系统。
包括：
1. 加载文档
2. 构建索引
3. 创建查询引擎
4. 执行查询
"""

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from src.utils import setup_environment, get_llm

# 加载环境变量
load_dotenv()
setup_environment()

# 获取 LLM（使用项目的配置系统，支持 DeepSeek/Kimi/GLM）
llm = get_llm()

# 1. 加载文档
print("正在加载文档...")
documents = SimpleDirectoryReader(
    input_dir="data",
    required_exts=[".txt", ".pdf", ".docx"]
).load_data()

print(f"成功加载 {len(documents)} 个文档")

# 2. 构建索引
print("正在构建索引...")
index = VectorStoreIndex.from_documents(documents)

print("索引构建完成")

# 3. 创建查询引擎
print("正在创建查询引擎...")
query_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=3
)

print("查询引擎创建完成")

# 4. 执行查询
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
