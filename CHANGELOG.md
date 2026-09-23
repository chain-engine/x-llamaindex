# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-23

### Added

- **端到端 RAG 流水线**：文档加载（TXT/PDF/DOCX/MD/JSON/Web）、文本切分、元数据提取、Embedding 向量化、Chroma 向量存储
- **混合检索与重排序**：向量 + BM25 混合检索，可选重排序链路
- **FastAPI 服务化**：REST API 提供健康检查、RAG 查询、多轮对话、文档摄入、索引状态查询等端点
- **Swagger / ReDoc / OpenAPI 文档**：自动生成交互式 API 文档
- **多 LLM 提供商支持**：DeepSeek、Kimi (Moonshot)、GLM (智谱) 可切换
- **分层配置**：`.env` 环境变量 + `config.yaml` YAML 配置，与代码解耦
- **Docker 部署**：多阶段构建 Dockerfile，docker-compose 开发/生产编排，健康检查，卷持久化
- **测试框架**：pytest 覆盖加载器、处理器、索引、引擎等模块
- **示例与教程**：基础/高级/企业级 RAG 示例脚本，入门教程
- **双语文档**：README.md（中文）、README.en.md（英文）
