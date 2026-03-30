基于 RAG + qwen:1.8b 实现的本地成语接龙小游戏

## 项目介绍
使用 LangChain + FAISS 向量库 + Ollama 本地大模型，严格从本地成语库中进行成语接龙，不会乱造词。

## 技术栈
- Python
- LangChain
- FAISS 向量检索
- HuggingFace Embedding
- Ollama(qwen:1.8b)

## 使用方法
1. 准备 `idiom.txt`，每行一个四字成语
2. 安装依赖
   ```bash
   pip install langchain langchain_community langchain_huggingface faiss-cpu
3.备有本地的大模型
