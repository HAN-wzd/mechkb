"""向量库：Embedding + ChromaDB 入库与检索"""
import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from core.loader import load_documents
from core.splitter import split_documents

load_dotenv()

# 用硅基流动的 bge-m3 模型做 Embedding（免费、中文效果好、OpenAI 兼容格式）
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-m3",
    # 不检查上下文长度（bge-m3 接口与 OpenAI 原生行为略有差异，关掉这个检查避免报错）
    check_embedding_ctx_length=False,
)

PERSIST_DIR = "chroma_db"  # 向量库落盘位置，已在 .gitignore 里
COLLECTION = "mechkb"      # 集合名：固定下来，重建时删集合而不是删文件夹


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=PERSIST_DIR)


def build_vectorstore(chunk_size: int = 500, chunk_overlap: int = 50) -> Chroma:
    """全量构建向量库：加载 → 切分 → 向量化 → 入库（重复调用会先清空集合重建）"""
    client = _client()
    try:
        # 重建用"删集合"而不是 shutil.rmtree 删文件夹：
        # Windows 下文件夹里的 SQLite 文件被 Chroma 占用时会报 PermissionError
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # 集合不存在（首次构建/手动删过文件夹），跳过
    docs = load_documents()
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    # from_documents 内部会逐个 chunk 调 Embedding API
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,  # 落盘，下次启动不用重新入库
    )
    print(f"[vectorstore] 入库完成，共 {len(chunks)} 个 chunk，保存于 ./{PERSIST_DIR}")
    return vectorstore


def get_vectorstore() -> Chroma:
    """加载已存在的向量库（Day 4/5 问答链直接用这个，不重复入库）"""
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError("向量库不存在，先运行 build_vectorstore()")
    return Chroma(
        client=_client(),
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )


def search(query: str, k: int = 3) -> list[Document]:
    """检索与 query 最相关的 k 个 chunk"""
    return get_vectorstore().similarity_search(query, k=k)


if __name__ == "__main__":
    # 命令行单独运行本文件 = 重建向量库并做一次检索自测
    build_vectorstore()

    print("\n=== 检索自测 ===")
    for q in ["45号钢怎么做调质", "冲压件毛刺过大怎么排查", "什么是基孔制"]:
        print(f"\n问题: {q}")
        for i, doc in enumerate(search(q, k=3), 1):
            print(f"  {i}. [{doc.metadata['source']}] {doc.page_content[:80]}...")
