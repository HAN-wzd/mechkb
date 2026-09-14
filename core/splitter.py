"""文档切分：把长文档切成适合检索的 chunk"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    chunk_size: 每段最大字符数（Day 6 会实验 200/500/1000 三档对比）
    chunk_overlap: 相邻 chunk 的重叠字符数，防止关键句被切断在边界
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 分隔符按优先级从前往后尝试：先按空行(段落)、再换行(行)、再中文句号、
        # 分号、逗号，最后才硬切。这就是"Recursive(递归)"的含义——尽量在自然边界断开
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[splitter] {len(docs)} 个文档 → {len(chunks)} 个 chunk (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


if __name__ == "__main__":
    from core.loader import load_documents

    chunks = split_documents(load_documents())
    # 打印前 3 个 chunk 看看切得怎么样
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- chunk {i} | 来源: {c.metadata['source']} | {len(c.page_content)}字符 ---")
        print(c.page_content[:200])