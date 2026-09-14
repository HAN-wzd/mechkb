"""文档加载：支持 Markdown / TXT / PDF"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(data_dir: str = "data") -> list[Document]:
    """读取 data 目录下所有支持的文档，返回 Document 列表"""
    docs: list[Document] = []
    for path in sorted(Path(data_dir).glob("*")):
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            loaded = TextLoader(str(path), encoding="utf-8").load()
        elif suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            print(f"[loader] 跳过不支持的文件: {path.name}")
            continue
        # 给每个 Document 标记来源文件名 —— Day 4 引用溯源全靠它
        for doc in loaded:
            doc.metadata["source"] = path.name
        docs.extend(loaded)
    print(f"[loader] 共加载 {len(docs)} 个文档")
    return docs


if __name__ == "__main__":
    for d in load_documents():
        print(f"- {d.metadata['source']}: {len(d.page_content)} 字符")