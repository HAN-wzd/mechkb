"""RAG 问答链：检索 → 拼 Prompt → LLM 生成带引用的回答"""
import json
import os
from dataclasses import dataclass, field

from langchain_core.documents import Document

from core.llm import chat, chat_stream
from core.vectorstore import search

HISTORY_FILE = "chat_history.json"  # 会话历史持久化文件（已在 .gitignore）

SYSTEM_PROMPT = """你是一名严谨的机械工程知识助手，服务对象是工厂的工程师。

回答规则：
1. 只根据用户消息中提供的【参考资料】回答问题，禁止使用资料之外的知识。
2. 如果参考资料不足以回答问题，直接说明"参考资料中未提及"，不要猜测或编造。
3. 回答要专业、简洁、分点，适合工程师快速阅读。
4. 回答末尾用一行标注依据，格式：【依据：资料N（来源文件名）】。

示例格式：
45号钢最常用的热处理方式是调质（淬火+高温回火）...
【依据：资料1（金属材料与热处理.md）】"""


def _build_context(docs: list[Document]) -> str:
    """把检索到的 chunk 拼成带编号和来源的参考资料块"""
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        blocks.append(f"【资料{i}｜来源：{source}】\n{doc.page_content}")
    return "\n\n".join(blocks)


@dataclass
class ChatSession:
    """一次多轮会话：维护对话历史，供追问使用；历史持久化到本地 JSON"""
    history: list = field(default_factory=list)  # [{"role": ..., "content": ...}, ...]

    def __post_init__(self):
        # 启动时恢复历史：浏览器 F5 刷新后 Streamlit 会重建 Session，靠文件找回
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, encoding="utf-8") as f:
                self.history = json.load(f)

    def _save(self):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def clear(self):
        """清空对话（含持久化文件）"""
        self.history = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)

    def ask(self, question: str, k: int = 3) -> str:
        """问一个问题，返回带引用的回答；对话历史自动累积"""
        # ① 检索：只根据"当前问题"检索（不是整段历史，避免追问时检索漂移）
        docs = search(question, k=k)
        context = _build_context(docs)

        # ② 组装消息：system 规则 + 历史对话 + 本轮问题（带资料）
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({
            "role": "user",
            "content": f"参考资料：\n{context}\n\n问题：{question}",
        })

        # ③ 调 LLM 生成回答（Day 2 的重试/超时封装在这里生效）
        answer = chat(messages)

        # ④ 把本轮问答追加进历史（追问时模型有上下文）并落盘
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self._save()
        return answer, docs

    
    def ask_stream(self, question: str, k: int = 3):
        """
        流式版 ask：逐段 yield 回答内容；结束后本轮问答自动入历史。
        检索到的文档存在 self.last_docs，供前端展示引用来源。
        """
        docs = search(question, k=k)
        self.last_docs = docs
        context = _build_context(docs)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({
            "role": "user",
            "content": f"参考资料：\n{context}\n\n问题：{question}",
        })

        pieces = []
        for piece in chat_stream(messages):   # Day 2 的流式封装在这里复用
            pieces.append(piece)
            yield piece

        answer = "".join(pieces)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self._save()

if __name__ == "__main__":
    # 多轮对话自测
    session = ChatSession()

    print("=" * 50)
    print("第1问（知识库内有）")
    print("=" * 50)
    answer, docs = session.ask("45号钢常用的热处理方式有哪些？")
    print(answer)
    print("\n检索到的来源:", [d.metadata["source"] for d in docs])

    print("\n" + "=" * 50)
    print("第2问（追问，考验多轮上下文）")
    print("=" * 50)
    answer, docs = session.ask("它和正火的主要区别是什么？")
    print(answer)

    print("\n" + "=" * 50)
    print("第3问（知识库外，考验幻觉抑制）")
    print("=" * 50)
    answer, docs = session.ask("钛合金 TC4 的激光焊接参数怎么设定？")
    print(answer)