"""MechKB Web 界面：Streamlit 实现"""
import streamlit as st

from core.chain import ChatSession
from core.vectorstore import build_vectorstore, get_vectorstore

st.set_page_config(page_title="MechKB 机械知识库", page_icon="🔧", layout="wide")

st.title("MechKB — 机械工艺知识库问答助手")
st.caption("基于 RAG：回答附带原文出处；资料里没有的问题会如实告知，不编造。")

# 云端首次部署时向量库不存在（chroma_db 不进仓库），自动构建；
# 本地已有库则 get_vectorstore() 成功，直接跳过
try:
    get_vectorstore()
except FileNotFoundError:
    with st.spinner("首次启动，正在构建知识库（约 1-3 分钟）..."):
        build_vectorstore()

# session_state：Streamlit 每次交互都会重跑整个脚本，
# 把会话对象存在这里才能跨交互保留多轮对话历史
if "session" not in st.session_state:
    st.session_state.session = ChatSession()

# ---------- 侧边栏：知识库管理 ----------
with st.sidebar:
    st.header("知识库管理")

    if st.button("🔄 重建知识库", help="语料变更后重新切分并入库，约 1-3 分钟"):
        with st.spinner("切分并向量化中..."):
            build_vectorstore()
        st.success("知识库重建完成")

    st.divider()
    uploaded = st.file_uploader("上传新文档（md / txt）", type=["md", "txt"])
    if uploaded is not None and st.button("📥 入库"):
        # 先存到 data/ 目录，再全量重建（build 内部会先清旧库）
        with open(f"data/{uploaded.name}", "wb") as f:
            f.write(uploaded.getbuffer())
        with st.spinner(f"已保存 {uploaded.name}，重建知识库中..."):
            build_vectorstore()
        st.success(f"{uploaded.name} 已入库")

    st.divider()
    st.markdown("**内置语料**：金属材料与热处理 · 机械加工工艺 · 设备维护与故障排查")

    st.divider()
    if st.button("🗑️ 清空对话", help="清空当前对话历史（含持久化文件）"):
        st.session_state.session.clear()
        st.rerun()

# ---------- 主区：对话 ----------
# 先渲染历史消息（刷新页面后仍在，因为历史在 session_state 里）
for msg in st.session_state.session.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入框（固定在页面底部）
question = st.chat_input("例如：45号钢怎么做调质？")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 流式渲染：用 empty 占位符不断覆盖更新
        placeholder = st.empty()
        answer = ""
        for piece in st.session_state.session.ask_stream(question):
            answer += piece
            placeholder.markdown(answer + "▌")   # ▌ 是打字机光标
        placeholder.markdown(answer)              # 播完后去掉光标

    # 引用来源：本轮检索到的 chunk
    docs = getattr(st.session_state.session, "last_docs", [])
    if docs:
        with st.expander("📚 检索到的参考资料"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**资料{i}** ｜ 来源：`{d.metadata['source']}`")
                st.caption(d.page_content[:150] + "...")