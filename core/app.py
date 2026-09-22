from core.vectorstore import build_vectorstore, get_vectorstore

# 云端首次部署时向量库不存在，自动构建（本地已有库则直接跳过）
try:
    get_vectorstore()
except FileNotFoundError:
    with st.spinner("首次启动，正在构建知识库（约 1-3 分钟）..."):
        build_vectorstore()