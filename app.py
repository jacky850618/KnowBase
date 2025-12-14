# app.py
import os
import streamlit as st

# ==================== 持久化配置 ====================
from config_manager import load_config, save_config, get_proxy_url

if "config" not in st.session_state:
    st.session_state.config = load_config()

config = st.session_state.config

# 应用代理
proxy_url = get_proxy_url(config)
if proxy_url:
    os.environ["HTTP_PROXY"] = proxy_url
    os.environ["HTTPS_PROXY"] = proxy_url
else:
    for key in ["HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(key, None)

# ==================== 其他导入 ====================
from knowledge_manager import (
    add_document, delete_document, list_documents, get_document_content
)
from embeddings import get_embedding_model
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from config import CHROMA_DB_PATH

# ==================== Streamlit 全局配置 ====================
st.set_page_config(
    page_title="AI知识库问答系统",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "# 基于 RAG + DeepSeek/ChatGPT 的知识库系统\n由 Grok 协助构建"
    }
)

# ==================== 当前模型信息 ====================
current_provider_key = config["model_provider"]
current_provider = config["providers"][current_provider_key]


# ==================== 初始化组件 ====================
@st.cache_resource
def get_retriever():
    embedding = get_embedding_model()
    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding,
        collection_name="rag_collection"
    )
    return vectordb.as_retriever(search_kwargs={"k": config["rag_settings"]["retriever_k"]})


@st.cache_resource
def get_llm():
    if not current_provider["api_key"]:
        return None  # 未配置 Key 时不创建
    return ChatOpenAI(
        model=current_provider["model_name"],
        api_key=current_provider["api_key"],
        base_url=current_provider["base_url"],
        temperature=0.3,
        max_tokens=2048,
        timeout=60.0,
    )


@st.cache_resource
def get_rag_chain():
    llm = get_llm()
    if llm is None:
        st.error("请先在【管理后台】→【系统设置】中配置 API Key")
        st.stop()
    retriever = get_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个严谨、乐于助人的AI助手。请仅基于以下检索到的上下文回答问题。\n"
                   "如果上下文没有相关信息，请回答“我不知道”，不要胡编。\n\n上下文：{context}"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# ==================== 侧边栏导航 ====================
st.sidebar.title("🗂️ 导航")
page = st.sidebar.radio(
    "选择页面",
    ["💬 智能问答", "⚙️ 管理后台"],
    index=0
)

# ==================== 页面 1: 智能问答（默认首页） ====================
if page == "💬 智能问答":
    st.title("🧠 欢迎使用你的知识库问答系统")
    st.markdown(
        """
        你可以向我提问任何关于**你上传的文档**的内容，我会基于知识库精准回答。  
        当前使用模型：**{name}** (`{model}`)
        """.format(
            name=current_provider["name"],
            model=current_provider["model_name"]
        )
    )

    # 显示文档统计
    docs_count = len(list_documents())
    st.info(f"📚 当前知识库中共有 **{docs_count}** 个文档可供查询")
    st.caption(
        "💡 小提示：文档数指你上传的文件数量，“块数”是系统自动将长文档切分成的小段，用于提升检索精度和回答质量，完全正常～")

    if docs_count == 0:
        st.warning("⚠️ 知识库为空，请到【管理后台】上传文档后即可提问")
        st.stop()

    # 问答输入
    question = st.text_input(
        "请输入你的问题：",
        placeholder="例如：公司年假政策是什么？项目进度如何？",
        key="main_question"
    )

    if st.button("🚀 发送问题", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("请输入问题内容")
        else:
            with st.spinner("正在检索知识库并生成答案..."):
                try:
                    rag_chain = get_rag_chain()
                    result = rag_chain.invoke({"input": question})

                    st.markdown("### 📝 答案")
                    st.write(result["answer"])

                    if result.get("context"):
                        st.markdown("### 📑 来源出处")
                        for i, doc in enumerate(result["context"]):
                            source = doc.metadata.get("source", "未知文档")
                            with st.expander(f"来源 {i + 1}: {source}"):
                                st.caption(doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))
                    else:
                        st.info("未检索到相关内容，答案基于模型通用知识生成")

                except Exception as e:
                    st.error(f"查询失败：{str(e)}")
                    st.info("请检查【管理后台】→【系统设置】中的模型配置和网络代理")

    st.caption("💡 提示：上传越多文档，回答越精准！点击侧边栏进入管理后台上传文档或切换模型")

# ==================== 页面 2: 管理后台 ====================
else:  # page == "⚙️ 管理后台"
    st.title("⚙️ 管理后台")

    manage_tabs = st.tabs(["📚 知识库管理", "🔧 系统设置"])

    # ---------- 子页面：知识库管理 ----------
    with manage_tabs[0]:
        st.header("📚 知识库文档管理")

        # 上传部分（保持不变）
        uploaded_files = st.file_uploader(
            "上传文档（支持 PDF、TXT、MD、DOCX，可多选）",
            type=["pdf", "txt", "md", "docx", "doc"],
            accept_multiple_files=True,
            key="admin_uploader"
        )

        if uploaded_files:
            os.makedirs("data", exist_ok=True)
            progress = st.progress(0)
            for i, file in enumerate(uploaded_files):
                file_path = f"data/{file.name}"
                if os.path.exists(file_path):
                    st.warning(f"⚠️ {file.name} 已存在，将被覆盖")
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                with st.spinner(f"正在解析 {file.name} 的表格和图片..."):
                    add_document(file_path, original_name=file.name)
                st.success(f"✅ {file.name} 入库成功！")
                progress.progress((i + 1) / len(uploaded_files))
            st.rerun()

        st.divider()

        # ==================== 文档列表：搜索 + 分页 ====================
        st.subheader("已入库文档列表")
        st.info(
            "📌 说明：每个文档会被智能切分成多个“块”（chunk），块数越多表示文档越长。这样做是为了让系统更精准地找到相关内容并生成更好答案。一个文档显示多块是正常现象，不是重复存储。")

        all_docs = list_documents()

        if not all_docs:
            st.info("📭 暂无文档，请上传后即可查看")
            st.stop()

        # 搜索框
        search_term = st.text_input(
            "🔍 搜索文档（支持文件名模糊匹配）",
            placeholder="输入文件名关键词...",
            key="doc_search"
        )

        # 过滤文档
        if search_term.strip():
            filtered_docs = [
                doc for doc in all_docs
                if search_term.lower() in doc['filename'].lower()
            ]
            st.caption(f"找到 **{len(filtered_docs)}** 个匹配结果（共 {len(all_docs)} 个文档）")
        else:
            filtered_docs = all_docs
            st.caption(f"共 **{len(filtered_docs)}** 个文档")

        if not filtered_docs:
            st.info("未找到匹配的文档")
            st.stop()

        # 分页设置
        page_size = 10
        total_pages = max(1, (len(filtered_docs) + page_size - 1) // page_size)

        # 当前页（从 session_state 读取，避免输入搜索时跳回第一页）
        if "doc_page" not in st.session_state:
            st.session_state.doc_page = 1

        current_page = st.session_state.doc_page

        # 页码控制
        col_page1, col_page2, col_page3, col_page4 = st.columns([1, 1, 2, 2])
        with col_page1:
            if st.button("« 上一页", disabled=(current_page <= 1)):
                st.session_state.doc_page = max(1, current_page - 1)
                st.rerun()
        with col_page2:
            if st.button("下一页 »", disabled=(current_page >= total_pages)):
                st.session_state.doc_page = min(total_pages, current_page + 1)
                st.rerun()
        with col_page3:
            st.write(f"第 {current_page} / {total_pages} 页")
        with col_page4:
            jump_page = st.number_input(
                "跳转到",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                step=1,
                key="jump_page"
            )
            if jump_page != current_page:
                st.session_state.doc_page = jump_page
                st.rerun()

        # 计算当前页数据
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_docs = filtered_docs[start_idx:end_idx]

        # 显示当前页文档
        for doc in page_docs:
            with st.expander(f"📄 **{doc['filename']}** · {doc['chunks']} 块 · ID: `{doc['doc_id'][:8]}`"):
                col1, col2 = st.columns([6, 1])
                with col1:
                    preview = get_document_content(doc['doc_id'])
                    st.code(preview, language="text")
                with col2:
                    if st.button("🗑️ 删除", key=f"del_{doc['doc_id']}"):
                        if delete_document(doc['doc_id']):
                            st.success(f"已删除 {doc['filename']}")
                            # 删除后重新计算分页
                            st.session_state.doc_page = 1  # 重置到第一页
                            st.rerun()
                        else:
                            st.error("删除失败")

        # 底部提示
        st.caption(f"显示第 {start_idx + 1} - {min(end_idx, len(filtered_docs))} 条，共 {len(filtered_docs)} 条")

    # ---------- 子页面：系统设置 ----------
    with manage_tabs[1]:
        st.header("系统设置")

        # 模型配置（同之前）
        st.subheader("大模型配置")
        provider_options = {p["name"]: key for key, p in config["providers"].items()}
        selected_name = st.selectbox(
            "选择模型",
            options=list(provider_options.keys()),
            index=list(provider_options.values()).index(current_provider_key)
        )
        selected_key = provider_options[selected_name]
        if selected_key != current_provider_key:
            config["model_provider"] = selected_key
            st.rerun()

        p = config["providers"][selected_key]
        p["api_key"] = st.text_input("API Key", value=p["api_key"], type="password")
        p["base_url"] = st.text_input("Base URL", value=p["base_url"])
        p["model_name"] = st.text_input("模型名称", value=p["model_name"])

        # 代理设置（同之前，已包含“主机+端口必填才可测试”）
        st.subheader("网络代理（可选）")
        proxy = config["proxy"]
        proxy["enabled"] = st.checkbox("启用代理", value=proxy["enabled"])

        proxy_test_disabled = True
        if proxy["enabled"]:
            col1, col2 = st.columns(2)
            proxy["protocol"] = col1.selectbox("协议", ["http", "https", "socks5"],
                                               index=["http", "https", "socks5"].index(proxy["protocol"]))
            proxy["host"] = col2.text_input("主机/IP", value=proxy["host"])

            col3, col4 = st.columns(2)
            proxy["port"] = col3.text_input("端口", value=proxy["port"])
            proxy["username"] = col4.text_input("用户名（可选）", value=proxy["username"])
            proxy["password"] = st.text_input("密码（可选）", value=proxy["password"], type="password")

            if proxy["host"].strip() and proxy["port"].strip():
                proxy_test_disabled = False

            show_detail = st.checkbox("🛠️ 显示详细错误堆栈（调试用）", value=False)

            if st.button("🔗 测试连接", disabled=proxy_test_disabled):
                from config_manager import test_proxy_connection

                with st.spinner("测试中..."):
                    success, msg = test_proxy_connection(config, show_traceback=show_detail)
                    if success:
                        st.success(f"✅ {msg}（代理和大模型 API 均正常）")
                    else:
                        st.error(f"❌ {msg}")
                        st.info("常见解决办法：检查代理是否开启、API Key 是否正确、是否需要科学上网")

        if st.button("💾 保存所有设置（永久生效）", type="primary", use_container_width=True):
            save_config(config)
            st.success("配置已永久保存！")
            st.balloons()
