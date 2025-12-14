import os
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from embeddings import get_embedding_model
from config import CHROMA_DB_PATH, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, CHAT_MODEL

# 1. 加载向量库
embedding = get_embedding_model()
vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding)

# 2. DeepSeek 兼容 OpenAI 接口
llm = ChatOpenAI(
    model=CHAT_MODEL,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.3,
    max_tokens=2048,
)

# 3. Prompt 模板（中文优化版）
template = """你是一个乐于助人的AI助手，使用以下检索到的上下文来回答用户问题。
如果上下文里没有相关信息，就说“我不知道”，不要胡编乱造。

上下文：
{context}

问题：{question}
请用中文详细、准确地回答："""

prompt = PromptTemplate.from_template(template)

# 4. 构建 RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def ask(question: str):
    print(f"\n问题：{question}")
    result = qa_chain.invoke({"query": question})
    print(f"\n答案：{result['result']}\n")
    print("="*80)

def rag_query(question: str):
    result = qa_chain.invoke({"query": question})
    return result['result'], result['source_documents']

if __name__ == "__main__":
    print("RAG + DeepSeek 已就绪，输入 exit 退出")
    while True:
        q = input("\n你问：")
        if q.strip().lower() in ["exit", "quit"]:
            break
        if q.strip():
            ask(q)