import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
import requests
#streamlit run example_strlit.py

llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model="openai/gpt-oss-120b",
        streaming=False,
        )

# Math tools
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b

# Web search tool
@tool
def web_search(query: str) -> str:
        """Search the web using Tinyfish. Use this to get current information accurately."""
        url = "https://api.search.tinyfish.ai"
        headers = {"X-API-Key": st.secrets["TINYFISH_API_KEY"]}
        params = {"query": query}
        
        response = requests.get(url, headers=headers, params=params)
        results = response.json()
        
        output = ""
        for r in results.get("results", []):
            output += f"Title: {r.get('title', '')}\nSummary: {r.get('snippet', '')}\nURL: {r.get('url', '')}\n\n"
        
        return output if output else "No results found."
# All tools together
tools = [add, multiply, subtract, web_search]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed. Use the web_search tool only when you need current or external information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)


def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def split_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def store_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        cache_folder=".cache/huggingface/hub"
    )
    vectorstore = Chroma.from_texts(
        texts = chunks,
        embedding=embeddings,
        persist_directory=None
        )
    return vectorstore



def ask(query,vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer questions using only the context provided."
            "List ALL points mentioned in the context without skipping or summarizing any of them. "
            "Use markdown formatting — bullet points for lists, code blocks for code. "
            "Do not list everything in the context — only what is relevant to the question asked."
            "Do not add any closing or summary sentences at the end of your response."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}"
                )
                ])
    chain = prompt | llm | StrOutputParser()
    docs = retriever.invoke(query)
    
    results = vectorstore.similarity_search_with_score(query, k=5)
    best_score = results[0][1]
    
    scores = [score for doc, score in results]
    print(f"All scores: {scores}")
    print(f"Best score: {results[0][1]}")
    
    if best_score < 1.2: # Threshold for relevance
        context = "\n\n".join([doc.page_content for doc, score in results])
        return chain.invoke({"context": context, "question": query})  
    else:
        return agent_executor.invoke({"input": query})["output"]

st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing your PDF..."):
        text = extract_text(uploaded_file)
        chunks = split_chunks(text)
        collection = store_chunks(chunks)
    st.success(f"Ready! Indexed {len(chunks)} chunks.")

    query = st.text_input("Ask a question about your PDF")

    if query:
        with st.spinner("Thinking..."):
            answer = ask(query, collection)
        st.write(answer)
        
# learn about "with" function
# caching in streamlit to avoid loading every time