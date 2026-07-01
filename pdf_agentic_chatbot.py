import streamlit as st
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
import requests

def extract_text(uploaded_file):
    data = uploaded_file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


def store_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="pdf_Agentic_db",
    )
    return vectorstore

def build_agent(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF for relevant information."""
        docs = retriever.invoke(query)
        if not docs:
            return "NO_CONTEXT_FOUND"
        return "\n\n".join([doc.page_content for doc in docs])
    @tool
    def web_search(query: str) -> str:
        """Search the web using Tinyfish. Use this only if search_pdf returns NO_CONTEXT_FOUND."""
        url = "https://api.search.tinyfish.ai"  # or tavily
        headers = {"X-API-Key": st.secrets["TINYFISH_API_KEY"]}
        params = {"query": query}
        
        response = requests.get(url, headers=headers, params=params)
        results = response.json() #creates a dictionary from the json response
        
        output = ""
        for r in results.get("results", []):
            output += f"Title: {r.get('title', '')}\nSummary: {r.get('snippet', '')}\nURL: {r.get('url', '')}\n\n"
        
        return output if output else "No results found."
    tools = [search_pdf, web_search]
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant. ALways use search_pdf first"
         "If it returns NO_CONTEXT_FOUND then use web_search"
         "Decorate your answer with bullet points and give detailed information"
         "Current year is 2026"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agents=create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agents,
        tools=tools,
        verbose=True #shows the reasoning steps of the agent
    )
    

# UI
st.title("Smart PDF & Web Search Agent")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing your PDF..."):
        text = extract_text(uploaded_file)
        chunks = split_chunks(text)
        vectorstore = store_chunks(chunks)
        st.session_state["agent"] = build_agent(vectorstore)
    st.success(f"Ready! Indexed {len(chunks)} chunks.")

question = st.text_input("Ask a question")

if question:
    if "agent" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
             result = st.session_state["agent"].invoke({"input": question})
             st.write(result["output"])
