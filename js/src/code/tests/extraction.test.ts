import { describe, test, expect } from "vitest";
import { _extractCodeFromMarkdownCodeBlocks } from "../base";

describe("code extraction", () => {
  test("extraction test 1", () => {
    const CODE = `
I'll provide a comprehensive guide to using Ollama for building a Retrieval-Augmented Generation (RAG) agent with LangChain. Here's a step-by-step approach:

## Setup and Installation

1. Install Ollama
• Download and install Ollama from the [official website](https://ollama.ai/download) [1]
• Pull a suitable model, such as Llama3.1: 
\`\`\`bash
ollama pull llama3.1
\`\`\`

2. Install Required Packages
\`\`\`bash
pip install langchain-ollama
pip install langchain
\`\`\`

## Components for RAG

### 1. Embeddings
Use Ollama embeddings to create vector representations:
\`\`\`python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")
\`\`\`

### 2. Vectorstore
Create a vectorstore to index your documents:
\`\`\`python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()
\`\`\`

### 3. Chat Model
Use ChatOllama for generation:
\`\`\`python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1", 
    temperature=0
)
\`\`\`

### 4. RAG Chain
Construct a basic RAG pipeline:
\`\`\`python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Create a prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm
)

# Invoke the chain
response = rag_chain.invoke("Your question here")
\`\`\`

## Key Considerations

• Model Selection: Ollama supports multiple open-source models [1]
• Local Inference: All processing happens on your machine
• Tool Calling: Ollama supports OpenAI-compatible tool calling [1]

## Advanced Features

### Tool Calling
\`\`\`python
from langchain_core.tools import tool

@tool
def search_tool(query: str):
    """Search and retrieve information"""
    # Implement search logic

# Bind tools to the model
llm_with_tools = llm.bind_tools([search_tool])
\`\`\``;

    const expectedOutput = `from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")

from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1", 
    temperature=0
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Create a prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm
)

# Invoke the chain
response = rag_chain.invoke("Your question here")

from langchain_core.tools import tool

@tool
def search_tool(query: str):
    """Search and retrieve information"""
    # Implement search logic

# Bind tools to the model
llm_with_tools = llm.bind_tools([search_tool])
`;

    const extractedCode = _extractCodeFromMarkdownCodeBlocks(CODE);
    expect(extractedCode).toBe(expectedOutput);
  });
});
