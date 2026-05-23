# =========================================================
# IMPORTS
# =========================================================

# kuzu -> graph database

import kuzu

# =========================================================
# VECTOR DATABASE
# =========================================================

# Chroma is used as vector database

from langchain_chroma import Chroma

# =========================================================
# EMBEDDING MODEL
# =========================================================

# HuggingFaceEmbeddings converts text into vectors

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

# =========================================================
# OLLAMA MODEL
# =========================================================

# ChatOllama is used to connect local LLM

from langchain_ollama import ChatOllama

# =========================================================
# MESSAGE TYPES
# =========================================================

# HumanMessage -> user prompt
# SystemMessage -> system instructions

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

# This embedding model converts:
#
# text -> vector embeddings
#
# Same model must be used during:
# 1. ingestion
# 2. retrieval
#
# Otherwise vector similarity will fail.

embedding_model = HuggingFaceEmbeddings(

    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# LOAD VECTOR DATABASE
# =========================================================

# Load existing ChromaDB
#
# persist_directory:
# folder where embeddings are stored

vector_db = Chroma(

    persist_directory="db/chroma_db",

    embedding_function=embedding_model
)

# =========================================================
# LOAD GRAPH DATABASE
# =========================================================

# Load Kuzu Graph DB

graph_db = kuzu.Database("db/kuzu_graph")

# Create graph connection

conn = kuzu.Connection(graph_db)

# =========================================================
# LOAD LLM MODEL
# =========================================================

# phi3:mini is used because:
#
# ✅ faster
# ✅ better grounding
# ✅ less hallucination
#
# compared to tinyllama

model = ChatOllama(model="phi3:mini")

# =========================================================
# GRAPH RETRIEVAL FUNCTION
# =========================================================

# This function retrieves:
# graph relationships
#
# Example:
#
# Sundar Pichai -> CEO_of -> Google

def get_graph_context(query):

    # Stores graph results
    graph_context = ""

    # =====================================================
    # CYPHER QUERY
    # =====================================================

    # Fetch all graph relationships

    result = conn.execute("""

    MATCH (a:Entity)-[r:RELATED]->(b:Entity)

    RETURN a.name, r.relation, b.name

    """)

    # =====================================================
    # PROCESS GRAPH RESULTS
    # =====================================================

    while result.has_next():

        # Get one row
        row = result.get_next()

        # Convert graph relation into sentence
        #
        # Example:
        #
        # Sundar Pichai CEO_of Google

        sentence = (
            f"{row[0]} "
            f"{row[1]} "
            f"{row[2]}"
        )

        # =================================================
        # SIMPLE RELEVANCE FILTERING
        # =================================================

        # Convert query to lowercase
        query_lower = query.lower()

        # If any query word exists in sentence
        # keep graph sentence

        if any(word in sentence.lower() for word in query_lower.split()):

            graph_context += sentence + ". "

    return graph_context

# =========================================================
# CHAT LOOP
# =========================================================

# Infinite loop keeps chatbot running

print("\n🚀 HYBRID GRAPH RAG READY")

print("Type 'exit' to quit.\n")

while True:

    # =====================================================
    # USER INPUT
    # =====================================================

    query = input("🔍 Ask Question: ")

    # =====================================================
    # EXIT CONDITION
    # =====================================================

    if query.lower() == "exit":

        print("\n👋 Exiting Hybrid Graph RAG...")

        break

    # =====================================================
    # VECTOR RETRIEVAL
    # =====================================================

    # similarity_search_with_score:
    #
    # Retrieves:
    # 1. similar chunks
    # 2. similarity score
    #
    # k=3 means:
    # retrieve top 3 chunks

    results = vector_db.similarity_search_with_score(
        query,
        k=3
    )

    # Stores filtered chunks
    relevant_docs = []

    # =====================================================
    # FILTER RETRIEVED DOCUMENTS
    # =====================================================

    for doc, score in results:

        # Lower score = better similarity
        #
        # score < 1.8 means:
        # only reasonably relevant chunks allowed

        if score < 1.8:

            relevant_docs.append(doc)

    # =====================================================
    # VECTOR CONTEXT
    # =====================================================

    # Combine retrieved chunks into one text

    vector_context = "\n".join([

        doc.page_content

        for doc in relevant_docs
    ])

    # =====================================================
    # GRAPH CONTEXT
    # =====================================================

    # Retrieve graph relationships

    graph_context = get_graph_context(query)

    # =====================================================
    # COMBINED CONTEXT
    # =====================================================

    # Combine:
    #
    # 1. Vector context
    # 2. Graph context
    #
    # This is HYBRID RETRIEVAL

    combined_context = f"""

    {vector_context}

    {graph_context}

    """

    # =====================================================
    # LIMIT CONTEXT SIZE
    # =====================================================

    # Reduce prompt size
    #
    # Helps:
    # ✅ faster generation
    # ✅ less hallucination
    # ✅ smaller context window

    combined_context = combined_context[:2000]

    # =====================================================
    # NO CONTEXT FOUND
    # =====================================================

    # If no relevant information retrieved

    if not combined_context.strip():

        print("\n🤖 Answer:\n")

        print("I don't have enough information.")

        print("\n" + "=" * 60 + "\n")

        continue

    # =====================================================
    # FINAL PROMPT
    # =====================================================

    # Prompt engineering
    #
    # Restricts hallucination

    final_prompt = f"""

Answer ONLY from the provided context.

Question:
{query}

Context:
{combined_context}

Rules:
- Give short precise answers.
- Do NOT hallucinate.
- Do NOT guess.
- Do NOT generate long summaries.
- If answer is not present say:
  "I don't have enough information."

"""

    # =====================================================
    # CREATE MESSAGES
    # =====================================================

    messages = [

        # System instructions
        SystemMessage(
            content="You are a helpful assistant."
        ),

        # User query + context
        HumanMessage(
            content=final_prompt
        )
    ]

    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    # Send prompt to LLM

    result = model.invoke(messages)

    # =====================================================
    # PRINT ANSWER
    # =====================================================

    print("\n🤖 Answer:\n")

    print(result.content)

    print("\n" + "=" * 60 + "\n")