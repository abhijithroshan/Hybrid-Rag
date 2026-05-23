import kuzu

from langchain_chroma import Chroma

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_ollama import ChatOllama

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(

    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# LOAD CHROMADB
# =========================================================

vector_db = Chroma(

    persist_directory="db/chroma_db",

    embedding_function=embedding_model
)

# =========================================================
# LOAD KUZU GRAPH DB
# =========================================================

graph_db = kuzu.Database("db/kuzu_graph")

conn = kuzu.Connection(graph_db)

# =========================================================
# LOAD OLLAMA MODEL
# =========================================================

# Better than tinyllama for grounding
model = ChatOllama(model="phi3:mini")

# =========================================================
# GRAPH RETRIEVAL
# =========================================================

def get_graph_context(query):

    graph_context = ""

    result = conn.execute("""

    MATCH (a:Entity)-[r:RELATED]->(b:Entity)

    RETURN a.name, r.relation, b.name

    """)

    while result.has_next():

        row = result.get_next()

        sentence = (
            f"{row[0]} "
            f"{row[1]} "
            f"{row[2]}"
        )

        # =========================================
        # SIMPLE RELEVANCE FILTERING
        # =========================================

        query_lower = query.lower()

        if any(word in sentence.lower() for word in query_lower.split()):

            graph_context += sentence + ". "

    return graph_context

# =========================================================
# CHAT LOOP
# =========================================================

print("\n🚀 HYBRID GRAPH RAG READY")
print("Type 'exit' to quit.\n")

while True:

    # =====================================================
    # USER INPUT
    # =====================================================

    query = input("🔍 Ask Question: ")

    if query.lower() == "exit":

        print("\n👋 Exiting Hybrid Graph RAG...")
        break

    # =====================================================
    # VECTOR RETRIEVAL
    # =====================================================

    results = vector_db.similarity_search_with_score(
        query,
        k=3
    )

    relevant_docs = []

    # =====================================================
    # FILTER RETRIEVED DOCUMENTS
    # =====================================================

    for doc, score in results:

        # Lower score = better similarity

        if score < 1.8:

            relevant_docs.append(doc)

    # =====================================================
    # VECTOR CONTEXT
    # =====================================================

    vector_context = "\n".join([

        doc.page_content

        for doc in relevant_docs
    ])

    # =====================================================
    # GRAPH CONTEXT
    # =====================================================

    graph_context = get_graph_context(query)

    # =====================================================
    # COMBINED CONTEXT
    # =====================================================

    combined_context = f"""

    {vector_context}

    {graph_context}

    """

    # =====================================================
    # LIMIT CONTEXT SIZE
    # =====================================================

    combined_context = combined_context[:2000]

    # =====================================================
    # NO CONTEXT FOUND
    # =====================================================

    if not combined_context.strip():

        print("\n🤖 Answer:\n")

        print("I don't have enough information.")

        print("\n" + "=" * 60 + "\n")

        continue

    # =====================================================
    # FINAL PROMPT
    # =====================================================

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
    # MESSAGES
    # =====================================================

    messages = [

        SystemMessage(
            content="You are a helpful assistant."
        ),

        HumanMessage(
            content=final_prompt
        )
    ]

    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    result = model.invoke(messages)

    # =====================================================
    # PRINT ANSWER
    # =====================================================

    print("\n🤖 Answer:\n")

    print(result.content)

    print("\n" + "=" * 60 + "\n")