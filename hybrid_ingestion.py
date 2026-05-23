import os
import re
import kuzu

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader
)

from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\[\d+\]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =========================================================
# ENTITY + RELATION EXTRACTION
# =========================================================

def extract_entities_and_relations(text):

    entities = []

    relations = []

    # =========================================
    # COMPANY NAMES
    # =========================================

    companies = [
        "Nvidia",
        "Google",
        "Tesla",
        "SpaceX",
        "Microsoft"
    ]

    detected_company = None

    # =========================================
    # DETECT COMPANY
    # =========================================

    for company in companies:

        if company.lower() in text.lower():

            entities.append((company, "Company"))

            detected_company = company

    # =========================================
    # NVIDIA
    # =========================================

    if (
        "Jensen Huang" in text
        and detected_company == "Nvidia"
    ):

        entities.append(("Jensen Huang", "Person"))

        relations.append(
            ("Jensen Huang", "founder_of", "Nvidia")
        )

        relations.append(
            ("Jensen Huang", "CEO_of", "Nvidia")
        )

    # =========================================
    # GOOGLE
    # =========================================

    if (
        "Sundar Pichai" in text
        and detected_company == "Google"
    ):

        entities.append(("Sundar Pichai", "Person"))

        relations.append(
            ("Sundar Pichai", "CEO_of", "Google")
        )

    if (
        "Larry Page" in text
        and detected_company == "Google"
    ):

        entities.append(("Larry Page", "Person"))

        relations.append(
            ("Larry Page", "founder_of", "Google")
        )

    if (
        "Sergey Brin" in text
        and detected_company == "Google"
    ):

        entities.append(("Sergey Brin", "Person"))

        relations.append(
            ("Sergey Brin", "founder_of", "Google")
        )

    # =========================================
    # MICROSOFT
    # =========================================

    if (
        "Bill Gates" in text
        and detected_company == "Microsoft"
    ):

        entities.append(("Bill Gates", "Person"))

        relations.append(
            ("Bill Gates", "founder_of", "Microsoft")
        )

    # =========================================
    # SPACEX
    # =========================================

    if (
        "Elon Musk" in text
        and detected_company == "SpaceX"
    ):

        entities.append(("Elon Musk", "Person"))

        relations.append(
            ("Elon Musk", "founder_of", "SpaceX")
        )

    # =========================================
    # TESLA
    # =========================================

    if (
        "Elon Musk" in text
        and detected_company == "Tesla"
    ):

        entities.append(("Elon Musk", "Person"))

        relations.append(
            ("Elon Musk", "CEO_of", "Tesla")
        )

    # =========================================
    # HEADQUARTERS EXTRACTION
    # =========================================

    hq_match = re.search(

        r"headquartered in ([A-Z][a-zA-Z\s,]+)",

        text
    )

    if hq_match and detected_company:

        location = hq_match.group(1).strip()

        entities.append((location, "Location"))

        relations.append(
            (
                detected_company,
                "headquartered_in",
                location
            )
        )

    return entities, relations

# =========================================================
# LOAD DOCUMENTS
# =========================================================

def load_documents(docs_path="docs"):

    loader = DirectoryLoader(

        path=docs_path,

        glob="*.txt",

        loader_cls=lambda path:
        TextLoader(path, encoding="utf-8")
    )

    documents = loader.load()

    for doc in documents:

        doc.page_content = clean_text(
            doc.page_content
        )

    return documents

# =========================================================
# SEMANTIC CHUNKING
# =========================================================

def split_documents(documents):

    print("\n🧠 Using Semantic Chunking...\n")

    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    text_splitter = SemanticChunker(

        embeddings=embedding_model,

        breakpoint_threshold_type=
        "standard_deviation",

        breakpoint_threshold_amount=1
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(f"✅ Created {len(chunks)} semantic chunks")

    return chunks

# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(

        model_name=
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        persist_directory="db/chroma_db"
    )

    return vectorstore

# =========================================================
# STORE GRAPH
# =========================================================

def store_graph(chunks):

    db = kuzu.Database("db/kuzu_graph")

    conn = kuzu.Connection(db)

    for chunk in chunks:

        text = chunk.page_content

        # =====================================
        # EXTRACT ENTITIES + RELATIONS
        # =====================================

        entities, relations = (
            extract_entities_and_relations(text)
        )

        # =====================================
        # STORE ENTITIES
        # =====================================

        for entity_name, entity_type in entities:

            conn.execute(f"""

            MERGE (e:Entity {{
                name:'{entity_name}',
                type:'{entity_type}'
            }})

            """)

        # =====================================
        # STORE RELATIONS
        # =====================================

        for source, relation, target in relations:

            conn.execute(f"""

            MATCH (a:Entity), (b:Entity)

            WHERE a.name='{source}'
            AND b.name='{target}'

            MERGE (a)-[:RELATED {{
                relation:'{relation}'
            }}]->(b)

            """)

    print("✅ Graph relationships stored")

# =========================================================
# MAIN
# =========================================================

def main():

    print("🚀 HYBRID GRAPH RAG INGESTION")

    documents = load_documents()

    print(f"✅ Loaded {len(documents)} docs")

    chunks = split_documents(documents)

    create_vector_store(chunks)

    print("✅ Stored embeddings in ChromaDB")

    store_graph(chunks)

    print("✅ Stored graph in KuzuDB")

if __name__ == "__main__":

    main()