import kuzu
import os

os.makedirs("db", exist_ok=True)

db = kuzu.Database("db/kuzu_graph")
conn = kuzu.Connection(db)

try:
    conn.execute("""
    CREATE NODE TABLE Entity(
        name STRING,
        type STRING,
        PRIMARY KEY(name)
    )
    """)
except:
    pass

try:
    conn.execute("""
    CREATE REL TABLE RELATED(
        FROM Entity TO Entity,
        relation STRING
    )
    """)
except:
    pass

print("✅ Hybrid Graph Schema Ready 🚀")