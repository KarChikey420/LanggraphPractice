import os
from typing import List, Optional

import typer
from dotenv import load_dotenv
from phi.agent import Agent
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.model.groq import Groq
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.pgvector import PgVector, SearchType

try:
    from phi.embedder.fastembed import FastEmbedEmbedder
except ImportError as exc:
    raise RuntimeError(
        "Install the `fastembed` package in this project environment before running this script."
    ) from exc

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GroqAPI")
if not groq_api_key:
    raise RuntimeError("Set GROQ_API_KEY or GroqAPI in your .env file before running this script.")

os.environ["GROQ_API_KEY"] = groq_api_key

db_url = os.getenv("DATABASE_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")
embedding_model = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=FastEmbedEmbedder(model=embedding_model),
    ),
)

# Load the PDF into pgvector. After the first successful run, you can comment this out.
knowledge_base.load(upsert=True)

storage = PgAgentStorage(table_name="pdf_agent", db_url=db_url)


def pdf_agent(new: bool = False, user: str = "user") -> None:
    session_id: Optional[str] = None

    if not new:
        existing_sessions: List[str] = storage.get_all_session_ids(user)
        if existing_sessions:
            session_id = existing_sessions[0]

    agent = Agent(
        session_id=session_id,
        user_id=user,
        model=Groq(id="llama-3.3-70b-versatile"),
        knowledge=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if session_id is None:
        print(f"Started session: {agent.session_id}")
    else:
        print(f"Continuing session: {session_id}")

    agent.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(pdf_agent)
            
