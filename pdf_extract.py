import typer
from typing import Optional,List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlknowledgeBase
from phi.vectordb.pgvector import PgVectorDB
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["GroqAPIkey"]= os.getenv("GroqAPI")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFUrlknowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVectorDB(collection="recipes", db_url=db_url)
)

knowledge_base=knowledge_base.load()
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

def pdf_assistant(new:bool=False,user:str="user"):
    run_id = Optional[str]= None
    
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user=user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]
            
        assistant = Assistant(
            run_id=run_id,
            user_id=user,
            storage=storage,
            knowledge_bases=[knowledge_base],
            show_tool_calls=True,
            search_knowledge=True,
            read_chat_history=True
        )
        if run_id is None:
            run_id = assistant.run_id
            print(f"Starting a new assistant with run_id: {run_id}")
        else:
            print(f"Resuming assistant with run_id: {run_id}")
        
        assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)
            