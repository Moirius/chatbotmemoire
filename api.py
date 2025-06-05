# api.py — version épurée pour mémoire universitaire

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain

# 🌍 Chargement des variables d'environnement
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Clé API OpenAI manquante. Vérifie ton fichier .env.")

# 🚀 Initialisation de l'app FastAPI
app = FastAPI()

# 🔓 CORS pour autoriser ton frontend (ex : Render)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← autorise tout (OK pour test local)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 📦 Chargement de l'index FAISS
print("🔄 Chargement de l'index FAISS...")
vectorstore = FAISS.load_local(
    "faiss_index",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
print("✅ Index chargé.")

# 🧠 Prompt pour ton mémoire
prompt_template = PromptTemplate(
    input_variables=["context", "input"],
    template="""
Tu es un assistant intelligent, spécialisé dans l’art génératif et la performance audiovisuelle.

Tu aides à comprendre un mémoire universitaire intitulé :
**"TouchDesigner : le pont entre art génératif et performance live"**.

Utilise uniquement les informations suivantes :
{context}

Question : {input}

Réponse (en **Markdown** avec des éléments mis en valeur) :
"""
)

llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.3)
qa_chain = create_retrieval_chain(retriever, prompt_template | llm)

# 📥 Schéma de requête utilisateur
class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(question: Question):
    print(f"🧠 Question reçue : {question.query}")
    response = qa_chain.invoke({"input": question.query})
    print("🧾 Réponse :", response)

    return {
        "answer": response["answer"].content if hasattr(response["answer"], "content") else str(response.get("answer", "❌ Réponse vide")),
        "sources": [
            doc.metadata.get("source", "inconnu")
            for doc in response.get("source_documents", [])
        ]
    }



# 🌐 Interface HTML statique (si utilisée)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return {"status": "ok"}

# Pour exécution locale
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
