import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import certifi
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import Query
from pymongo import DESCENDING

# ===== إعداد Logging =====
# logging.basicConfig(
#     level=logging.DEBUG,  # DEBUG = كل التفاصيل
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )

logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_dotenv()

# ===== إعداد FastAPI =====
app = FastAPI(title="🚀 NASA Papers Q&A API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # ممكن تحدد ["http://localhost:5500"] بدل * لو بتفتح من لايف سيرفر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MongoDB =====
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")
# logger.info(f"Connecting to MongoDB: {MONGO_URI}")
client = MongoClient(
    MONGO_URI, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=60000
)
db = client["nasa_papers"]
collection = db["articles"]

# ===== Pinecone =====
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY environment variable is not set")
INDEX_NAME = "nasa-articles-chunks"

# logger.info("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
# logger.info(f"Existing Pinecone indexes: {existing_indexes}")

if INDEX_NAME not in existing_indexes:
    logger.warning(f"Index {INDEX_NAME} not found. Creating it...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# ===== Embedding model =====
# logger.info("Loading embedding model all-MiniLM-L6-v2...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== Gemini =====
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY environment variable is not set")
genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# ===== Models =====
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


# ===== Routes =====
@app.get("/")
def home():
    return {"message": "🚀 NASA Papers Q&A API running"}


@app.post("/ask")
def ask_question(req: SearchRequest):
    query = req.query
    # logger.debug(f"Received query: {query}")

    # ===== Embedding =====
    vector = embedding_model.encode(query).tolist()
    # logger.debug(f"Generated embedding length: {len(vector)}")

    # ===== Pinecone search =====
    results = index.query(vector=vector, top_k=req.top_k, include_metadata=True)
    # logger.debug(f"Pinecone results: {results}")

    docs = []
    sources = []

    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        pmc_id = meta.get("pmc_id")
        title = meta.get("title")
        link = meta.get("link")

        # logger.debug(
        #     f"Found match: pmc_id={pmc_id}, title={title}, score={match['score']}"
        # )

        doc = collection.find_one({"pmc_id": pmc_id})
        if doc:
            # logger.debug(f"Document found in Mongo for {pmc_id}")
            docs.append(
                doc.get("abstract", "")[:1500]
            )  # خذ abstract أو content لو عندك
        else:
            logger.warning(f"No Mongo document found for pmc_id={pmc_id}")

        sources.append(
            {
                "pmc_id": pmc_id,
                "title": title,
                "link": link,
                "section": meta.get("section"),
            }
        )

    if not docs:
        logger.error("No documents found after matching")
        raise HTTPException(status_code=404, detail="No documents found")

    # ===== Context =====
    context_text = "\n\n".join(docs)
    # logger.debug(f"Context text length: {len(context_text)}")

    # ===== Send to Gemini =====
    prompt = f"""Answer the following question using the context below. 
If you don’t know, just say you don’t know.

Context:
{context_text}

Question: {query}
Answer:"""

    response = gemini_model.generate_content(prompt)
    answer_text = response.text if hasattr(response, "text") else str(response)
    # logger.debug(f"Gemini response: {answer_text[:300]}...")  # عرض أول 300 حرف بس

    return {"query": query, "answer": answer_text, "sources": sources}


@app.get("/articles/{pmc_id}")
def get_article_by_id(pmc_id: str):
    # logger.debug(f"Fetching article with pmc_id={pmc_id}")

    # لو عايز تدور بالـ pmc_id (string field عندك في الكولكشن)
    doc = collection.find_one({"pmc_id": pmc_id})

    # أو لو عايز تدور بالـ _id بتاع مونجو نفسه
    # try:
    #     obj_id = ObjectId(pmc_id)
    #     doc = collection.find_one({"_id": obj_id})
    # except Exception:
    #     raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    if not doc:
        logger.warning(f"No article found for pmc_id={pmc_id}")
        raise HTTPException(status_code=403, detail="Article not found")

    # تحويل ObjectId عشان يتسيريالاين كويس
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/articles")
def get_all_articles(page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100)):
    # logger.debug(f"Fetching articles page={page}, limit={limit}")

    skip = (page - 1) * limit

    cursor = (
        collection.find(
            {},
            {"pmc_id": 1, "summary": 1, "title": 1, "published_date": 1, "authors": 1},
        )
        .sort("published_date", DESCENDING)
        .skip(skip)
        .limit(limit)
    )

    articles = []
    for doc in cursor:
        articles.append(
            {
                "pmc_id": str(doc["pmc_id"]),
                "summary": doc.get("summary", ""),
                "title": doc.get("title", ""),
                "published_date": doc.get("published_date", ""),
                "publisher": doc.get("authors", [None])[0],  # أول author
            }
        )

    total_count = collection.count_documents({})
    total_pages = (total_count + limit - 1) // limit

    return {
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "total_count": total_count,
        "articles": articles,
    }


def build_article_mindmap(doc):
    title = doc.get("title", "Untitled")
    publisher = doc.get("authors", ["Unknown"])[0]

    mindmap = f"""mindmap
  root(("{title}"))
    Publisher "{publisher}"
    Introduction "{doc.get("introduction", "")[:80]}..."
    Methods "{doc.get("methods", "")[:80]}..."
    Results "{doc.get("results", "")[:80]}..."
    Discussion "{doc.get("discussion", "")[:80]}..."
    Conclusion "{doc.get("conclusion", "")[:80]}..."
    Significant "{doc.get("significant", "")[:80]}..."
    Summary "{doc.get("summary", "")[:80]}..."
    Figures
"""
    for i, fig in enumerate(doc.get("figures", []), 1):
        mindmap += f'      "Figure {i}: {fig.get("text","")[:60]}..."\n'

    mindmap += "    Tables\n"
    for i, tbl in enumerate(doc.get("tables", []), 1):
        mindmap += f'      "Table {i}: {tbl.get("text","")[:60]}..."\n'

    return mindmap


@app.get("/articles/{pmc_id}/mindmap")
def get_article_mindmap(pmc_id: str):
    doc = collection.find_one({"pmc_id": pmc_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"mindmap": build_article_mindmap(doc)}
