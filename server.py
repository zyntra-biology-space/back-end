"""
🚀 NASA Research Papers API - Combined Version
Combines:
- RAG-based Q&A (semantic search + Gemini) from nasa_qa_api
- Rich Mindmap generation from improved_mindmap_api
- Article management and search
"""

import logging
import json
import certifi
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient, DESCENDING
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import google.generativeai as genai

# =====================
# Logging Setup
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# =====================
# FastAPI Setup
# =====================
app = FastAPI(
    title="🚀 NASA Research Papers API",
    description="Advanced Q&A + Mindmap generation for NASA research papers",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# MongoDB Setup
# =====================
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

logger.info("Connecting to MongoDB...")
client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=60000
)

db = client["nasa_papers"]
collection = db["articles"]
logger.info("✅ MongoDB connected")

# =====================
# Pinecone Setup (for RAG)
# =====================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY environment variable is not set")

INDEX_NAME = "nasa-articles-chunks"

logger.info("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    logger.warning(f"Index {INDEX_NAME} not found. Creating it...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    logger.info(f"✅ Created index {INDEX_NAME}")
else:
    logger.info(f"✅ Index {INDEX_NAME} already exists")

index = pc.Index(INDEX_NAME)

# =====================
# Embedding Model Setup
# =====================
embedding_model = None

@app.on_event("startup")
def load_model():
    global embedding_model
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model ready")

# =====================
# Gemini Setup
# =====================
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY environment variable is not set")

genai.configure(api_key=GENAI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
logger.info("✅ Gemini configured")

# =====================
# Pydantic Models
# =====================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# =====================
# Helper Functions
# =====================

def clean_for_mindmap(text: str, max_length: int = 80) -> str:
    """Clean text for Mermaid mindmap syntax"""
    if not text:
        return "No content"
    
    text = str(text)
    
    # Remove problematic characters
    text = text.replace('"', "'")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    
    # Remove special symbols
    for char in "()[]{}/<>|\\\"'":
        text = text.replace(char, "")
    
    # Clean multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    
    text = text.strip()
    
    # Smart truncation
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."
    
    return text if text else "No content"

# =====================
# Research Extraction (for Mindmap)
# =====================

def extract_research(doc: dict) -> dict | None:
    """Extract comprehensive research data for mindmap"""
    prompt = f"""
You are an expert scientific research analyst specialized in extracting structured research insights.

From the article below, extract COMPREHENSIVE research-level information that would be valuable for researchers.
Return STRICT JSON ONLY - NO MARKDOWN, NO BACKTICKS, JUST RAW JSON.

Format (ALL fields required, use empty arrays/strings if no data):
{{
  "research_question": "Main research question or objective (1-2 sentences max)",
  "methodology": ["Method 1", "Method 2", "Method 3"],
  "key_findings": ["Finding 1 with specific results", "Finding 2 with numbers/results"],
  "statistical_significance": "p-values, effect sizes, or confidence intervals if mentioned",
  "mechanisms": ["Mechanism or pathway 1", "Mechanism or pathway 2"],
  "implications": ["Practical implication 1", "Theoretical implication 2"],
  "limitations": ["Limitation 1", "Limitation 2"],
  "open_questions": ["Question 1 for future research", "Question 2"],
  "relevant_fields": ["Field 1", "Field 2", "Field 3"],
  "sample_details": "Sample size, population, or experimental setup"
}}

Article:
Title: {doc.get("title","")}
Abstract: {doc.get("abstract","")}
Keywords: {doc.get("keywords","")}
Introduction: {doc.get("introduction","")}
Methods: {doc.get("methods","")}
Results: {doc.get("results","")}
Discussion: {doc.get("discussion","")}
Conclusion: {doc.get("conclusion","")}
"""

    try:
        resp = gemini_model.generate_content(prompt)
        text = resp.text.strip()

        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.split("```")[1].strip()

        result = json.loads(text)
        return result

    except Exception as e:
        logger.warning("Research extraction failed: %s", e)
        return None

# =====================
# Mindmap Builders
# =====================

def build_research_mindmap(doc: dict) -> str:
    """Build detailed mindmap with research data"""
    data = extract_research(doc)

    if not data:
        # Fallback to simple mindmap
        title = clean_for_mindmap(doc.get('title', 'Untitled'), 150)
        return f"""mindmap
  root(({title}))
    Abstract
      {clean_for_mindmap(doc.get('abstract', ''), 200)}
    Conclusion
      {clean_for_mindmap(doc.get('conclusion', ''), 200)}
"""

    title = clean_for_mindmap(doc.get('title', 'Untitled'), 150)
    mm = f"""mindmap
  root(({title}))
"""

    # Research Question
    rq = clean_for_mindmap(data.get('research_question', ''), 250)
    if rq:
        mm += f"""    Research Question
      {rq}
"""

    # Methodology
    if data.get("methodology"):
        mm += "    Methodology\n"
        for i, method in enumerate(data["methodology"][:4], 1):
            method_clean = clean_for_mindmap(method, 200)
            if method_clean:
                mm += f"      M{i}. {method_clean}\n"

    # Key Findings
    if data.get("key_findings"):
        mm += "    Key Findings\n"
        for i, finding in enumerate(data["key_findings"][:5], 1):
            finding_clean = clean_for_mindmap(finding, 220)
            if finding_clean:
                mm += f"      F{i}. {finding_clean}\n"

    # Statistical Significance
    sig = clean_for_mindmap(data.get("statistical_significance", ""), 200)
    if sig:
        mm += f"""    Statistics
      {sig}
"""

    # Sample Details
    sample = clean_for_mindmap(data.get("sample_details", ""), 200)
    if sample:
        mm += f"""    Sample Details
      {sample}
"""

    # Mechanisms
    if data.get("mechanisms"):
        mm += "    Mechanisms\n"
        for i, mechanism in enumerate(data["mechanisms"][:4], 1):
            mech_clean = clean_for_mindmap(mechanism, 200)
            if mech_clean:
                mm += f"      M{i}. {mech_clean}\n"

    # Implications
    if data.get("implications"):
        mm += "    Implications\n"
        for i, implication in enumerate(data["implications"][:4], 1):
            impl_clean = clean_for_mindmap(implication, 220)
            if impl_clean:
                mm += f"      I{i}. {impl_clean}\n"

    # Limitations
    if data.get("limitations"):
        mm += "    Limitations\n"
        for i, limitation in enumerate(data["limitations"][:3], 1):
            lim_clean = clean_for_mindmap(limitation, 210)
            if lim_clean:
                mm += f"      L{i}. {lim_clean}\n"

    # Open Questions
    if data.get("open_questions"):
        mm += "    Open Questions\n"
        for i, question in enumerate(data["open_questions"][:3], 1):
            q_clean = clean_for_mindmap(question, 220)
            if q_clean:
                mm += f"      Q{i}. {q_clean}\n"

    # Relevant Fields
    if data.get("relevant_fields"):
        mm += "    Related Fields\n"
        for field in data["relevant_fields"][:4]:
            field_clean = clean_for_mindmap(field, 120)
            if field_clean:
                mm += f"      {field_clean}\n"

    return mm

# =====================
# Routes - Health & Info
# =====================

@app.get("/", tags=["Health"])
def home():
    """API Health Check"""
    return {
        "message": "🚀 NASA Research Papers API v3.0",
        "status": "running",
        "features": [
            "Ask global questions (RAG with embeddings)",
            "Search across all papers",
            "Generate research mindmaps",
            "Browse articles with pagination",
            "AI-powered insights"
        ]
    }

# =====================
# Routes - Q&A (RAG-based)
# =====================

@app.post("/ask", tags=["Q&A"])
def ask_question(req: SearchRequest):
    """
    🎯 Ask a global question across ALL articles
    
    Uses semantic search + Gemini for intelligent answers
    - query: Your research question
    - top_k: Number of relevant chunks (default: 5)
    """
    query = req.query
    logger.info(f"❓ Question: {query}")
    
    if not query or len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
    
    try:
        # Generate embedding
        vector = embedding_model.encode(query).tolist()
        
        # Search Pinecone
        results = index.query(vector=vector, top_k=req.top_k, include_metadata=True)
        
        docs = []
        sources = []
        
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            pmc_id = meta.get("pmc_id")
            title = meta.get("title")
            section = meta.get("section")
            score = match['score']
            
            doc = collection.find_one({"pmc_id": pmc_id})
            if doc:
                section_content = ""
                if section and section in doc:
                    section_content = str(doc.get(section, ""))[:1500]
                
                if not section_content:
                    section_content = doc.get("abstract", "")[:1500]
                
                if section_content:
                    docs.append(section_content)
            
            sources.append({
                "pmc_id": pmc_id,
                "title": title,
                "section": section,
                "relevance_score": round(score, 3)
            })
        
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Build context
        context_text = "\n\n".join(docs)
        
        # Generate answer with Gemini
        prompt = f"""You are a helpful scientific research assistant. Answer the following question 
using the provided research context. Be accurate and concise.

Context:
{context_text}

Question: {query}

Answer:"""
        
        response = gemini_model.generate_content(prompt)
        answer_text = response.text if hasattr(response, "text") else str(response)
        
        logger.info(f"✅ Answer generated")
        
        return {
            "status": "success",
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "source_count": len(sources)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# =====================
# Routes - Articles Management
# =====================

@app.get("/articles", tags=["Browse"])
def list_articles(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
):
    """Get paginated list of all articles"""
    try:
        skip = (page - 1) * limit
        total = collection.count_documents({})
        articles = list(collection.find(
            {},
            {
                "pmc_id": 1,
                "title": 1,
                "abstract": 1,
                "published_date": 1,
                "_id": 0
            }
        ).skip(skip).limit(limit).sort("published_date", DESCENDING))
        
        return {
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "pages": (total + limit - 1) // limit
            },
            "articles": articles
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, "Failed to list articles")

@app.get("/search", tags=["Browse"])
def search_articles(
    q: str = Query(..., min_length=1),
    field: str = Query("title", enum=["title", "abstract", "keywords"]),
    limit: int = Query(10, ge=1, le=50),
):
    """Search articles by keyword"""
    try:
        search_query = {field: {"$regex": q, "$options": "i"}}
        
        results = list(collection.find(
            search_query,
            {
                "pmc_id": 1,
                "title": 1,
                "abstract": 1,
                "_id": 0
            }
        ).limit(limit))
        
        return {
            "query": q,
            "field": field,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, "Search failed")

@app.get("/articles/{pmc_id}", tags=["Browse"])
def get_article(pmc_id: str):
    """Get full article details"""
    try:
        doc = collection.find_one({"pmc_id": pmc_id})
        if not doc:
            raise HTTPException(404, "Article not found")
        
        doc.pop("_id", None)
        return {"status": "success", "article": doc}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, "Failed to get article")

# =====================
# Routes - Visualization (Mindmaps)
# =====================

@app.get("/articles/{pmc_id}/mindmap", tags=["Visualization"])
def get_article_mindmap(
    pmc_id: str,
    mode: str = Query("research", enum=["overview", "research"]),
):
    """
    🧠 Generate mindmap for article
    
    Modes:
    - overview: Simple abstract + conclusion
    - research: Detailed research breakdown
    """
    try:
        doc = collection.find_one({"pmc_id": pmc_id})
        if not doc:
            raise HTTPException(404, "Article not found")
        
        if mode == "research":
            mindmap = build_research_mindmap(doc)
        else:
            # Simple overview
            title = clean_for_mindmap(doc.get('title', 'Untitled'), 150)
            abstract = clean_for_mindmap(doc.get('abstract', ''), 300)
            conclusion = clean_for_mindmap(doc.get('conclusion', ''), 300)
            
            mindmap = f"""mindmap
  root(({title}))
    Abstract
      {abstract}
    Conclusion
      {conclusion}
"""
        
        return {
            "status": "success",
            "pmc_id": pmc_id,
            "mode": mode,
            "mindmap": mindmap
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, "Failed to generate mindmap")

@app.get("/articles/{pmc_id}/research-data", tags=["Visualization"])
def get_research_data(pmc_id: str):
    """Get structured research data"""
    try:
        doc = collection.find_one({"pmc_id": pmc_id})
        if not doc:
            raise HTTPException(404, "Article not found")
        
        data = extract_research(doc)
        if not data:
            raise HTTPException(500, "Failed to extract research data")
        
        return {
            "status": "success",
            "pmc_id": pmc_id,
            "title": doc.get("title"),
            "research_data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, "Failed to get research data")

# =====================
# Routes - Info
# =====================

@app.get("/stats", tags=["Info"])
def get_stats():
    """Get system statistics"""
    try:
        total_articles = collection.count_documents({})
        
        return {
            "status": "success",
            "statistics": {
                "total_articles": total_articles,
                "embedding_model": "all-MiniLM-L6-v2",
                "llm": "Gemini 2.5 Flash",
                "vector_db": "Pinecone",
                "api_version": "3.0",
                "features": [
                    "RAG-based Q&A",
                    "Research mindmaps",
                    "Article search",
                    "Semantic search"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, "Failed to get stats")

# =====================
# Run
# =====================
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting NASA Research Papers API v3.0...")
    uvicorn.run(app, host="0.0.0.0", port=8000)