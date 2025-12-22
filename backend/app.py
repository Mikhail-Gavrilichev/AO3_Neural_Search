from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from models.ollama_client import OllamaExplainer

from config import settings
from models.neural_model import NeuralSearchModel
from enum import Enum

app = FastAPI(
    title="Neural Search API для литературных произведений",
    description="API для нейронного поиска и автоматического присвоения тегов",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_model = NeuralSearchModel(settings)
llm_explainer = OllamaExplainer(
    model_name="deepseek-r1:8b"
)

class LLMMode(str, Enum):
    off = "off"
    query_only = "query_only"
    full = "full"
class SearchRequest(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    fandoms: Optional[str] = None
    characters: Optional[str] = None
    story: Optional[str] = None
    top_k: Optional[int] = 5

    llm_mode: LLMMode = Field(
        LLMMode.full,
        description="Режим использования LLM"
    )


class NewWorkRequest(BaseModel):
    work_id: str = Field(..., min_length=1, description="ID произведения")
    title: str = Field(..., description="Название произведения")
    summary: Optional[str] = Field(None, description="Краткое описание")
    tags: Optional[List[str]] = Field([], description="Теги")
    fandoms: Optional[List[str]] = Field([], description="Фэндомы")
    characters: Optional[List[str]] = Field([], description="Персонажи")
    categories: Optional[List[str]] = Field([], description="Категории")
    relationships: Optional[List[str]] = Field([], description="Отношения/пейринги")
    text_sample: Optional[str] = Field(None, description="Отрывок текста")


class DeveloperInfo(BaseModel):
    name: str
    group: str
    email: str
    role: str


@app.get("/")
async def root():
    return {
        "message": "Neural Search API для литературных произведений",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search (POST)",
            "add_work": "/add-work (POST)",
            "developers": "/developers (GET)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "embeddings_shape": search_model.embeddings.shape
    }


@app.post("/search")
async def search_similar_works(request: SearchRequest):
    try:
        query_text = search_model.prepare_query_text(
            request.title or "",
            request.summary or "",
            request.fandoms or "",
            request.characters or "",
            request.story or ""
        )

        results = search_model.search(query_text, top_k=request.top_k)

        llm_answer = None

        if request.llm_mode == LLMMode.query_only:
            llm_answer = llm_explainer.explain(
                query_text=query_text,
                retrieved_works=results,
                include_full_text=False
            )

        elif request.llm_mode == LLMMode.full:
            llm_answer = llm_explainer.explain(
                query_text=query_text,
                retrieved_works=results,
                include_full_text=True
            )

        elif request.llm_mode == LLMMode.off:
            llm_answer = None

        return {
            "query": {
                "title": request.title,
                "summary": request.summary,
                "fandoms": request.fandoms,
                "characters": request.characters
            },
            "results_count": len(results),
            "results": results,
            "llm_explanation": llm_answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-work")
async def add_new_work(work: NewWorkRequest):
    try:
        if not work.work_id or not work.work_id.strip():
            raise HTTPException(
                status_code=400,
                detail="work_id обязателен"
            )

        if str(work.work_id) in search_model.dataset["work_id"].astype(str).values:
            raise HTTPException(
                status_code=409,
                detail="Работа с таким work_id уже существует"
            )

        work_data = {
            "work_id": work.work_id,
            "title": work.title,
            "summary": work.summary,
            "tags": work.tags,
            "fandoms": work.fandoms,
            "characters": work.characters,
            "categories": work.categories,
            "relationships": work.relationships,
            "text_sample": work.text_sample
        }

        success, result = search_model.add_new_work(work_data)

        if success:
            return {
                "success": True,
                "message": "Работа успешно добавлена",
                "work_id": result
            }
        else:
            raise HTTPException(status_code=500, detail=f"Ошибка при добавлении: {result}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get("/dataset-info")
async def get_dataset_info():
    return {
        "total_works": len(search_model.dataset),
        "total_tags": len(search_model.mlb.classes_),
        "embeddings_dimension": search_model.embeddings.shape[1]
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )