import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BigBirdTokenizer, BigBirdModel
from data.blocked_tags import BLOCKED_TAGS

def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj

class BigBirdForTags(nn.Module):
    def __init__(self, num_classes, model_name):
        super().__init__()
        self.body = BigBirdModel.from_pretrained(model_name)
        hidden = self.body.config.hidden_size
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def encode(self, input_ids, attention_mask):
        out = self.body(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.norm(out.last_hidden_state[:, 0])
        return cls

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        cls = self.encode(input_ids, attention_mask)
        if return_embeddings:
            return cls
        return self.classifier(cls)

class NeuralSearchModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BigBirdTokenizer.from_pretrained(config.TOKENIZER_PATH)
        with open(config.MLB_PATH, "rb") as f:
            self.mlb = pickle.load(f)
        self.model = BigBirdForTags(len(self.mlb.classes_), config.MODEL_PATH)
        self.model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=self.device))
        self.model.to(self.device).eval()
        emb = torch.load(config.EMBEDDINGS_PATH, map_location="cpu")["train_embeddings"]
        self.embeddings = emb.numpy() if isinstance(emb, torch.Tensor) else emb
        self.dataset = pd.read_json(config.DATASET_PATH)
        self.dataset["tags"] = self.dataset["tags"].apply(self._normalize_tags)

    def _normalize_tags(self, tags):
        if tags is None or (isinstance(tags, float) and np.isnan(tags)):
            return []
        if isinstance(tags, str):
            tags = tags.split(",")
        return [str(t).lower().strip() for t in tags if str(t).strip()]

    def prepare_query_text(self, title, summary, fandoms, characters, story):
        parts = []
        if title:
            parts.append(f"[TITLE] {title}")
        if summary:
            parts.append(f"[SUMMARY] {summary}")
        if fandoms:
            parts.append(f"[FANDOMS] {fandoms}")
        if characters:
            parts.append(f"[CHARACTERS] {characters}")
        if story:
            parts.append(f"[STORY] {story}")
        return "\n".join(parts)

    def get_query_embedding(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.MAX_LENGTH,
            return_tensors="pt"
        )
        with torch.no_grad():
            emb = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
                return_embeddings=True
            )
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

    def predict_tags(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.MAX_LENGTH,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        idx = np.where(probs > self.config.TAG_THRESHOLD)[0]
        return [self.mlb.classes_[i].lower().strip() for i in idx]

    def search(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.config.top_k

        if not hasattr(self, "reranker"):
            from transformers import AutoTokenizer, AutoModel
            self.reranker_tokenizer = AutoTokenizer.from_pretrained("models/bge-reranker-large")
            self.reranker_model = AutoModel.from_pretrained("models/bge-reranker-large").to(self.device).eval()

        query_emb = self.get_query_embedding(query_text)
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        query_tags = set(self.predict_tags(query_text))

        max_tag_matches = len(query_tags) if query_tags else 1

        results = []
        for idx, sim in enumerate(similarities):
            if idx >= len(self.dataset):
                continue

            row = self.dataset.iloc[idx]
            tags = row.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            work_tags = set(tags)

            tag_matches = len(query_tags.intersection(work_tags))
            normalized_cosine = (sim + 1) / 2
            normalized_tags = tag_matches / max_tag_matches if max_tag_matches > 0 else 0
            score = 0.8 * normalized_tags + 0.2 * normalized_cosine

            rating = str(row.get("rating", "")).lower().strip()
            categories = row.get("categories", [])
            categories = [str(c).lower().strip() for c in categories] if isinstance(categories, list) else []

            is_blocked = (
                    rating in {"explicit", "mature"}
                    or any(cat in {"m/m", "f/f"} for cat in categories)
                    or any(tag in BLOCKED_TAGS for tag in tags)
            )

            results.append({
                "blocked": is_blocked,
                'index': int(idx),
                'score': float(score),
                'cosine_sim': float(sim),
                'tag_bonus': float(tag_matches),
                'text_sample': str(row.get('text_sample', '') if not is_blocked else "")
            })

        results.sort(key=lambda x: x['score'], reverse=True)

        rerank_candidates = results[:self.config.RERANKER_CANDIDATES]
        if rerank_candidates:
            texts = [self.dataset.iloc[r['index']].get('text_sample', '') for r in rerank_candidates]
            enc = self.reranker_tokenizer([query_text] * len(texts), texts, padding=True, truncation=True,
                                          return_tensors="pt").to(self.device)
            with torch.no_grad():
                rerank_emb = self.reranker_model(**enc).last_hidden_state[:, 0, :]
                rerank_scores = torch.nn.functional.cosine_similarity(rerank_emb[:1], rerank_emb[1:],
                                                                      dim=-1).cpu().numpy()
            for i, r in enumerate(rerank_candidates):
                r['score'] = float(r['score'] + rerank_scores[i - 1] if i > 0 else r['score'])
            rerank_candidates.sort(key=lambda x: x['score'], reverse=True)

        search_results = []
        for res in rerank_candidates[:top_k]:
            idx = res['index']
            row = self.dataset.iloc[idx]

            if res["blocked"]:
                search_results.append({
                    'title': str("Работа заблокирована"),
                    'summary': str("Работа заблокирована"),
                    'tags': list([]),
                    'fandoms': list([]),
                    'characters': list([]),
                    'categories': list([]),
                    'relationships': list([]),
                    'work_id': str(row.get('work_id')),
                    'similarity_score': float(res['score']),
                    'cosine_similarity': float(res['cosine_sim']),
                    'tag_overlap': int(res['tag_bonus'])
                })
            else:
                search_results.append({
                    'title': str(row.get('title', 'Без названия')),
                    'summary': str(row.get('summary', '')),
                    'tags': list(row.get('tags', [])),
                    'predicted_tags': list(query_tags),
                    'fandoms': list(row.get('fandoms', [])),
                    'characters': list(row.get('characters', [])),
                    'categories': list(row.get('categories', [])),
                    'relationships': list(row.get('relationships', [])),
                    'work_id': str(row.get('work_id')),
                    'similarity_score': float(res['score']),
                    'cosine_similarity': float(res['cosine_sim']),
                    'tag_overlap': int(res['tag_bonus'])
                })

        return to_python(search_results)

    def add_new_work(self, work_data):
        if not work_data.get("work_id"):
            return False, "work_id обязателен"

        work_data["full_text_input"] = self.prepare_query_text(
            work_data.get("title", ""),
            work_data.get("summary", ""),
            work_data.get("fandoms", ""),
            work_data.get("characters", ""),
            work_data.get("text_sample", "")
        )
        work_data["tags"] = self._normalize_tags(work_data.get("tags", []))

        new_row = pd.DataFrame([work_data])

        self.dataset = pd.concat([self.dataset, new_row], ignore_index=True)

        emb = self.get_query_embedding(work_data["full_text_input"])
        self.embeddings = np.vstack([self.embeddings, emb])

        self.dataset.to_json(
            self.config.DATASET_PATH,
            orient="records",
            force_ascii=False,
            indent=2
        )

        torch.save({
            "train_embeddings": torch.tensor(self.embeddings)
        }, self.config.EMBEDDINGS_PATH)

        return True, work_data["work_id"]