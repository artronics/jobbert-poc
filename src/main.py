from typing import List, Tuple

from fastapi import FastAPI
import uvicorn
from sentence_transformers.util import cos_sim

from data_types import SocTitleMatching, SocJobAdvertMatching, Advert, SkillsJobAdvertMatching
from datasets import get_soc_dataset, get_skills_dataset
from embeddings import make_embeddings
from model import JobBertV2
from numpy.typing import NDArray

app = FastAPI()
model = JobBertV2(batch_size=8)

soc_dataset = get_soc_dataset()
soc_titles: NDArray = soc_dataset.dataset

# Below calculate the embeddings for all the SOC codes
soc_embeddings = make_embeddings(model, "soc_titles", soc_titles)

skills_dataset = get_skills_dataset()
skills_titles= skills_dataset.dataset
skills_embeddings = make_embeddings(model, "skills", skills_titles)

def _match_soc(texts: List[str]) -> List[Tuple[str, float]]:
    """Given an input `text`, we sort all the SOC codes based on similarity to the text"""
    text_emb = model.encode(texts)[0]
    similarities = cos_sim(text_emb, soc_embeddings)[0].numpy()
    return sorted(zip(soc_titles, similarities), key=lambda x: x[1], reverse=True)

def _match_skills(texts: List[str]) -> List[Tuple[str, float]]:
    """Given an input `text`, we sort all the skill titles based on similarity to the text"""
    text_emb = model.encode(texts)[0]
    similarities = cos_sim(text_emb, skills_embeddings)[0].numpy()
    return sorted(zip(skills_titles, similarities), key=lambda x: x[1], reverse=True)


@app.get("/socs")
async def socs():
    return get_soc_dataset()


@app.get("/match/socs")
async def match_soc_titles(q: str, limit: int = 10):
    """Return SOC titles that matches the query string `q`."""
    similarities = _match_soc([q])[:limit]

    titles = [item[0] for item in similarities]
    # scores = [float(item[1]) for item in similarities] # We need to convert float32 to float

    return SocTitleMatching(original_title=q, soc_titles=titles)


@app.post("/match/socs")
async def match_soc_titles(advert: Advert, limit: int = 10):
    """Return SOC titles that match the string in the request body."""
    similarities = _match_soc(advert.contents)[:limit]
    titles = [item[0] for item in similarities]

    return SocJobAdvertMatching(soc_titles=titles)

@app.post("/match/skills")
async def match_soc_titles(advert: Advert, limit: int = 10):
    """Return skills that match the string in the request body."""
    similarities = _match_skills(advert.contents)[:limit]
    skills = [item[0] for item in similarities]

    return SkillsJobAdvertMatching(skills=skills)

@app.get("/health")
async def health():
    return {"message": "ok"}


def main():
    port = 8080
    print(f"Starting the server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
