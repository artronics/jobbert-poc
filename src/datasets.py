import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import kagglehub
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cache import Cache
from data_types import Advert

DATA_PATH = Path(__file__).parent.parent / "data"
SOC_TITLES = DATA_PATH / "soc_titles_cleaned.json"
_SOCS = json.load(SOC_TITLES.open())

cache = Cache()


def _sort_and_cleanup(data: NDArray) -> NDArray:
    unique = np.unique(data)
    sorted_indices = np.argsort([len(item) for item in unique])
    length_sorted = [unique[i] for i in sorted_indices]
    return np.array(length_sorted)


@dataclass
class BaseDataset:
    dataset: NDArray
    name: str

    def __init__(self, name: str, dataset: NDArray):
        self.name = name
        self.dataset = dataset

    def choice(self):
        return np.random.choice(self.dataset)


def get_soc_dataset():
    job_titles = np.array(_SOCS)

    return BaseDataset("soc_titles", job_titles)


def get_skills_dataset() -> BaseDataset:
    skills_path = DATA_PATH / "skills_10k_cleaned.json"
    with open(skills_path, "r") as f:
        skills = json.load(f)

    arr = np.array(skills, dtype=str)
    cache.set("skills_10k", arr)

    return BaseDataset("skills_10k", arr)


def get_job_adverts_dataset() -> BaseDataset:
    ds_name = "ds_armenian_online_job_postings"
    if cache.exists(ds_name):
        adverts = cache.get(ds_name)
        BaseDataset(ds_name, adverts)

    path = kagglehub.dataset_download("udacity/armenian-online-job-postings")
    df = pd.read_csv(f"{path}/online-job-postings.csv")

    df["title"] = df["Title"].fillna("").astype(str)
    df["jobDescription"] = df["JobDescription"].fillna("").astype(str)
    df["jobRequirement"] = df["JobRequirment"].fillna("").astype(str)

    adverts = []
    for _, row in df.iterrows():
        title = row["title"]
        contents = [row['jobDescription'], row['jobRequirement']]
        adverts.append(Advert(title, contents))

    adverts = np.array(adverts)
    cache.set(ds_name, adverts)

    return BaseDataset(ds_name, adverts)


def filter_by_keywords(data: NDArray, keywords: List[str], case_sensitive: bool = False) -> List[Tuple[int, str]]:
    """
    Filter entries in a NumPy array that contain ALL specified keywords.

    Args:
        data: NumPy NDArray containing string entries
        keywords: List of keywords that must ALL be present in an entry
        case_sensitive: Whether the search should be case-sensitive (default: False)

    Returns:
        List of tuples containing (index, matching_entry)
    """
    _matches = []

    for idx, entry in enumerate(data):
        text = entry.contents[0] + entry.contents[1]
        compare_text = text if case_sensitive else text.lower()
        compare_keywords = keywords if case_sensitive else [kw.lower() for kw in keywords]

        if all(keyword in compare_text for keyword in compare_keywords):
            _matches.append((idx, entry))

    return _matches


if __name__ == "__main__":
    # socs = get_soc_dataset()
    # jobs = get_job_adverts_dataset()
    # matches = filter_by_keywords(jobs.dataset, ["Embedded",  "system", "c++"])
    get_skills_dataset()
    # print(matches[0][1])
    # print(socs.choice())
