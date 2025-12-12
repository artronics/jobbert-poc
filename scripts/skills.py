import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data"

def make_skills_dataset():
    """this creates a clean json file from a skill2vec file"""
    filename = "skill2vec_10K.csv"
    csv_path = DATA_PATH / filename

    df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False)
    raw = df.to_numpy().ravel()

    result = []
    for val in raw:
        val = val.lower()
        val = val.replace("&ampquot", "")
        val = val.replace("&quot", "")
        val = val.replace("&amp", "")
        val = val.replace(";", "")
        val = val.replace("#", "")
        val = val.replace("*", "")
        val = val.replace("_", " ")
        val = val.strip()
        if val and not val.isdigit():
            result.append(val)

    cleaned = np.unique(np.array(result, dtype=str))
    with open(DATA_PATH / "skills_10k_cleaned.json", "w") as f:
        json.dump(cleaned.tolist(), f, indent=4)


if __name__ == "__main__":
    make_skills_dataset()
