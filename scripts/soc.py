"""All SOC related functions are here, including:
Removing duplicated titles from a list of all job titles,
"""
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def clean_soc_titles():
    soc_titles = []
    with open(DATA_PATH / "soc_titles.json", "r") as f:
        soc_titles = json.load(f)

    cleaned_titles = [title.strip() for title in soc_titles if len(title) > 3]
    unique_titles = set(cleaned_titles)
    sorted_titles = sorted(unique_titles)

    with open(DATA_PATH / "soc_titles_cleaned.json", "w") as f:
        json.dump(list(sorted_titles), f, indent=4)


def main():
    clean_soc_titles()


if __name__ == "__main__":
    main()
