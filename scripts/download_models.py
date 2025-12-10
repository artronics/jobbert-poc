from pathlib import Path
from sentence_transformers import SentenceTransformer

CACHE_PATH = Path(__file__).parent.parent / ".cache"

MODELS = [
    {"path": "TechWolf/JobBERT-v2", "name": "JobBERT-v2"},
]


def download_model(model_data: object):
    destination = CACHE_PATH / model_data["name"]

    print(f"Downloading {model_data['path']} to {destination}...")
    model = SentenceTransformer(model_data['path'])
    model.save(str(destination))
    print("Download finished.")


def main():
    for model in MODELS:
        download_model(model)


if __name__ == "__main__":
    main()
