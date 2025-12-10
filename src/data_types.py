from dataclasses import dataclass
from typing import List


@dataclass
class JobTitle:
    title: str

    def __str__(self):
        return self.title


@dataclass
class Advert:
    title: str
    contents: List[str]


class Matching:
    """Response object contains **sorted** titles. Scores are provided as a separate list."""
    titles: List[str]
    scores: List[float]

    def __init__(self, titles: List[str], scores: List[float]):
        self.titles = titles
        self.scores = scores

    def as_dict(self):
        return {
            'titles': self.titles,
            'scores': self.scores
        }


@dataclass
class SocTitleMatching:
    original_title: str
    soc_titles: List[str]

    def __init__(self, original_title: str, soc_titles: List[str]):
        self.original_title = original_title
        self.soc_titles = soc_titles

    def as_dict(self):
        return {
            'original_title': self.original_title,
            'soc_titles': self.soc_titles,
        }


@dataclass
class SocJobAdvertMatching:
    soc_titles: List[str]

    def __init__(self, soc_titles: List[str]):
        self.soc_titles = soc_titles

    def as_dict(self):
        return {
            'soc_titles': self.soc_titles,
        }
