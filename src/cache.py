import hashlib
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

import numpy
import numpy as np
from numpy.typing import NDArray

CACHE_PATH = Path(__file__).parent.parent / ".cache"


def _list_files():
    return [f for f in listdir(CACHE_PATH) if isfile(join(CACHE_PATH, f))]


class Cache(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print('Creating cache')
            cls._instance = super(Cache, cls).__new__(cls)
            if not Path(CACHE_PATH).exists():
                os.makedirs(CACHE_PATH)

        return cls._instance

    @staticmethod
    def exists(key: str) -> bool:
        keys = [Path(f).stem for f in _list_files()]
        return key in keys

    @staticmethod
    def get(key: str) -> NDArray:
        file_path = f"{CACHE_PATH}/{key}.npy"
        print(f"Loading cache file {file_path}")
        return numpy.load(file_path, allow_pickle=True)

    @staticmethod
    def set(key: str, data: NDArray) -> None:
        file_path = CACHE_PATH / f"{key}.npy"
        print(f"Saving cache file {file_path}")
        np.save(file_path, data, allow_pickle=True)

    @staticmethod
    def sha256sum(filename):
        h = hashlib.sha256()
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        with open(filename, 'rb', buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                h.update(mv[:n])
        return h.hexdigest()
