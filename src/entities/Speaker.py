from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Utterance:
    name: str
    raw_data: List
    frame_rate: int
    mfcc_frames: np.ndarray
    mfcc_stacked_frames: np.ndarray
    d_vector: np.ndarray


@dataclass
class Speaker:
    name: str
    id: int
    utterances: List[Utterance]
    d_vector: np.ndarray
