import numpy as np
from dataclasses import dataclass, field
from textwrap import dedent

from util.util import normalize_img


@dataclass
class SCCompressedImg:
    compressed: np.ndarray
    U: np.ndarray
    S: np.ndarray
    V: np.ndarray
    energy_factor: float
    rank: int
    compression_ratio: float = field(init=False)

    def __post_init__(self):
        self.compressed = normalize_img(self.compressed)
        self.compression_ratio = _calculate_compression_ratio(
            self.compressed, self.rank
        )

    def __getitem__(self, idx):
        props = (
            self.compressed,
            self.U,
            self.S,
            self.V,
            self.energy_factor,
            self.rank,
            self.compression_ratio,
        )
        return props[idx]

    def __repr__(self):
        return "SCCompressedImg(compressed_img, U, S, V, energy_factor)"

    def __str__(self):
        name = self.__class__.__name__
        return dedent(
            f"""
            \n
            {'*'*50}
            {name} Object

            Image Shape: {self.compressed.shape}
            Energy Factor: {self.energy_factor:.2%}
            Rank: {self.rank}
            Compression Ratio: {self.compression_ratio:.2%}
            {'*'*50}
            \n
            """
        )


@dataclass
class MCCompressedImg:
    compressed: np.ndarray
    R: SCCompressedImg
    G: SCCompressedImg
    B: SCCompressedImg
    energy_factor: float
    min_chan_rank: int = field(init=False)
    compression_ratio: float = field(init=False)

    def __post_init__(self):
        self.compressed = normalize_img(self.compressed)
        self.min_chan_rank = min([self.R.rank, self.G.rank, self.B.rank])
        self.compression_ratio = (
            self.R.compression_ratio
            + self.G.compression_ratio
            + self.B.compression_ratio
        )

    def __getitem__(self, idx):
        props = (
            self.compressed,
            self.R,
            self.G,
            self.B,
            self.energy_factor,
            self.min_chan_rank,
            self.compression_ratio,
        )
        return props[idx]

    def __repr__(self):
        return "MCCompressedImg(compressed_img, R, G, B, energy_factor)"

    def __str__(self):
        name = self.__class__.__name__
        return dedent(
            f"""
            \n
            {'*'*50}
            {name} Object

            Image Shape: {self.compressed.shape}
            Energy Factor: {self.dist_rate:.2%}
            Minimum Channel Rank: {self.min_chan_rank}
            Compression Ratio: {self.compression_ratio:.2%}
            {'*'*50}
            \n
            """
        )


def _calculate_compression_ratio(img, rank):
    m = len(img[0])
    n = len(img[1])

    return (m + n + 1) * rank / (m * n)
