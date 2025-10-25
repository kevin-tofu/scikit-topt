from typing import Optional
from dataclasses import dataclass

import numpy as np

import skfem


@dataclass
class BaseFilter():
    mesh: skfem.Mesh
    elements_volume: np.ndarray
    radius: float
    design_mask: Optional[np.ndarray] = None

    def update_radius(
        self,
        radius: float,
        **args
    ):
        raise NotImplementedError("")

    @classmethod
    def from_defaults(
        cls,
        mesh: skfem.Mesh,
        elements_volume: np.ndarray,
        radius: float = 0.3,
        design_mask: Optional[np.ndarray] = None
    ) -> 'BaseFilter':
        raise NotImplementedError("")

    def run(self, rho_element: np.ndarray) -> np.ndarray:
        raise NotImplementedError("")

    def gradient(self, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError("")
