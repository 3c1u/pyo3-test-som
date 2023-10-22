import numpy.typing as npt
from typing import Unpack, TypedDict

class SomFitOptions(TypedDict):
    tau: float
    max_iter: int

class Som:
    """
    Som class.
    """

    def __init__(self, dim_latent: int, dim_input: int) -> None:
        """
        Som constructor.
        """
        pass
    # getter
    def latent(self) -> npt.ArrayLike:
        """
        Get latent.
        """
    def fit(
        self, input: npt.ArrayLike, grid: npt.ArrayLike, **kwargs: Unpack[SomFitOptions]
    ) -> None:
        """
        Fit input.
        """
        pass
