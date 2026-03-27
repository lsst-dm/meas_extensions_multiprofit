# This file is part of meas_extensions_multiprofit.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ("WrappedWcsBase",)

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class WrappedWcsBase(ABC):
    """Wrapper for arbitrary WCS implementations.

    Implementations do not necessarily need to meet all FITS WCS standards as
    long as all abstract methods work correctly.
    """

    def convert_moments(
        self,
        i_xx: float,
        i_yy: float,
        i_xy: float,
        cd_matrix: np.ndarray,
    ) -> tuple[float, float, float]:
        """Convert moments/ellipse parameters from pixel to sky coordinates.

        Parameters
        ----------
        i_xx
            The sum of squared x-axis moments.
        i_yy
            The sum of squared y-axis moments.
        i_xy
            The sum of the product of x-y axis moments.
        cd_matrix
            The FITS CD matrix terms.

        Returns
        -------
        i_uu, i_vv, i_uv
            The on-sky moments.
        """
        CD_1_1 = cd_matrix[0, 0]
        CD_1_2 = cd_matrix[0, 1]
        CD_2_1 = cd_matrix[1, 0]
        CD_2_2 = cd_matrix[1, 1]

        i_uu = CD_1_1 * (i_xx * CD_1_1 + i_xy * CD_2_1) + CD_1_2 * (i_xy * CD_1_1 + i_yy * CD_2_1)
        i_vv = CD_2_1 * (i_xx * CD_1_2 + i_xy * CD_2_2) + CD_2_2 * (i_xy * CD_1_2 + i_yy * CD_2_2)
        i_uv = (CD_1_1 * i_xx + CD_1_2 * i_xy) * CD_2_1 + (CD_1_1 * i_xy + CD_1_2 * i_yy) * CD_2_2

        return i_uu, i_vv, i_uv

    @abstractmethod
    def get_cd_matrix(self) -> np.ndarray:
        """Return the WCS' CD matrix."""

    @abstractmethod
    def get_wcs(self) -> Any:
        """Return the wrapped WCS object."""

    @abstractmethod
    def get_pixel_to_ra_dec(self, x: float, y: float) -> tuple[float, float]:
        """Convert pixel coordinates to RA, dec in degrees.

        Parameters
        ----------
        x
            The x-axis pixel coordinate.
        y
            The y-axis pixel coordinate.

        Returns
        -------
        ra, dec
            Right ascension and declination in degrees.
        """
