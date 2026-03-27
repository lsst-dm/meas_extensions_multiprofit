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

__all__ = ("WrappedSkyWcs",)

from functools import cached_property
from typing import Any

import numpy as np
import pydantic

import lsst.afw.geom

from .wrappedwcsbase import WrappedWcsBase


@pydantic.dataclasses.dataclass(
    frozen=True, kw_only=True, config=pydantic.ConfigDict(arbitrary_types_allowed=True)
)
class WrappedSkyWcs(WrappedWcsBase):
    """Wrapper for afw's SkyWcs."""

    wcs: lsst.afw.geom.SkyWcs = pydantic.Field(title="The SkyWcs to wrap")

    @cached_property
    def cd_matrix(self) -> np.ndarray:
        return self.wcs.getCdMatrix()

    def get_cd_matrix(self) -> np.ndarray:
        return self.cd_matrix

    def get_wcs(self) -> Any:
        return self.wcs

    def get_pixel_to_ra_dec(self, x: float, y: float) -> tuple[float, float]:
        return tuple(x.asDegrees() for x in self.wcs.pixelToSky(x, y))
