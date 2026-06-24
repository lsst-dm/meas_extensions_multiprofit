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

import lsst.pex.config as pexConfig
from lsst.pex.config.configurableActions import ConfigurableActionField
import lsst.pipe.base.connectionTypes as connectionTypes

from .catalog_actions import CatalogAction


class InputConfig(pexConfig.Config):
    """Config for inputs to ConsolidateAstropyTableTask."""

    doc = pexConfig.Field[str](doc="Doc for connection", optional=False)
    action = ConfigurableActionField[CatalogAction](
        doc="Action to modify the input table",
        default=None,
    )
    columns = pexConfig.ListField[str](
        doc="Column names to copy; default of None copies all", optional=True, default=None
    )
    column_id = pexConfig.Field[str](doc="ID column to merge", optional=False, default="objectId")
    is_multiband = pexConfig.Field[bool](doc="Whether the dataset is multiband or not", default=False)
    is_multipatch = pexConfig.Field[bool](doc="Whether the dataset is multipatch or not", default=False)
    join_column = pexConfig.Field[str](
        doc="Column to join on if unequal length instead of stacking", default=None, optional=True
    )
    needs_metadata = pexConfig.Field[bool](doc="Whether the input needs metadata read", default=False)
    storageClass = pexConfig.Field[str](doc="Storage class for DatasetType", default="ArrowAstropy")

    def get_connection(self, name: str) -> connectionTypes.Input:
        dimensions = ["skymap", "tract"]
        if not self.is_multipatch:
            dimensions.append("patch")
        if not self.is_multiband:
            dimensions.append("band")
        connection = connectionTypes.Input(
            doc=self.doc,
            name=name,
            storageClass=self.storageClass,
            dimensions=dimensions,
            multiple=not (self.is_multiband and self.is_multipatch),
            deferLoad=(self.columns is not None) or self.needs_metadata,
        )
        return connection
