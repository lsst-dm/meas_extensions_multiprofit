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

__all__ = (
    "CatalogAction",
    "MergeMultibandFluxes",
)

from collections import defaultdict

import astropy.table
import numpy as np

import lsst.pex.config as pexConfig
from lsst.multiprofit.fitting.fit_catalog import CatalogFitterConfig
from lsst.pex.config.configurableActions import ConfigurableAction


class CatalogAction(ConfigurableAction):
    """Configurable action to return a catalog."""

    def __call__(self, data, **kwargs):
        """Return a catalog, potentially modified in-place.

        Parameters
        ----------
        data
            A dict-like catalog.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        data
            The original data, modified in-place.
        """
        return data


class MergeMultibandFluxes(CatalogAction):
    """Configurable action to merge single-band flux tables into one."""

    name_model = pexConfig.Field[str](doc="The name of the model that fluxes are measured from", default="")

    def __call__(self, data: astropy.table.Table, **kwargs):
        datasetType = kwargs.get("datasetType")
        prefix_model = self.name_model + ("_" if self.name_model else "")

        # Check if the table metadata has relevant config settings
        if (
            self.name_model
            and hasattr(data, "meta")
            and datasetType
            and (config := data.meta.get(datasetType))
        ):
            config_dict = config.get("config", {})
            prefix = config_dict.get("prefix_column", CatalogFitterConfig.prefix_column.default)
            suffix_error = config_dict.get("suffix_error", CatalogFitterConfig.suffix_error.default)
            column_id = config_dict.get("column_id")
        else:
            prefix = CatalogFitterConfig.prefix_column.default
            suffix_error = CatalogFitterConfig.suffix_error.default
            column_id = "id" if "id" in data.colnames else None

        columns_rest = [] if prefix else ([column_id] if column_id else [])
        columns_flux_band = defaultdict(list)
        for column in data.columns:
            if not prefix or column.startswith(prefix):
                if column.endswith("_flux"):
                    band = column.split("_")[-2]
                    columns_flux_band[band].append(column)
            else:
                columns_rest.append(column)

        columns_exclude_prefix = set(columns_rest) if prefix_model else set()

        for band, columns_band in columns_flux_band.items():
            column_flux = f"{band}_{prefix_model}flux"
            column_flux_err = f"{column_flux}{suffix_error}"
            if len(columns_band) > 1:
                # Sum up component fluxes and make a total flux column
                flux = np.nansum([data[column] for column in columns_band], axis=0)
                data[column_flux] = flux

                columns_band_err = [f"{column}{suffix_error}" for column in columns_band]
                errors = [data[column] ** 2 for column in columns_band_err if column in data.columns]
                if errors:
                    flux_err = np.sqrt(np.nansum(errors, axis=0))
                    flux_err[flux_err == 0] = np.nan
                    data[column_flux_err] = flux_err
                    columns_exclude_prefix.add(column_flux_err)
            else:
                data.rename_columns(
                    (columns_band[0], f"{columns_band[0]}{suffix_error}"), (column_flux, column_flux_err)
                )

            columns_exclude_prefix.add(column_flux)
            columns_exclude_prefix.add(f"{column_flux}{suffix_error}")

        if prefix_model:
            # Add prefixes to the column names, if needed
            colnames = [
                (
                    col
                    if (col in columns_exclude_prefix)
                    else (
                        f"{prefix}{prefix_model if (prefix_model != prefix) else ''}"
                        f"{col.split(prefix, 1)[1] if prefix else col}"
                    )
                )
                for col in data.columns
            ]
            if hasattr(data, "rename_columns"):
                data.rename_columns([x for x in data.columns], colnames)
            else:
                data.columns = colnames

        return data
