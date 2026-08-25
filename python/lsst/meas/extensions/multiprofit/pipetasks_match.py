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
    "MultiProFitMatchTractCatalogConfig",
    "MultiProFitMatchTractCatalogDC2Config",
    "MultiProFitMatchTractCatalogDC2Task",
)

from lsst.pipe.tasks.diff_matched_tract_catalog import (
    DiffMatchedTractCatalogConfig,
    DiffMatchedTractCatalogConnections,
    DiffMatchedTractCatalogTask,
)
from lsst.pipe.tasks.match_tract_catalog import (
    MatchTractCatalogConfig,
    MatchTractCatalogConnections,
    MatchTractCatalogTask,
)
from lsst.pipe.tasks.match_tract_catalog_probabilistic import MatchTractCatalogProbabilisticTask


class MultiProFitMatchTractCatalogConfig(
    MatchTractCatalogConfig,
    pipelineConnections=MatchTractCatalogConnections,
):
    """Generic MultiProFit reference source match task config."""

    def setDefaults(self):
        self.connections.name_input_cat_target = "objectTable_tract_multiprofit"
        self.match_tract_catalog.retarget(MatchTractCatalogProbabilisticTask)

        self.match_tract_catalog.columns_ref_copy = ["id", "truth_type"]
        self.match_tract_catalog.columns_target_copy = ["objectId"]
        # Must be set since there's no default - subclasses should add to this
        self.match_tract_catalog.columns_ref_meas = ["ra", "dec"]
        # Subclasses must format these
        columns_meas = ["{model_prefix}_cen_ra", "{model_prefix}_cen_dec"]
        self.match_tract_catalog.columns_target_meas = columns_meas
        self.match_tract_catalog.columns_target_err = [f"{col}_err" for col in columns_meas]
        # Override detect_isPrimary default because MultiProFit doesn't fit
        # non-primary rows anyway
        self.match_tract_catalog.columns_target_select_true = []


class MultiProFitMatchTractCatalogDC2Config(
    MultiProFitMatchTractCatalogConfig,
    pipelineConnections=MatchTractCatalogConnections,
):
    """PipelineTaskConfig for MultiProFitMatchDC2CatalogTask."""

    def finalize(
        self,
        model_prefix: str,
        bands_match: list[str] | None = None,
    ):
        """Finalize configuration for a given model.

        Parameters
        ----------
        model_prefix
            The model column prefix, e.g. mpf_Sersic.
        bands_match
            List of bands to match fluxes on.
        """
        if bands_match is None:
            bands_match = ["u", "g", "r", "i", "z", "y"]
        fluxes_ref = [f"flux_{band}" for band in bands_match]
        self.match_tract_catalog.columns_ref_flux = fluxes_ref
        self.match_tract_catalog.columns_ref_meas += fluxes_ref
        fluxes_meas = [f"{model_prefix}_{band}_flux" for band in bands_match]
        self.match_tract_catalog.columns_target_meas = [
            col.format(model_prefix=model_prefix) for col in self.match_tract_catalog.columns_target_meas
        ] + fluxes_meas
        self.match_tract_catalog.columns_target_err = [
            f"{col}_err" for col in self.match_tract_catalog.columns_target_meas
        ]
        self.match_tract_catalog.coord_format.column_target_coord1 = f"{model_prefix}_cen_ra"
        self.match_tract_catalog.coord_format.column_target_coord2 = f"{model_prefix}_cen_dec"
        self.match_tract_catalog.columns_target_select_false = [
            f"{model_prefix}_not_primary_flag",
        ]

    def setDefaults(self):
        super().setDefaults()
        self.match_tract_catalog.mag_faintest_ref = 27.0
        self.match_tract_catalog.columns_ref_select_true = ["is_unique_truth_entry"]


class MultiProFitMatchTractCatalogDC2Task(MatchTractCatalogTask):
    """Match DC2 truth_summary to a single model from an
    objectTable_tract_multiprofit.
    """

    _DefaultName = "multiProFitMatchTractCatalogDC2"
    ConfigClass = MultiProFitMatchTractCatalogDC2Config


class MultiProFitDiffMatchedTractCatalogConfig(
    DiffMatchedTractCatalogConfig,
    pipelineConnections=DiffMatchedTractCatalogConnections,
):
    """Generic MultiProFit reference matched catalog writing task config."""

    def _finalize_models(
        self,
        model_prefixes: str | list[str],
        bands: list[str] | None = None,
        fluxes_include: dict[str, list[str]] | None = None,
        sizes_include: dict[str, list[str]] | None = None,
        sersics_include: dict[str, list[str]] | None = None,
        is_v2: bool = False,
    ):
        """Finalize matched catalog configuration for given models.

        Total source fluxes are included for all bands. Individual component
        fluxes are optional and must be specified.

        Parameters
        ----------
        model_prefixes
            One or more model column prefixes, e.g. mpf_Sersic. Only the first
            model will have its centroid parameters copied.
        bands
            The bands to add fluxes for.
        fluxes_include
            Short column names of components whose sizes (reff) should be
            included in the matched catalog.
        sizes_include
            Short column names of components whose sizes (reff) should be
            included in the matched catalog.
        sersics_include
            Short column names of components whose Sersic index should be
            included in the matched catalog.
        is_v2
            Whether the matched catalog is a truth_summary_v2 with moment and
            bulge fraction columns.
        """
        if isinstance(model_prefixes, str):
            model_prefixes = [model_prefixes]
        elif not len(model_prefixes) > 0:
            raise ValueError(f"{model_prefixes} must have len > 0")
        if bands is None:
            bands = ["u", "g", "r", "i", "z", "y"]
        if fluxes_include is None:
            fluxes_include = {}
        if sizes_include is None:
            sizes_include = {}
        if sersics_include is None:
            sersics_include = {}
        columns_target_add = []
        for model_prefix in model_prefixes:
            self.columns_target_copy += [
                f"{model_prefix}_cen_x",
                f"{model_prefix}_cen_y",
                f"{model_prefix}_cen_x_err",
                f"{model_prefix}_cen_y_err",
            ]
            for band in bands:
                columns_target_add.append(f"{model_prefix}_{band}_flux")
                for component in fluxes_include.get(model_prefix, []):
                    columns_target_add.append(f"{model_prefix}_{component}_{band}_flux")
            for size_include in sizes_include.get(model_prefix, []):
                for ax in ("x", "y"):
                    columns_target_add.append(f"{model_prefix}_{size_include}_reff_{ax}")
                columns_target_add.append(f"{model_prefix}_{size_include}_rho")
            for sersic_include in sersics_include.get(model_prefix, []):
                columns_target_add.append(f"{model_prefix}_{sersic_include}_sersicindex")
        self.coord_format.column_target_coord1 = f"{model_prefixes[0]}_cen_ra"
        self.coord_format.column_target_coord2 = f"{model_prefixes[0]}_cen_dec"
        self.columns_ref_copy.extend([f"flux_{band}" for band in bands])
        if is_v2:
            self.columns_ref_copy.extend(
                [
                    "positionAngle",
                    "diskMajorAxisArcsec",
                    "diskAxisRatio",
                    "spheroidMajorAxisArcsec",
                    "spheroidAxisRatio",
                ]
            )
            self.columns_ref_copy.extend([f"bulge_to_total_{band}" for band in bands])
        self.columns_target_copy.extend(columns_target_add)
        self.columns_target_copy.extend([f"{col}_err" for col in columns_target_add])
        self.columns_target_coord_err = [
            col.format(model_prefix=model_prefixes[0]) for col in self.columns_target_coord_err
        ]
        self.columns_target_select_false = [f"{model_prefixes[0]}_not_primary_flag"]
        self.columns_target_select_true = []

    def finalize(
        self,
        prefix_Sersic: str | None = None,
        prefix_ExpDeV: str | None = None,
        is_v2: bool = False,
    ):
        model_prefixes = []
        fluxes_include = {}
        sizes_include = {}
        sersics_include = {}
        if prefix_Sersic:
            model_prefixes.append(prefix_Sersic)
            # TODO: get component prefix
            components = ["sersic"]
            sizes_include[prefix_Sersic] = components
            sersics_include[prefix_Sersic] = components
        if prefix_ExpDeV:
            model_prefixes.append(prefix_ExpDeV)
            components = ["exp", "deV"]
            fluxes_include[prefix_ExpDeV] = components
            sizes_include[prefix_ExpDeV] = components
        self._finalize_models(
            model_prefixes=model_prefixes,
            fluxes_include=fluxes_include,
            sizes_include=sizes_include,
            sersics_include=sersics_include,
            is_v2=is_v2,
        )

    def setDefaults(self):
        self.connections.name_input_cat_target = "objectTable_tract_multiprofit"
        self.columns_ref_copy = ["is_pointsource"]
        self.columns_target_copy = [
            "objectId",
            "patch",
        ]
        self.columns_target_coord_err = [
            "{model_prefix}_cen_ra_err",
            "{model_prefix}_cen_dec_err",
        ]


class MultiProFitDiffMatchedTractCatalogTask(DiffMatchedTractCatalogTask):
    _DefaultName = "multiProFitDiffMatchedTractCatalogTask"
    ConfigClass = MultiProFitDiffMatchedTractCatalogConfig
