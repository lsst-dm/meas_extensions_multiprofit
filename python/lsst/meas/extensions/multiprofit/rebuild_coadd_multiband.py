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

__all__ = ["ModelDRebuilder", "PatchModelDMatches", "PatchCoaddRebuilder"]

from functools import cached_property
from typing import Iterable

import astropy.table
import astropy.units as u
import lsst.gauss2d.fit as g2f
import lsst.afw.table as afwTable
import lsst.daf.butler as dafButler
import lsst.geom as geom
import numpy as np
import pydantic
from lsst.meas.extensions.scarlet.io import updateCatalogFootprints
from lsst.pipe.base import QuantumContext, QuantumGraph
from lsst.pipe.tasks.fit_coadd_multiband import (
    CoaddMultibandFitBaseTemplates,
    CoaddMultibandFitInputConnections,
    CoaddMultibandFitTask,
)
from lsst.skymap import BaseSkyMap, TractInfo

from .fit_coadd_multiband import (
    CatalogExposurePsfs,
    CatalogSourceFitterConfigData,
    MultiProFitSourceConfig,
    MultiProFitSourceTask,
)

astropy_to_geom_units = {
    u.arcmin: geom.arcminutes,
    u.arcsec: geom.arcseconds,
    u.mas: geom.milliarcseconds,
    u.deg: geom.degrees,
    u.rad: geom.radians,
}


def astropy_unit_to_geom(unit: u.Unit, default=None) -> geom.AngleUnit:
    """Convert an astropy unit to an lsst.geom unit.

    Parameters
    ----------
    unit
        The astropy unit to convert.
    default
        The default value to return if no known conversion is found.

    Returns
    -------
    unit_geom
        The equivalent unit, if found.

    Raises
    ------
    ValueError
        Raised if no equivalent unit is found.
    """
    unit_geom = astropy_to_geom_units.get(unit, default)
    if unit_geom is None:
        raise ValueError(f"{unit=} not found in {astropy_to_geom_units=}")
    return unit_geom


def find_patches(tract_info: TractInfo, ra_array, dec_array, unit: geom.AngleUnit) -> list[int]:
    """Find the patches containing a list of ra/dec values within a tract.

    Parameters
    ----------
    tract_info
        The TractInfo object for the tract.
    ra_array
        The array of right ascension values.
    dec_array
        The array of declination values (must be same length as ra_array).
    unit
        The unit of the RA/dec values.

    Returns
    -------
    patches
        A list of patches containing the specified RA/dec values.
    """
    radec = [geom.SpherePoint(ra, dec, units=unit) for ra, dec in zip(ra_array, dec_array, strict=True)]
    points = np.array([geom.Point2I(tract_info.wcs.skyToPixel(coords)) for coords in radec])
    x_list, y_list = (points[:, idx] // tract_info.patch_inner_dimensions[idx] for idx in range(2))
    patches = [tract_info.getSequentialPatchIndexFromPair((x, y)) for x, y in zip(x_list, y_list)]
    return patches


def get_radec_unit(table: astropy.table.Table, coord_ra: str, coord_dec: str, default=None):
    """Get the RA/dec units for columns in a table.

    Parameters
    ----------
    table
        The table to determine units for.
    coord_ra
        The key of the right ascension column.
    coord_dec
        The key of the declination column.
    default
        The default value to return if no unit is found.

    Returns
    -------
    unit
        The unit of the RA/dec columns or None if none is found.

    Raises
    ------
    ValueError
        Raised if the units are inconsistent.
    """
    unit_ra, unit_dec = (
        astropy_unit_to_geom(table[coord].unit, default=default) for coord in (coord_ra, coord_dec)
    )
    if unit_ra != unit_dec:
        units = {coord: table[coord].unit for coord in (coord_ra, coord_dec)}
        raise ValueError(f"Reference table has inconsistent {units=}")
    return unit_ra


class DataLoader(pydantic.BaseModelD):
    """A collection of data that can be used to rebuild models."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    catexps: list[CatalogExposurePsfs] = pydantic.Field(
        doc="List of MultiProFit catalog-exposure-psf objects used to fit PSF-convolved models",
    )
    catalog_multi: afwTable.SourceCatalog = pydantic.Field(
        doc="Patch-level multiband reference catalog (deepCoadd_ref)",
    )

    @cached_property
    def channels(self) -> tuple[g2f.Channel]:
        channels = tuple(g2f.Channel.get(catexp.band) for catexp in self.catexps)
        return channels

    @classmethod
    def from_butler(
        cls, butler: dafButler.Butler, data_id: dict[str], bands: Iterable[str], name_coadd=None, **kwargs
    ):
        """Construct a DataLoader from a Butler and dataId.

        Parameters
        ----------
        butler
            The butler to load from.
        data_id
            Key-value pairs for the {name_coadd}Coadd_* dataId.
        bands
            The list of bands to load.
        name_coadd
            The prefix of the Coadd datasettype name.
        **kwargs
            Additional keyword arguments to pass to the init method for
            `CoaddMultibandFitInputConnections`.

        Returns
        -------
        data_loader
            An initialized DataLoader.
        """
        bands = tuple(bands)
        if len(set(bands)) != len(bands):
            raise ValueError(f"{bands=} is not a set")
        if name_coadd is None:
            name_coadd = CoaddMultibandFitBaseTemplates["name_coadd"]

        catalog_multi = butler.get(
            CoaddMultibandFitInputConnections.cat_ref.name.format(name_coadd=name_coadd), **data_id, **kwargs
        )

        catexps = {}
        for band in bands:
            data_id["band"] = band
            catalog = butler.get(
                CoaddMultibandFitInputConnections.cats_meas.name.format(name_coadd=name_coadd),
                **data_id,
                **kwargs,
            )
            exposure = butler.get(
                CoaddMultibandFitInputConnections.coadds.name.format(name_coadd=name_coadd),
                **data_id,
                **kwargs,
            )
            models_scarlet = butler.get(
                CoaddMultibandFitInputConnections.models_scarlet.name.format(name_coadd=name_coadd),
                **data_id,
                **kwargs,
            )
            updateCatalogFootprints(
                modelData=models_scarlet,
                catalog=catalog,
                band=data_id["band"],
                imageForRedistribution=exposure,
                removeScarletData=True,
                updateFluxColumns=False,
            )
            # The config and table are harmless dummies
            catexps[band] = CatalogExposurePsfs(
                catalog=catalog,
                exposure=exposure,
                table_psf_fits=astropy.table.Table(),
                dataId=data_id,
                id_tract_patch=data_id["patch"],
                channel=g2f.Channel.get(band),
                config_fit=MultiProFitSourceConfig(),
            )
        return cls(
            catalog_multi=catalog_multi,
            catexps=list(catexps.values()),
        )

    def load_deblended_object(
        self,
        idx_row: int,
    ) -> list[g2f.ObservationD]:
        """Load a deblended object from catexps.

        Parameters
        ----------
        idx_row
            The index of the object to load.

        Returns
        -------
        observations
            The observations of the object (deblended if it is a child).
        """
        observations = []
        for catexp in self.catexps:
            observations.append(catexp.get_source_observation(catexp.get_catalog()[idx_row]))
        return observations


class ModelDRebuilder(DataLoader):
    """A rebuilder of MultiProFit models from their inputs and best-fit
    parameter values.
    """

    fit_results: astropy.table.Table = pydantic.Field(doc="Multiprofit model fit results")
    task_fit: MultiProFitSourceTask = pydantic.Field(doc="The task")

    @cached_property
    def config_data(self) -> CatalogSourceFitterConfigData:
        config_data = self.make_config_data()
        return config_data

    @classmethod
    def from_quantumGraph(
        cls,
        butler: dafButler.Butler,
        quantumgraph: QuantumGraph,
        dataId: dict = None,
    ):
        """Make a rebuilder from a butler and quantumgraph.

        Parameters
        ----------
        butler
            The butler that the quantumgraph was built for.
        quantumgraph
            The quantum graph file from a CoaddMultibandFitTask using the
            MultiProFitSourceTask.
        dataId
            The dataId for the fit, including skymap, tract and patch.

        Returns
        -------
        rebuilder
            A ModelDRebuilder instance initialized with the necessary kwargs.
        """
        if dataId is None:
            quantum = next(iter(quantumgraph.outputQuanta)).quantum
        else:
            quantum = None
            for node in quantumgraph.outputQuanta:
                if node.quantum.dataId.to_simple().dataId == dataId:
                    quantum = node.quantum
                    break
            if quantum is None:
                raise ValueError(
                    f"{dataId=} not found in {[x.quantum.dataId for x in quantumgraph.outputQuanta]=}"
                )
        taskDef = next(iter(quantumgraph.iterTaskGraph()))
        butlerQC = QuantumContext(butler, quantum)
        config = butler.get(f"{taskDef.label}_config")
        # I have no idea what to put for initInputs.
        # quantum.initInputs looks wrong - the values can be lists
        # quantumgraph.initInputRefs(taskDef) returns a list of DatasetRefs...
        # ... but I'm not sure how to map that to connection names?
        task: CoaddMultibandFitTask = taskDef.taskClass(config=config, initInputs={})
        if not isinstance(task, CoaddMultibandFitTask):
            raise ValueError(f"{task=} type={type(task)} !isinstance of {CoaddMultibandFitTask=}")
        task_fit: MultiProFitSourceTask = task.fit_coadd_multiband
        if not isinstance(task_fit, MultiProFitSourceTask):
            raise ValueError(f"{task_fit=} type={type(task_fit)} !isinstance of {MultiProFitSourceTask=}")
        inputRefs, outputRefs = taskDef.connections.buildDatasetRefs(quantum)
        inputs = butlerQC.get(inputRefs)
        catexps = task.build_catexps(butlerQC, inputRefs, inputs)
        catexps = [task_fit.make_CatalogExposurePsfs(catexp) for catexp in catexps]
        cat_output: astropy.table.Table = butler.get(outputRefs.cat_output, storageClass="ArrowAstropy")
        return cls(
            catexps=catexps,
            task_fit=task_fit,
            catalog_multi=inputs["cat_ref"],
            fit_results=cat_output,
        )

    def make_config_data(self):
        """Make a ConfigData object out of self's channels and fit task
        config.
        """
        config_data = CatalogSourceFitterConfigData(channels=self.channels, config=self.task_fit.config)
        return config_data

    def make_model(
        self,
        idx_row: int,
        config_data: CatalogSourceFitterConfigData = None,
        init: bool = True,
    ) -> g2f.ModelD:
        """Make a ModelD for a single row from the originally fitted catalog.

        Parameters
        ----------
        idx_row
            The index of the row to make a model for.
        config_data
            The model configuration data object.
        init
            Whether to initialize the model parameters as they would have been
            prior to fitting.

        Returns
        -------
        model
            The rebuilt model.
        """
        if config_data is None:
            config_data = self.config_data
        model = self.task_fit.get_model(
            idx_row=idx_row,
            catalog_multi=self.catalog_multi,
            catexps=self.catexps,
            config_data=config_data,
            results=self.fit_results,
            set_flux_limits=False,
        )
        if init:
            self.set_model(idx_row, config_data)
        return model

    def set_model(self, idx_row: int, config_data: CatalogSourceFitterConfigData = None) -> None:
        """Set model parameters to the best-fit values for a given row.

        Parameters
        ----------
        idx_row
            The index of the row in the fit parameter table to initialize from.
        config_data
            The model configuration data object.
        """
        if config_data is None:
            config_data = self.config_data
        row = self.fit_results[idx_row]
        prefix = config_data.config.prefix_column
        offsets = {}
        offset_cen = config_data.config.centroid_pixel_offset
        if offset_cen != 0:
            offsets[g2f.CentroidXParameterD] = -offset_cen
            offsets[g2f.CentroidYParameterD] = -offset_cen
        for key, param in config_data.parameters.items():
            param.value = row[f"{prefix}{key}"] + offsets.get(type(param), 0.0)


class PatchModelDMatches(pydantic.BaseModelD):
    """Storage for MultiProFit tables matched to a reference catalog."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    matches: astropy.table.Table | None = pydantic.Field(doc="Catalogs of matches")
    quantumgraph: QuantumGraph | None = pydantic.Field(doc="Quantum graph for fit task")
    rebuilder: DataLoader | ModelDRebuilder | None = pydantic.Field(doc="MultiProFit object model rebuilder")


class PatchCoaddRebuilder(pydantic.BaseModelD):
    """A rebuilder for patch-level coadd catalog/exposure fits."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    matches: dict[str, PatchModelDMatches] = pydantic.Field("ModelD matches by algorithm name")
    name_model_ref: str = pydantic.Field(doc="The name of the reference model in matches")
    objects: astropy.table.Table = pydantic.Field(doc="Object table")
    objects_multiprofit: astropy.table.Table | None = pydantic.Field(doc="Object table for MultiProFit fits")
    reference: astropy.table.Table = pydantic.Field(doc="Reference object table")

    skymap: str = pydantic.Field(doc="The skymap name")
    tract: int = pydantic.Field(doc="The tract index")
    patch: int = pydantic.Field(doc="The patch index")

    @classmethod
    def from_butler(
        cls,
        butler: dafButler.Butler,
        skymap: str,
        tract: int,
        patch: int,
        collection_merged: str,
        matches: dict[str, QuantumGraph | None],
        bands: Iterable[str] = None,
        name_model_ref: str = None,
        format_collection: str = "{run}",
        load_multiprofit: bool = True,
        dataset_type_ref: str = "truth_summary",
    ):
        """Construct a PatchCoaddRebuilder from a single Butler collection.

        Parameters
        ----------
        butler
            The butler to load from.
        skymap
            The skymap for the collection.
        tract
            The skymap tract id.
        patch
            The skymap patch id.
        collection_merged
            The name of the collection with the merged objectTable(s).
        matches
            A dictionary of model names with corresponding QuantumGraphs.
            These may be None but must be provided for MultiProFit model
            reconstruction to be possible.
        bands
            The list of bands to load data for.
        name_model_ref
            The name of the model to use as a reference. Must be a key in
            `matches`.
        format_collection
            A format string for the output collection(s) defined in the
            `matches` QuantumGraphs.
        load_multiprofit
            Whether to attempt to load an objectTable_tract_multiprofit.
        dataset_type_ref
            The dataset type of the reference catalog.

        Returns
        -------
        rebuilder
            The fully-configured PatchCoaddRebuilder.
        """
        if name_model_ref is None:
            for name, quantumgraph in matches.items():
                if quantumgraph is not None:
                    name_model_ref = name
                    break
        if name_model_ref is None:
            raise ValueError("Must supply name_model_ref or at least one matches with a quantumgraph")
        dataId = dict(skymap=skymap, tract=tract, patch=patch)
        objects = butler.get(
            "objectTable_tract", collections=[collection_merged], storageClass="ArrowAstropy", **dataId
        )
        objects = objects[objects["patch"] == patch]
        if load_multiprofit:
            objects_multiprofit = butler.get(
                "objectTable_tract_multiprofit",
                collections=[collection_merged],
                storageClass="ArrowAstropy",
                **dataId,
            )
            objects_multiprofit = objects_multiprofit[objects_multiprofit["patch"] == patch]
        else:
            objects_multiprofit = None
        reference = butler.get(
            dataset_type_ref, collections=[collection_merged], storageClass="ArrowAstropy", **dataId
        )
        skymap_tract = butler.get(BaseSkyMap.SKYMAP_DATASET_TYPE_NAME, skymap=skymap)[tract]
        unit_coord_ref = get_radec_unit(reference, "ra", "dec", default=geom.degrees)
        if "patch" not in reference.columns:
            patches = find_patches(skymap_tract, reference["ra"], reference["dec"], unit=unit_coord_ref)
            reference["patch"] = patches
        elif reference["patch"].dtype != int:
            # the ci_imsim truth_summary still has string patches
            index_patch = skymap_tract[patch].index
            str_patch = f"{index_patch.y},{index_patch.x}"
            reference = reference[
                (reference["patch"] == str_patch) & (reference["is_unique_truth_entry"] == True)  # noqa: E712
            ]
            del reference["patch"]
            reference["patch"] = patch
        reference = reference[reference["patch"] == patch]
        points = skymap_tract.wcs.skyToPixel(
            [geom.SpherePoint(row["ra"], row["dec"], units=geom.degrees) for row in reference]
        )
        reference["x"] = [point.x for point in points]
        reference["y"] = [point.y for point in points]
        matches_name = {}
        for name, quantumgraph in matches.items():
            is_mpf = quantumgraph is not None
            matched = butler.get(
                f"matched_{dataset_type_ref}_objectTable_tract{'_multiprofit' if is_mpf else ''}",
                collections=[
                    format_collection.format(run=quantumgraph.metadata["output"], name=name)
                    if is_mpf
                    else collection_merged
                ],
                storageClass="ArrowAstropy",
                **dataId,
            )
            # unmatched ref objects don't have a patch set
            # should probably be fixed in diff_matched
            # but need to decide priority on matched - ref first? or target?
            unit_coord_ref = get_radec_unit(
                matched,
                "refcat_ra",
                "refcat_dec",
                default=geom.degrees,
            )
            unmatched = (
                matched["patch"].mask if np.ma.is_masked(matched["patch"]) else ~(matched["patch"] >= 0)
            ) & np.isfinite(matched["refcat_ra"])
            patches_unmatched = find_patches(
                skymap_tract,
                matched["refcat_ra"][unmatched],
                matched["refcat_dec"][unmatched],
                unit=unit_coord_ref,
            )
            matched["patch"][np.where(unmatched)[0]] = patches_unmatched
            matched = matched[matched["patch"] == patch]
            rebuilder = (
                ModelDRebuilder.from_quantumGraph(butler, quantumgraph, dataId=dataId)
                if is_mpf
                else DataLoader.from_butler(
                    butler, data_id=dataId, bands=bands, collections=[collection_merged]
                )
            )
            matches_name[name] = PatchModelDMatches(
                matches=matched, quantumgraph=quantumgraph, rebuilder=rebuilder
            )
        return cls(
            matches=matches_name,
            objects=objects,
            objects_multiprofit=objects_multiprofit,
            reference=reference,
            skymap=skymap,
            tract=tract,
            patch=patch,
            name_model_ref=name_model_ref,
        )
