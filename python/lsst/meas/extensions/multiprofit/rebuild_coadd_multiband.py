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

__all__ = ["ModelRebuilder", "PatchModelMatches", "PatchCoaddRebuilder"]

from functools import cached_property

import astropy.table
import gauss2d.fit as g2f
import lsst.afw.table as afwTable
import lsst.daf.butler as dafButler
import lsst.geom as geom
from lsst.pipe.base import QuantumContext, QuantumGraph
from lsst.pipe.tasks.fit_coadd_multiband import CoaddMultibandFitTask
from lsst.skymap import BaseSkyMap
import pydantic
from pydantic.dataclasses import dataclass

from .fit_coadd_multiband import (
    CatalogExposurePsfs, CatalogSourceFitterConfigData, MultiProFitSourceTask,
)


@dataclass(frozen=True, config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class ModelRebuilder:
    """A rebuilder of MultiProFit models from their inputs and best-fit
    parameter values.
    """

    catexps: list[CatalogExposurePsfs] = pydantic.Field(doc="List of catalog-exposure-psf objects")
    catalog_multi: afwTable.SourceCatalog = pydantic.Field(
        doc="Patch-level multiband reference catalog (deepCoadd_ref)",
    )
    fit_results: astropy.table.Table = pydantic.Field(doc="Multiprofit model fit results")
    task_fit: MultiProFitSourceTask = pydantic.Field(doc="The task")

    @cached_property
    def channels(self) -> tuple[g2f.Channel]:
        channels = tuple(g2f.Channel.get(catexp.band) for catexp in self.catexps)
        return channels

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
            A ModelRebuilder instance initialized with the necessary kwargs.
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
            catexps=catexps, task_fit=task_fit, catalog_multi=inputs["cat_ref"], fit_results=cat_output,
        )

    def make_config_data(self):
        config_data = CatalogSourceFitterConfigData(channels=self.channels, config=self.task_fit.config)
        return config_data

    def make_model(
        self, idx_row: int, config_data: CatalogSourceFitterConfigData = None, init: bool = True,
    ) -> g2f.Model:
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
            param.value = row[f"{prefix}{key}"] + offsets.get(type(param), 0.)


@dataclass(frozen=True, config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class PatchModelMatches:
    """Storage for MultiProFit tables matched to a reference catalog."""

    matches: astropy.table.Table | None = pydantic.Field(doc="Catalogs of matches")
    quantumgraph: QuantumGraph | None = pydantic.Field(doc="Quantum graph for fit task")
    rebuilder: ModelRebuilder | None = pydantic.Field(doc="MultiProFit object model rebuilder")


@dataclass(frozen=True, config=pydantic.ConfigDict(arbitrary_types_allowed=True))
class PatchCoaddRebuilder:
    """A rebuilder for patch-level coadd catalog/exposure fits."""

    matches: dict[str, PatchModelMatches] = pydantic.Field("Model matches by algorithm name")
    objects: astropy.table.Table = pydantic.Field(doc="Object table")
    objects_multiprofit: astropy.table.Table = pydantic.Field(doc="Object table for MultiProFit fits")
    reference: astropy.table.Table = pydantic.Field(doc="Reference object table")
    skymap: str = pydantic.Field(doc="The skymap name")
    tract: int = pydantic.Field(doc="The tract index")
    patch: int = pydantic.Field(doc="The patch index")
    model_ref: str = pydantic.Field(doc="The name of the reference model in matches")

    @classmethod
    def from_butler(
        cls,
        butler: dafButler.Butler,
        skymap: str,
        tract: int,
        patch: int,
        collection_merged: str,
        matches: dict[str, QuantumGraph],
        model_ref: str = None,
        format_collection: str = "{run}",
    ):
        if model_ref is None:
            for name, quantumgraph in matches.items():
                if quantumgraph is not None:
                    model_ref = name
                    break
        if model_ref is None:
            raise ValueError("At least one matches with a quantumgraph must be supplied")
        dataId = dict(skymap=skymap, tract=tract, patch=patch)
        objects = butler.get("objectTable_tract", collections=[collection_merged],
                             storageClass="ArrowAstropy", **dataId)
        objects = objects[objects["patch"] == patch]
        objects_multiprofit = butler.get(
            "objectTable_tract_multiprofit", collections=[collection_merged], storageClass="ArrowAstropy",
            **dataId
        )
        objects_multiprofit = objects_multiprofit[objects_multiprofit["patch"] == patch]
        reference = butler.get(
            "truth_summary", collections=[collection_merged], storageClass="ArrowAstropy", **dataId
        )
        skymap_tract = butler.get(BaseSkyMap.SKYMAP_DATASET_TYPE_NAME, skymap=skymap)[tract]
        # the ci_imsim truth_summary still has string patches
        if reference["patch"].dtype != int:
            index_patch = skymap_tract[patch].index
            str_patch = f"{index_patch.y},{index_patch.x}"
            reference = reference[
                (reference["patch"] == str_patch) & (reference["is_unique_truth_entry"] == True)]  # noqa: E712
            del reference["patch"]
            reference["patch"] = patch
        else:
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
                f"matched_truth_summary_objectTable_tract{'_multiprofit' if is_mpf else ''}",
                collections=[
                    format_collection.format(run=quantumgraph.metadata["output"], name=name)
                    if is_mpf else collection_merged
                ],
                storageClass="ArrowAstropy",
                **dataId,
            )
            matched = matched[matched["patch"] == patch]
            rebuilder = ModelRebuilder.from_quantumGraph(
                butler, quantumgraph, dataId=dataId) if is_mpf else None
            matches_name[name] = PatchModelMatches(matches=matched, quantumgraph=quantumgraph,
                                                   rebuilder=rebuilder)
        return cls(
            matches=matches_name, objects=objects, objects_multiprofit=objects_multiprofit,
            reference=reference,
            skymap=skymap, tract=tract, patch=patch, model_ref=model_ref,
        )
