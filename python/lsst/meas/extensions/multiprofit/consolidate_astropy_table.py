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
    "ConsolidateAstropyTableConfig",
    "ConsolidateAstropyTableConfigBase",
    "ConsolidateAstropyTableConnections",
    "ConsolidateAstropyTableTask",
)

from collections import defaultdict

import astropy.table as apTab
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as connectionTypes

from .input_config import InputConfig


class ConsolidateAstropyTableConfigBase(pexConfig.Config):
    """Config for ConsolidateAstropyTableTask."""

    inputs = pexConfig.ConfigDictField(
        doc="Mapping of input dataset type config by name",
        keytype=str,
        itemtype=InputConfig,
        default={},
    )


class ConsolidateAstropyTableConnections(
    # Ignore the undocumented inherited config arg in __init__
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "skymap"),  # numpydoc ignore=PR01
):
    """Connections for ConsolidateAstropyTableTask."""

    cat_output = connectionTypes.Output(
        doc="Per-tract horizontal concatenation of the input AstropyTables",
        name="objectAstropyTable_tract",
        storageClass="ArrowTable",
        dimensions=("tract", "skymap"),
    )

    def __init__(self, *, config: ConsolidateAstropyTableConfigBase):
        super().__init__(config=config)
        for name, config_input in config.inputs.items():
            if hasattr(self, name):
                raise ValueError(
                    f"{config_input=} {name=} is invalid, due to being an existing attribute of {self=}"
                )
            connection = config_input.get_connection(name)
            setattr(self, name, connection)


class ConsolidateAstropyTableConfig(
    pipeBase.PipelineTaskConfig,
    ConsolidateAstropyTableConfigBase,
    pipelineConnections=ConsolidateAstropyTableConnections,
):
    """PipelineTaskConfig for ConsolidateAstropyTableTask."""

    drop_duplicate_columns = pexConfig.Field[bool](
        doc="Whether to drop columns from a table if they occur in a previous table."
        " If False, astropy will rename them with its default scheme.",
        default=True,
    )
    join_type = pexConfig.ChoiceField[str](
        doc="Type of join to perform in the final hstack",
        allowed={
            "inner": "Inner join",
            "outer": "Outer join",
            "exact": "Exact join",
        },
        default="exact",
        optional=False,
    )
    validate_duplicate_columns = pexConfig.Field[bool](
        doc="Whether to check that duplicate columns are identical in any table they occur in.",
        default=True,
    )


class ConsolidateAstropyTableTask(pipeBase.PipelineTask):
    """Write patch-merged astropy tables to a tract-level astropy table."""

    _DefaultName = "consolidateAstropyTable"
    ConfigClass = ConsolidateAstropyTableConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        bands_ref, patches_ref = None, None
        band_null, patch_null = "", -1
        bands_null, patches_null = {band_null}, {patch_null: None}
        data = dict()
        bands_sorted = None

        # inputRefs are usually unsorted lists so they need to be sorted first
        for name, inputRef_list in inputRefs:
            inputConfig = self.config.inputs[name]
            bands, patches = set(), dict()
            data_name = defaultdict(dict)
            inputs_name = inputs[name]

            # if it's not a list, then it's a single object
            if not hasattr(inputRef_list, "__len__"):
                inputRef_list = tuple((inputRef_list,))
                inputs_name = tuple((inputs_name,))

            # Add every ref by band (if not multiband)
            for dataRef, data_in in zip(inputRef_list, inputs_name):
                dataId = dataRef.dataId
                band = dataId.band.name if not inputConfig.is_multiband else band_null

                if inputConfig.columns is not None:
                    columns = inputConfig.columns
                    data_in = data_in.get(parameters={"columns": columns})
                else:
                    columns = tuple(data_in.columns)

                if inputConfig.storageClass == "DataFrame":
                    data_in = apTab.Table.from_pandas(data_in.reset_index(drop=False))
                elif inputConfig.storageClass == "ArrowAstropy":
                    data_in.meta = {name: data_in.meta}

                if not inputConfig.is_multiband:
                    columns_new = [
                        column if column == inputConfig.column_id else f"{band}_{column}"
                        for column in columns
                    ]
                    data_in.rename_columns(columns, columns_new)
                if inputConfig.action is not None:
                    data_in = inputConfig.action(data_in, datasetType=name)

                if inputConfig.is_multipatch:
                    patch = patch_null
                    patches[patch] = None
                else:
                    patch = dataId.patch.id
                    patches[patch] = min(data_in[inputConfig.column_id])
                data_name[patch][band] = data_in
                bands.add(band)

            # Validate the bands
            if inputConfig.is_multiband:
                if bands != bands_null:
                    raise RuntimeError(f"multiband {inputConfig=} has non-trivial {bands=}")
            else:
                if bands_ref is None:
                    bands_ref = bands
                    bands_sorted = tuple(band for band in sorted(bands_ref))
                else:
                    if bands != bands_ref:
                        raise RuntimeError(f"{inputConfig=} {bands=} != {bands_ref=}")

            # Check that every dataset has the same set of patches
            if inputConfig.is_multipatch:
                if patches != patches_null:
                    raise RuntimeError(f"{inputConfig=} {patches=} != {patches_null=}")
            else:
                column_id = inputConfig.column_id
                if patches_ref is None:
                    bands = tuple(bands) if inputConfig.is_multiband else bands_sorted
                    for patch in patches:
                        data_patch = data_name[patch]
                        # Make sure any one-time operations are done once
                        # rather than for every band
                        added = False
                        for band in bands:
                            if tab := data_patch.get(band):
                                if not added:
                                    # add a patch column to fill in later
                                    tab.add_column(np.full(len(tab), patch), name="patch", index=1)
                                    # The id column should be objectId
                                    tab.rename_column(column_id, "objectId")
                                    added = True
                                else:
                                    del tab[column_id]
                    patches_objid = {objid: patch for patch, objid in patches.items()}
                    patches_ref = {patch: objid for objid, patch in sorted(patches_objid.items())}
                elif {patch: patches[patch] for patch in patches_ref.keys()} != patches_ref:
                    raise RuntimeError(f"{inputConfig=} {patches=} != {patches_ref=}")
                else:
                    for data_patch in data_name.values():
                        for tab in data_patch.values():
                            del tab[column_id]

            data[name] = data_name

        self.log.info("Concatenating %s per-patch astropy Tables", len(patches))

        tables_read = []
        check_columns = self.config.drop_duplicate_columns or self.config.validate_duplicate_columns
        n_bands = len(bands_sorted)

        for name, data_name in data.items():
            config_input = self.config.inputs[name]
            tables = []
            bands_missing = False

            # If this is a multipatch dataset, loop over patches
            # Otherwise, loop over the single "null" patch
            for patch in patches_ref if not config_input.is_multipatch else patches_null:
                data_name_patch = data_name[patch]
                # If this is multiband, use the null band, and return an empty
                # list if there's no corresponding dataset
                if config_input.is_multiband:
                    tables_patch = data_name_patch.get(band_null, [])
                else:
                    # Get the tables (or None if it's missing) in sorted order
                    tables_patch = [
                        _tab for band in bands_sorted if (_tab := data_name_patch.get(band)) is not None
                    ]
                    # Check if any bands are missing
                    if not bands_missing and (len(tables_patch) != n_bands):
                        bands_missing = True
                # Join only if there's something to join
                if tables_patch:
                    table_patch = apTab.hstack(tables_patch, join_type="exact")
                    tables.append(table_patch)
                # If there's nothing to join, presumably the task failed
                # stacking should handle some tasks failing but not others, but
                # this shouldn't be relied upon

            table_new = (
                tables[0]
                if (len(tables) == 1)
                else apTab.vstack(tables, join_type="outer" if bands_missing else "exact")
            )

            if check_columns:
                columns_new = set(x for x in table_new.colnames if x != config_input.join_column)
                for name_previous in tables_read:
                    table_old = data[name_previous]
                    columns_common = columns_new.intersection(
                        x for x in table_old.colnames if x != self.config.inputs[name_previous].join_column
                    )
                    for column_common in columns_common:
                        if self.config.validate_duplicate_columns:
                            if not np.array_equal(
                                table_new[column_common],
                                table_old[column_common],
                                equal_nan=True,
                            ):
                                raise RuntimeError(
                                    f"Joined table column={column_common} differs between {name} and"
                                    f" {name_previous} tables"
                                )
                        if self.config.drop_duplicate_columns:
                            del table_new[column_common]

            data[name] = table_new
            tables_read.append(name)

        # This will break if all tables have config.join_column
        # ... but that seems unlikely.
        table = apTab.hstack(
            [data[name] for name, config in self.config.inputs.items() if config.join_column is None],
            join_type=self.config.join_type,
        )
        for name, config in self.config.inputs.items():
            if config.join_column:
                table = apTab.join(
                    table,
                    data[name],
                    join_type=self.config.join_type,
                    keys=config.join_column,
                )

        butlerQC.put(pipeBase.Struct(cat_output=table), outputRefs)
