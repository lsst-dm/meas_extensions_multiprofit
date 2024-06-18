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


from abc import ABC, abstractmethod
from typing import Any, Iterable, Self, Type

import astropy.table
import astropy.units as u
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pydantic
from lsst.multiprofit.plots import bands_weights_lsst, plot_model_rgb

from .rebuild_coadd_multiband import DataLoader, PatchCoaddRebuilder

__all__ = [
    "ObjectTableBase",
    "TruthSummaryTable",
    "ObjectTable",
    "ObjectTableCModelD",
    "ObjectTableMultiProFit",
    "ObjectTablePsf",
    "downselect_table",
    "downselect_table_axis",
    "plot_blend",
    "plot_objects",
]

Figure = matplotlib.figure.Figure
Axes = matplotlib.axes.Axes | Iterable[matplotlib.axes.Axes]
FigureAxes = tuple[Figure, Axes]


class ObjectTableBase(ABC, pydantic.BaseModelD):
    """Base class for retrieving columns from tract-based object tables."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)

    table: astropy.table.Table = pydantic.Field(doc="The object table")

    @abstractmethod
    def get_flux(self, band: str) -> np.ndarray:
        """Return the flux in a given band.

        Parameters
        ----------
        band
            The name of the band.

        Returns
        -------
        flux
            The configured flux in that band.
        """

    @abstractmethod
    def get_id(self) -> np.ndarray:
        """Return a unique source id."""

    @abstractmethod
    def get_is_extended(self) -> np.ndarray:
        """Return if the source is extended."""

    @abstractmethod
    def get_is_variable(self) -> np.ndarray:
        """Return if the source is variable."""

    @abstractmethod
    def get_x(self) -> np.ndarray:
        """Return the x pixel coordinates."""

    @abstractmethod
    def get_y(self) -> np.ndarray:
        """Return the y pixel coordinates."""

    def make_subset(self, subset) -> Self:
        """Make a new table of the same type as self with a subset of rows.

        Parameters
        ----------
        subset
            An array that can be used to select asubset of the rows in
            self.table.

        Returns
        -------
        table
            An object of the same type as self with a subsetted table.
            The table will be a copy, as it does not appear to be possible
            to return views of slices of astropy Table instances.
        """
        kwargs_table = {name: getattr(self, name) for name in self.model_fields if name != "table"}
        return type(self)(table=self.table[subset], **kwargs_table)


class TruthSummaryTable(ObjectTableBase):
    """Class for retrieving columns from DC2 truth tables."""

    def get_flux(self, band: str) -> np.ndarray:
        return self.table[f"flux_{band}"]

    def get_id(self) -> np.ndarray:
        return self.table["id"]

    def get_is_extended(self) -> np.ndarray:
        return self.table["is_pointsource"] == False  # noqa: E712

    def get_is_variable(self) -> np.ndarray:
        return self.table["is_variable"] == True  # noqa: E712

    def get_x(self):
        return self.table["x"]

    def get_y(self):
        return self.table["y"]


class ObjectTable(ObjectTableBase, ABC):
    """Base class for objectTable_tract."""

    def get_id(self) -> np.ndarray:
        return self.table["objectId"]

    def get_is_extended(self) -> np.ndarray:
        return self.table["refExtendedness"] >= 0.5

    def get_is_variable(self) -> np.ndarray:
        return np.zeros(len(self.table), dtype=bool)

    def get_x(self):
        return self.table["x"]

    def get_y(self):
        return self.table["y"]


class ObjectTableCModelD(ObjectTable):
    """Class for retrieving CModelD fluxes from objectTable_tract."""

    def get_flux(self, band: str) -> np.ndarray:
        return self.table[f"{band}_cModelDFlux"]


class ObjectTableMultiProFit(ObjectTableBase):
    """Class for retrieving fluxes from objectTable_tract_multiprofit."""

    name_model: str = pydantic.Field(doc="The name of the MultiProFit model")
    prefix_col: str = pydantic.Field(doc="The prefix for object fit columns", default="mpf_")

    def get_flux(self, band: str) -> np.ndarray:
        return self.table[f"{self.prefix_col}{self.name_model}_{band}_flux"]

    def get_id(self) -> np.ndarray:
        return self.table["objectId"]

    def get_is_extended(self) -> np.ndarray:
        return self.table["refExtendedness"] >= 0.5

    def get_is_variable(self) -> np.ndarray:
        return np.zeros(len(self.table), dtype=bool)

    def get_x(self):
        return self.table[f"{self.prefix_col}{self.name_model}_cen_x"]

    def get_y(self):
        return self.table[f"{self.prefix_col}{self.name_model}_cen_y"]


class ObjectTablePsf(ObjectTable):
    """Class for retreiving PSF fluxes from objectTable_tract."""

    def get_flux(self, band: str) -> np.ndarray:
        return self.table[f"{band}_psfFlux"]


def downselect_table(
    table: ObjectTableBase,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> ObjectTableBase:
    """Select points from a table within an x,y extent.

    Parameters
    ----------
    table
        The table to downselect.
    x_min
        The minimum x value.
    x_max
        The maximum x value.
    y_min
        The minimum y value.
    y_max
        The maximum y value.

    Returns
    -------
    table
        A downselected table of the same class.
    """
    x_all = table.get_x()
    y_all = table.get_y()
    within = (x_all > x_min) & (x_all < x_max) & (y_all > y_min) & (y_all < y_max)
    return table.make_subset(within)


def downselect_table_axis(table: ObjectTableBase, axis) -> ObjectTableBase:
    """Select points from a table within a figure axis.

    Parameters
    ----------
    table
        The table to downselect.
    axis
        The figure axis to determine the extent from.

    Returns
    -------
    table
        A downselected table of the same class.
    """
    extent = np.array(axis.axis())
    return downselect_table(table, extent[0], extent[1], extent[2], extent[3])


def plot_objects(
    table: ObjectTableBase,
    axes: Axes,
    bands: Iterable[str],
    table_downselected: bool = False,
    kwargs_annotate: dict[str, Any] = None,
    kwargs_scatter: dict[str, Any] = None,
    labels_extended: tuple[str, str] = ("S", "G"),
) -> Axes:
    """Plot catalog objects on an existing image.

    Parameters
    ----------
    table
        The object table to plot source from.
    axes
        The figure axes to plot on.
    bands
        The bands to sum over fluxes to derive a total mag label.
    table_downselected
        Whether the table has already been downselected to contain only
        points within the bounds of the axes.
    kwargs_annotate
        Keyword arguments to pass to axes.annotate.
    kwargs_scatter
        Keyword arguments to pass to axes.scatter.
    labels_extended
        Label prefixes for non-extended and extended objects, respectively.

    Returns
    -------
    axes
        The input axes with added points and labels.
    """
    if kwargs_annotate is None:
        kwargs_annotate = dict(color="white", fontsize=14, ha="left", va="bottom")
    if kwargs_scatter is None:
        kwargs_scatter = dict(c="white", marker="+", s=100)
    table_within = table if table_downselected else downselect_table_axis(table, axes)
    x = table_within.get_x()
    y = table_within.get_y()
    axes.scatter(x, y, **kwargs_scatter)
    fluxes = [table_within.get_flux(band) for band in bands]
    is_extended = table_within.get_is_extended()
    is_variable = table_within.get_is_variable()

    for idx in range(len(table_within.table)):
        mag = u.nanojansky.to(u.ABmag, np.sum([fluxcol[idx] for fluxcol in fluxes]))
        type_src = f"{'V' if is_variable[idx] else ''}{labels_extended[1 if is_extended[idx] else 0]}"
        axes.annotate(f"{type_src}{mag:.1f}", (x[idx], y[idx]), **kwargs_annotate)

    return axes


def plot_blend(
    rebuilder: PatchCoaddRebuilder,
    idx_row_parent: int,
    weights: dict[str, float] = None,
    table_ref_type: Type = TruthSummaryTable,
    kwargs_plot_parent: dict[str, Any] = None,
    kwargs_plot_children: dict[str, Any] = None,
) -> tuple[Figure, Axes, Figure, Axes]:
    """Plot an image of an entire blend and its deblended children.

    Parameters
    ----------
    rebuilder
        The patch rebuilder to plot from.
    idx_row_parent
        The row index of the parent object in the reference SourceCatalog.
    weights
        Multiplicative weights by band name for RGB plots.
    table_ref_type
        The type of reference table to construct when downselecting.
    kwargs_plot_parent
        Keyword arguments to pass to make RGB plots of the parent blend.
    kwargs_plot_children
        Keyword arguments to pass to make RGB plots of deblended children.

    Returns
    -------
    fig_rgb
        The Figure for the RGB plots of the parent.
    ax_rgb
        The Axes for the RGB plots of the parent.
    fig_gs
        The Figure for the grayscale plots of the parent.
    ax_gs
        The Axes for the grayscale plots of the parent.
    """
    if kwargs_plot_parent is None:
        kwargs_plot_parent = {}
    if kwargs_plot_children is None:
        kwargs_plot_children = {}
    if weights is None:
        weights = bands_weights_lsst

    plot_chi_hist = kwargs_plot_children.pop("plot_chi_hist", True)
    rebuilder_ref = rebuilder.matches[rebuilder.name_model_ref].rebuilder
    observations = {
        catexp.band: catexp.get_source_observation(catexp.get_catalog()[idx_row_parent], skip_flags=True)
        for catexp in rebuilder_ref.catexps
    }

    fig_rgb, ax_rgb, fig_gs, ax_gs, *_ = plot_model_rgb(
        model=None, weights=weights, observations=observations, plot_singleband=False, plot_chi_hist=False,
        **kwargs_plot_parent
    )
    table_within_ref = downselect_table_axis(table_ref_type(table=rebuilder.reference), ax_rgb)
    plot_objects(table_within_ref, ax_rgb, weights, table_downselected=True)

    objects_primary = rebuilder.objects[rebuilder.objects["detect_isPrimary"] == True]  # noqa: E712
    kwargs_annotate_obs = dict(color="white", fontsize=14, ha="right", va="top")
    kwargs_scatter_obs = dict(c="white", marker="x", s=70)
    table_within_cmodel = downselect_table_axis(ObjectTableCModelD(table=objects_primary), ax_rgb)
    labels_extended_model = ("C", "E")
    plot_objects(
        table_within_cmodel,
        ax_rgb,
        weights,
        table_downselected=True,
        kwargs_annotate=kwargs_annotate_obs,
        kwargs_scatter=kwargs_scatter_obs,
        labels_extended=labels_extended_model,
    )
    plt.show()

    objects_mpf = rebuilder.objects_multiprofit
    objects_mpf_within = {}
    for name, matched in rebuilder.matches.items():
        if matched.rebuilder and objects_mpf:
            objects_mpf_within[name] = downselect_table_axis(
                ObjectTableMultiProFit(name_model=name, table=objects_mpf),
                ax_rgb,
            )

    cat_ref = rebuilder_ref.catalog_multi
    row_parent = cat_ref[idx_row_parent]
    idx_children = (idx_row_parent,) if (row_parent["parent"] == 0) else (
        np.where(rebuilder_ref.catalog_multi["parent"] == row_parent["id"])[0]
    )

    for idx_child in idx_children:
        for name, matched in rebuilder.matches.items():
            print(f"ModelD: {name}")
            rebuilder_child = matched.rebuilder
            is_dataloader = isinstance(rebuilder_child, DataLoader)
            is_scarlet = is_dataloader and (name == "scarlet")
            if is_scarlet or rebuilder_child:
                try:
                    if is_dataloader:
                        model = None
                        observations = rebuilder_child.load_deblended_object(idx_child)
                    else:
                        model = rebuilder_child.make_model(idx_child)
                        observations = None

                    _, ax_rgb_c, *_ = plot_model_rgb(
                        model=model, weights=weights, plot_singleband=False,
                        plot_chi_hist=(not is_dataloader) and plot_chi_hist,
                        observations=observations,
                        **kwargs_plot_children
                    )
                    ax_rgb_c0 = ax_rgb_c[0][0]
                    plot_objects(table_within_ref, ax_rgb_c0, weights)
                    tab_mpf = objects_mpf_within.get(name)
                    if tab_mpf:
                        plot_objects(
                            tab_mpf,
                            ax_rgb_c0,
                            weights,
                            kwargs_annotate=kwargs_annotate_obs,
                            kwargs_scatter=kwargs_scatter_obs,
                            labels_extended=labels_extended_model,
                        )
                    plt.show()
                except Exception as exc:
                    print(f"{idx_child=} failed to rebuild due to {exc}")

    return fig_rgb, ax_rgb, fig_gs, ax_gs
