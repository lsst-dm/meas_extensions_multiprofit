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

import logging
import math
from functools import cached_property
from typing import Any, Iterable, Mapping, Sequence

import lsst.gauss2d as g2
import lsst.gauss2d.fit as g2f
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.fit_coadd_multiband as fitMB
import lsst.utils.timer as utilsTimer
import numpy as np
import pydantic
from astropy.table import Table
from lsst.daf.butler.formatters.parquet import astropy_to_arrow
from lsst.multiprofit.config import set_config_from_dict
from lsst.multiprofit.errors import NoDataError, PsfRebuildFitFlagError
from lsst.multiprofit.fit_psf import CatalogPsfFitterConfig, CatalogPsfFitterConfigData
from lsst.multiprofit.fit_source import (
    CatalogExposureSourcesABC,
    CatalogSourceFitterABC,
    CatalogSourceFitterConfig,
    CatalogSourceFitterConfigData,
)
from lsst.multiprofit.utils import get_params_uniq
from lsst.pex.config.configurableActions import ConfigurableAction, ConfigurableActionField
from pydantic.dataclasses import dataclass

from .errors import IsParentError, NotPrimaryError
from .utils import get_spanned_image

TWO_SQRT_PI = 2 * math.sqrt(np.pi)


class PsfFitSuccessActionBase(ConfigurableAction):
    """Base action to return whether a source had a succesful PSF fit."""

    def get_schema(self) -> list[str]:
        """Return the list of columns required to call this action."""
        raise NotImplementedError("This method must be overloaded in subclasses")

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError("This method must be overloaded in subclasses")


class PsfComponentsActionBase(ConfigurableAction):
    """Base action to return a list of Gaussians from a source mapping."""

    def get_schema(self) -> list[str]:
        """Return the list of columns required to call this action."""
        raise NotImplementedError("This method must be overloaded in subclasses")

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> list[g2.Gaussian]:
        raise NotImplementedError("This method must be overloaded in subclasses")


class SourceTablePsfFitSuccessAction(PsfFitSuccessActionBase):
    """Action to return PSF fit status from a SourceTable row."""

    flag_format = pexConfig.Field[str](
        doc="Format for the flag field; flag_prefix, flag_suffix and flag_sub are substituted",
        default="{flag_prefix}{flag_suffix}{flag_sub}",
    )
    flag_prefix = pexConfig.Field[str](
        doc="Prefix for the key for the summed flag field",
        default="modelfit_DoubleShapeletPsfApprox",
    )
    flag_suffix = pexConfig.Field[str](
        doc="Suffix for all flag fields",
        default="_flag",
    )
    flags_sub = pexConfig.ListField[str](
        doc="Suffixes for specific flag fields that must not be true",
        default=["_invalidPointForPsf", "_invalidMoments", "_maxIterations"],
    )

    def _format(self, flag_sub: str) -> str:
        return self.flag_format.format(
            flag_prefix=self.flag_prefix,
            flag_sub=flag_sub,
            flag_suffix=self.flag_suffix,
        )

    def get_schema(self) -> Iterable[str]:
        for flag_sub in self.flags_sub:
            yield self._format(flag_sub=flag_sub)

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> bool:
        good = True
        for flag_sub in self.flags_sub:
            good &= not source[self._format(flag_sub=flag_sub)]
        return good


class SourceTablePsfComponentsAction(PsfComponentsActionBase):
    """Action to return PSF components from a SourceTable.

    This is anticipated to be a deepCoadd_meas with PSF fit parameters from a
    measurement plugin returning covariance matrix terms.
    """

    action_source = ConfigurableActionField[PsfFitSuccessActionBase](
        doc="Action to return whether the PSF fit was successful for a single source row",
        default=SourceTablePsfFitSuccessAction,
    )
    format = pexConfig.Field[str](
        doc="Format for the field names, where {idx_comp} is the index of the component and {moment}"
        "is the name of the moment (xx, xy or yy, integral)",
        default="modelfit_DoubleShapeletPsfApprox_{idx_comp}_{moment}",
    )
    name_moment_xx = pexConfig.Field[str](doc="Name of the xx (2nd x-axis) moment", default="xx")
    name_moment_xy = pexConfig.Field[str](doc="Name of the xy (covariance term) moment", default="xy")
    name_moment_yy = pexConfig.Field[str](doc="Name of the yy (2nd y-axis) moment", default="yy")
    name_moment_integral = pexConfig.Field[str](doc="Name of the integral (zeroth) moment", default="0")
    n_components = pexConfig.Field[int](
        doc="Number of Gaussian components",
        default=2,
        check=lambda x: x >= 2,
    )

    def get_integral(self, moment_integral) -> float:
        return moment_integral * TWO_SQRT_PI

    def get_schema(self) -> list[str]:
        names_moments = (
            self.name_moment_xx,
            self.name_moment_yy,
            self.name_moment_xy,
            self.name_moment_integral,
        )
        columns = [
            column
            for idx_comp in range(self.n_components)
            for column in (
                self.format.format(name_moment=name_moment, idx_comp=idx_comp)
                for name_moment in names_moments
            )
        ] + self.action_source.get_schema()
        return columns

    def __call__(self, source: Mapping[str, Any], *args: Any, **kwargs: Any) -> list[g2.Gaussian]:
        if not self.action_source(source):
            raise PsfRebuildFitFlagError(
                f"PSF fit failed due to action based on schema: {self.action_source.get_schema()}"
            )
        gaussians = [None] * self.n_components
        for idx_comp in range(self.n_components):
            gaussian = g2.Gaussian(
                ellipse=g2.Ellipse(
                    g2.Covariance(
                        sigma_x_sq=source[self.format.format(moment=self.name_moment_xx, idx_comp=idx_comp)],
                        sigma_y_sq=source[self.format.format(moment=self.name_moment_yy, idx_comp=idx_comp)],
                        cov_xy=source[self.format.format(moment=self.name_moment_xy, idx_comp=idx_comp)],
                    )
                ),
                integral=g2.GaussianIntegralValue(
                    value=self.get_integral(
                        source[self.format.format(moment=self.name_moment_integral, idx_comp=idx_comp)]
                    )
                ),
            )
            gaussians[idx_comp] = gaussian
        return gaussians


class MultiProFitSourceConfig(CatalogSourceFitterConfig, fitMB.CoaddMultibandFitSubConfig):
    """Configuration for the MultiProFit profile fitter."""

    action_psf = ConfigurableActionField[PsfComponentsActionBase](
        doc="The action to return PSF component values from catalogs, if implemented",
        default=None,
    )
    bands_fit = pexConfig.ListField(
        dtype=str,
        default=[],
        doc="list of bandpass filters to fit",
        listCheck=lambda x: (len(x) > 0) and (len(set(x)) == len(x)),
    )
    mask_names_zero = pexConfig.ListField[str](
        doc="Mask bits to mask out",
        default=["BAD", "EDGE", "SAT", "NO_DATA"],
    )
    psf_sigma_subtract = pexConfig.Field[float](
        doc="PSF x/y sigma value to subtract in quadrature from best-fit values",
        default=0.1,
        check=lambda x: np.isfinite(x) and (x >= 0),
    )
    prefix_column = pexConfig.Field[str](default="mpf_", doc="Column name prefix")

    def bands_read_only(self) -> set[str]:
        # TODO: Re-implement determination of prior-only bands once
        # data-driven priors are re-implemented (DM-4xxxx)
        return set()

    def requires_psf(self):
        return type(self.action_psf) is PsfComponentsActionBase

    def setDefaults(self):
        super().setDefaults()
        self.flag_errors = {
            IsParentError.column_name(): "IsParentError",
            NoDataError.column_name(): "NoDataError",
            NotPrimaryError.column_name(): "NotPrimaryError",
            PsfRebuildFitFlagError.column_name(): "PsfRebuildFitFlagError",
        }
        self.centroid_pixel_offset = -0.5


@dataclass(frozen=True, kw_only=True, config=fitMB.CatalogExposureConfig)
class CatalogExposurePsfs(fitMB.CatalogExposureInputs, CatalogExposureSourcesABC):
    """Input data from lsst pipelines, parsed for MultiProFit."""

    channel: g2f.Channel = pydantic.Field(title="Channel for the image's band")
    config_fit: MultiProFitSourceConfig = pydantic.Field(title="Config for fitting options")

    @cached_property
    def _psf_flux_params(self) -> tuple[list[g2f.ParameterD], bool]:
        psf_model = self.psf_model_data.psf_model
        n_comps = len(psf_model.components)
        params_flux = [None] * n_comps
        is_frac = [False] * n_comps
        for idx_comp, comp in enumerate(psf_model.components):
            # TODO: Change to comp.integralmodel when DM-44344 is fixed
            # integralmodels will still need to be handled differently
            params_all = get_params_uniq(comp)
            params_frac = [param for param in params_all if isinstance(param, g2f.ProperFractionParameterD)]
            if params_frac:
                is_last = idx_comp == (n_comps - 1)
                if len(params_frac) != (idx_comp + 1 - is_last):
                    raise RuntimeError(
                        f"Got unexpected {params_frac=} for"
                        f" {self.psf_model_data.psf_model.components[idx_comp]=} ({idx_comp=});"
                        f" len should be idx_comp+1"
                    )
                params_flux[idx_comp] = None if is_last else params_frac[idx_comp]
                is_frac[idx_comp] = True
            else:
                params_integral = [param for param in params_all if isinstance(param, g2f.IntegralParameterD)]
                if len(params_integral != 1):
                    raise RuntimeError(
                        f"Got unexpected {params_integral=} != 1 for"
                        f" {self.psf_model_data.psf_model.components[idx_comp]=} ({idx_comp=})"
                    )
                params_flux[idx_comp] = params_integral[0]
        is_frac_any = any(is_frac)
        if is_frac_any and not all(is_frac):
            # TODO: This should work by iterating through componentgroups
            # But that's not trivial or supported now
            raise RuntimeError("Got PSF model with a mix of fractional and linear models; cannot initialize")

        return params_flux, is_frac_any

    def get_psf_model(self, source):
        psf_model = self.psf_model_data.psf_model
        # PsfComponentsActionBase is an abstract class, so check if the action
        # is a subclass that needs to be called
        if not self.config_fit.requires_psf():
            try:
                gaussians = self.config_fit.action_psf(source)
            except PsfRebuildFitFlagError:
                return None
            n_comps = len(psf_model.components)
            fluxes = [0.0] * n_comps
            params_flux, is_frac = self._psf_flux_params
            for idx_comp, (comp, gaussian) in enumerate(zip(psf_model.components, gaussians)):
                ellipse_out = comp.ellipse
                ellipse_in = gaussian.ellipse
                ellipse_out.sigma_x = ellipse_in.sigma_x
                ellipse_out.sigma_y = ellipse_in.sigma_y
                ellipse_out.rho = ellipse_in.rho
                fluxes[idx_comp] = gaussian.integral.value
            flux_total = sum(fluxes)
            if is_frac:
                flux_remaining = 1.0
                for flux, param_frac in zip(fluxes, params_flux[:-1]):
                    flux_component = flux / flux_total
                    param_frac.value = flux_component / flux_remaining
                    flux_remaining -= flux_component
            else:
                for flux, param_flux in zip(fluxes, params_flux):
                    param_flux.value = flux / flux_total
        else:
            # TODO: this should probably use .index or something
            match = np.argwhere(
                self.table_psf_fits[self.psf_model_data.config.column_id] == source[self.config_fit.column_id]
            )[0][0]
            psf_model = self.psf_model_data.psf_model
            self.psf_model_data.init_psf_model(self.table_psf_fits[match])

        sigma_subtract = self.config_fit.psf_sigma_subtract
        if sigma_subtract > 0:
            sigma_subtract_sq = sigma_subtract * sigma_subtract
            for param in self.psf_model_data.parameters.values():
                if isinstance(
                    param,
                    g2f.SigmaXParameterD | g2f.SigmaYParameterD | g2f.ReffXParameterD | g2f.ReffYParameterD,
                ):
                    param.value = math.sqrt(param.value**2 - sigma_subtract_sq)
        return psf_model

    def get_source_observation(self, source, **kwargs) -> g2f.ObservationD:
        if not kwargs.get("skip_flags"):
            if (not source["detect_isPrimary"]) or source["merge_peak_sky"]:
                raise NotPrimaryError(f"source {source[self.config_fit.column_id]} has invalid flags for fit")
        footprint = source.getFootprint()
        bbox = footprint.getBBox()
        if not (bbox.getArea() > 0):
            return None
        bitmask = 0
        mask = self.exposure.mask[bbox]
        spans = footprint.spans.asArray()
        for bitname in self.config_fit.mask_names_zero:
            bitval = mask.getPlaneBitMask(bitname)
            bitmask |= bitval
        mask = ((mask.array & bitmask) != 0) & (spans != 0)
        mask = ~mask

        is_deblended_child = source["parent"] != 0

        img, _, sigma_inv = get_spanned_image(
            exposure=self.exposure,
            footprint=footprint if is_deblended_child else None,
            bbox=bbox,
            spans=spans,
            get_sig_inv=True,
        )
        x_min_bbox, y_min_bbox = bbox.beginX, bbox.beginY
        # Crop to tighter box for deblended model if edges are unusable
        # ... this rarely ever seems to happen though
        if is_deblended_child:
            coords = np.argwhere(np.isfinite(img) & (sigma_inv > 0) & np.isfinite(sigma_inv))
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            x_max += 1
            y_max += 1

            if (x_min > 0) or (y_min > 0) or (x_max < img.shape[0]) or (y_max < img.shape[1]):
                # Ensure the nominal centroid is still inside the box
                # ... although it's a bad sign if that row/column is all bad
                x_cen = source["slot_Centroid_x"] - x_min_bbox
                y_cen = source["slot_Centroid_y"] - y_min_bbox
                x_min = min(x_min, int(np.floor(x_cen)))
                x_max = max(x_max, int(np.ceil(x_cen)))
                y_min = min(y_min, int(np.floor(y_cen)))
                y_max = max(y_max, int(np.ceil(y_cen)))
                x_min_bbox += x_min
                y_min_bbox += y_min
                img = img[x_min:x_max, y_min:y_max]
                sigma_inv = sigma_inv[x_min:x_max, y_min:y_max]
                mask = mask[x_min:x_max, y_min:y_max]

        sigma_inv[~mask] = 0

        coordsys = g2.CoordinateSystem(1.0, 1.0, x_min_bbox, y_min_bbox)

        obs = g2f.ObservationD(
            image=g2.ImageD(img, coordsys),
            sigma_inv=g2.ImageD(sigma_inv, coordsys),
            mask_inv=g2.ImageB(mask, coordsys),
            channel=self.channel,
        )
        return obs

    def __post_init__(self):
        # TODO: Can/should this be the derived type (MultiProFitPsfConfig)?
        config = CatalogPsfFitterConfig()
        config_dict = self.table_psf_fits.meta.get("config")
        if config_dict:
            set_config_from_dict(config, config_dict)
        else:
            # TODO: How should this be set?
            # If using external PSF fits, it needs to be configured normally
            pass
        config_data = CatalogPsfFitterConfigData(config=config)
        object.__setattr__(self, "psf_model_data", config_data)


class MultiProFitSourceTask(CatalogSourceFitterABC, fitMB.CoaddMultibandFitSubTask):
    """Run MultiProFit on Exposure/SourceCatalog pairs in multiple bands.

    This task uses MultiProFit to fit a single model to all sources in a coadd,
    using a previously-fit PSF model for each exposure. The task may also use
    prior measurements from single- or merged multiband catalogs for
    initialization.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to CoaddMultibandFitSubTask.__init__.

    Notes
    -----
    See https://github.com/lsst-dm/multiprofit for more MultiProFit info.
    """

    ConfigClass = MultiProFitSourceConfig
    _DefaultName = "multiProFitSource"

    def __init__(self, **kwargs: Any):
        errors_expected = {} if "errors_expected" not in kwargs else kwargs.pop("errors_expected")
        # This ensures that all of these errors are caught as expected, but...
        # It means you can't turn them off even when specifying errors_expected
        # If there is a use case to disable any of them, this can be changed
        for error_catalog in (IsParentError, NoDataError, NotPrimaryError, PsfRebuildFitFlagError):
            if error_catalog not in errors_expected:
                errors_expected[error_catalog] = error_catalog.column_name()
        CatalogSourceFitterABC.__init__(self, errors_expected=errors_expected)
        fitMB.CoaddMultibandFitSubTask.__init__(self, **kwargs)

    def copy_centroid_errors(
        self,
        columns_cenx_err_copy: tuple[str],
        columns_ceny_err_copy: tuple[str],
        results: Table,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
    ):
        for column in columns_cenx_err_copy:
            results[column] = catalog_multi["slot_Centroid_xErr"]
        for column in columns_ceny_err_copy:
            results[column] = catalog_multi["slot_Centroid_yErr"]

    def get_model_radec(self, source: Mapping[str, Any], cen_x: float, cen_y: float):
        # no extra conversions are needed here - cen_x, cen_y are in catalog
        # coordinates already
        ra, dec = self.catexps[0].exposure.wcs.pixelToSky(cen_x, cen_y)
        return ra.asDegrees(), dec.asDegrees()

    def initialize_model(
        self,
        model: g2f.ModelD,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        values_init: Mapping[g2f.ParameterD, float] | None = None,
        centroid_pixel_offset: float = 0,
        **kwargs,
    ):
        if values_init is None:
            values_init = {}
        set_flux_limits = kwargs.pop("set_flux_limits") if "set_flux_limits" in kwargs else True
        if kwargs:
            raise ValueError(f"Unexpected {kwargs=}")
        sig_x = math.sqrt(source["slot_Shape_xx"])
        sig_y = math.sqrt(source["slot_Shape_yy"])
        # TODO: Verify if there's a sign difference here
        rho = np.clip(source["slot_Shape_xy"] / (sig_x * sig_y), -0.5, 0.5)
        if not np.isfinite(rho):
            sig_x, sig_y, rho = 0.5, 0.5, 0

        # Make restrictive centroid limits (intersection, not union)
        x_min, y_min, x_max, y_max = -np.Inf, -np.Inf, np.Inf, np.Inf

        fluxes_init = []
        fluxes_limits = []

        n_observations = len(model.data)
        n_components = len(model.sources[0].components)

        for idx_obs, observation in enumerate(model.data):
            coordsys = observation.image.coordsys
            catexp = catexps[idx_obs]
            band = catexp.band

            x_min = max(x_min, coordsys.x_min)
            y_min = max(y_min, coordsys.y_min)
            x_max = min(x_max, coordsys.x_min + float(observation.image.n_cols))
            y_max = min(y_max, coordsys.y_min + float(observation.image.n_rows))

            flux_total = np.nansum(observation.image.data[observation.mask_inv.data])

            column_ref = f"merge_measurement_{band}"
            if column_ref in source.schema.getNames() and source[column_ref]:
                row = source
            else:
                row = catexp.catalog.find(source["id"])

            if not row["base_SdssShape_flag"]:
                flux_init = row["base_SdssShape_instFlux"]
            else:
                flux_init = row["slot_GaussianFlux_instFlux"]
                if not (flux_init > 0):
                    flux_init = row["slot_PsfFlux_instFlux"]

            calib = catexp.exposure.photoCalib
            flux_init = calib.instFluxToNanojansky(flux_init) if (flux_init > 0) else max(flux_total, 1.0)
            if set_flux_limits:
                flux_max = 10 * max((flux_init, flux_total))
                flux_min = min(1e-12, flux_max / 1000)
            else:
                flux_min, flux_max = 0, np.Inf
            fluxes_init.append(flux_init / n_components)
            fluxes_limits.append((flux_min, flux_max))

        try:
            cen_x, cen_y = (
                source["slot_Centroid_x"] - centroid_pixel_offset,
                source["slot_Centroid_y"] - centroid_pixel_offset,
            )
        # TODO: determine which exceptions can occur above
        except Exception:
            # TODO: Add bbox coords or remove
            cen_x = observation.image.n_cols / 2.0
            cen_y = observation.image.n_rows / 2.0

        # An R_eff larger than the box size is problematic. This should also
        # stop unreasonable size proposals; a log10 transform isn't enough.
        # TODO: Try logit for r_eff?
        size_major = g2.EllipseMajor(g2.Ellipse(sigma_x=sig_x, sigma_y=sig_y, rho=rho)).r_major
        limits_size = max(5.0 * size_major, 2.0 * np.hypot(x_max - x_min, y_max - y_min))
        limits_xy = (1e-5, limits_size)
        params_limits_init = {
            g2f.CentroidXParameterD: (cen_x, (x_min, x_max)),
            g2f.CentroidYParameterD: (cen_y, (y_min, y_max)),
            g2f.ReffXParameterD: (sig_x, limits_xy),
            g2f.ReffYParameterD: (sig_y, limits_xy),
            g2f.SigmaXParameterD: (sig_x, limits_xy),
            g2f.SigmaYParameterD: (sig_y, limits_xy),
            g2f.RhoParameterD: (rho, None),
            # TODO: get guess from configs?
            g2f.SersicMixComponentIndexParameterD: (1.0, None),
        }

        # TODO: There ought to be a better way to not get the PSF centroids
        # (those are part of model.data's fixed parameters)
        params_init = (
            tuple(
                (
                    param
                    for param in get_params_uniq(model.sources[0])
                    if param.free
                    or (
                        isinstance(param, g2f.CentroidXParameterD)
                        or isinstance(param, g2f.CentroidYParameterD)
                    )
                )
            )
            if (len(model.sources) == 1)
            else tuple(
                {
                    param: None
                    for source in model.sources
                    for param in get_params_uniq(source)
                    if param.free
                    or (
                        isinstance(param, g2f.CentroidXParameterD)
                        or isinstance(param, g2f.CentroidYParameterD)
                    )
                }.keys()
            )
        )

        idx_obs = 0
        for param in params_init:
            if param.linear:
                value_init = fluxes_init[idx_obs]
                limits_new = fluxes_limits[idx_obs]
                idx_obs += 1
                if idx_obs == n_observations:
                    idx_obs = 0
            else:
                value_init, limits_new = params_limits_init.get(type(param), (values_init.get(param), None))
            if limits_new:
                param.limits = g2f.LimitsD(limits_new[0], limits_new[1])
            if value_init is not None:
                param.value = value_init

        for prior in model.priors:
            if isinstance(prior, g2f.GaussianPrior):
                # TODO: Add centroid prior
                pass
            elif isinstance(prior, g2f.ShapePrior):
                prior.prior_size.mean_parameter.value = size_major

    def make_CatalogExposurePsfs(self, catexp: fitMB.CatalogExposureInputs) -> CatalogExposurePsfs:
        catexp_psf = CatalogExposurePsfs(
            # dataclasses.asdict(catexp)_makes a recursive deep copy.
            # That must be avoided.
            **{key: getattr(catexp, key) for key in catexp.__dataclass_fields__.keys()},
            channel=g2f.Channel.get(catexp.band),
            config_fit=self.config,
        )
        return catexp_psf

    @utilsTimer.timeMethod
    def run(
        self,
        catalog_multi: Sequence,
        catexps: list[fitMB.CatalogExposureInputs],
        **kwargs,
    ) -> pipeBase.Struct:
        """Run the MultiProFit source fit task on catalog-exposure pairs.

        Parameters
        ----------
        catalog_multi : `typing.Sequence`
            A multi-band, indexable source catalog.
        catexps : list[`CatalogExposureInputs`]
            Catalog-exposure-PSF model tuples to fit source models for.
        **kwargs
            Additional keyword arguments to pass to self.fit.

        Returns
        -------
        catalog : `astropy.Table`
            A table with fit parameters for the PSF model at the location
            of each source.
        """
        n_catexps = len(catexps)
        catexps_conv: list[CatalogExposurePsfs] = [None] * n_catexps
        channels: list[g2f.Channel] = [None] * n_catexps
        for idx, catexp in enumerate(catexps):
            if not isinstance(catexp, CatalogExposurePsfs):
                catexp = self.make_CatalogExposurePsfs(catexp)
            catexps_conv[idx] = catexp
            channels[idx] = catexp.channel
        self.catexps = catexps
        config_data = CatalogSourceFitterConfigData(channels=channels, config=self.config)
        catalog = self.fit(
            catalog_multi=catalog_multi, catexps=catexps_conv, config_data=config_data, **kwargs
        )
        return pipeBase.Struct(output=astropy_to_arrow(catalog))

    def validate_fit_inputs(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposurePsfs],
        config_data: CatalogSourceFitterConfigData = None,
        logger: logging.Logger = None,
        **kwargs: Any,
    ) -> None:
        errors = []
        for idx, catexp in enumerate(catexps):
            if not isinstance(catexp, CatalogExposurePsfs):
                errors.append(f"catexps[{idx=} {type(catexp)=} !isinstance(CatalogExposurePsfs)")
        if errors:
            raise RuntimeError("\n".join(errors))
