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
    "component_names_default",
    "model_names_default",
    "MultiProFitCoaddPsfFitConfig",
    "MultiProFitCoaddPsfFitTask",
    "MultiProFitCoaddObjectFitConfig",
    "MultiProFitCoaddPointFitConfig",
    "MultiProFitCoaddSersicFitConfig",
    "MultiProFitCoaddSersicFitTask",
    "MultiProFitCoaddGaussFitConfig",
    "MultiProFitCoaddGaussFitTask",
    "MultiProFitCoaddExpFitConfig",
    "MultiProFitCoaddExpFitTask",
    "MultiProFitCoaddDeVFitConfig",
    "MultiProFitCoaddDeVFitTask",
    "MultiProFitCoaddExpDeVFitConfig",
    "MultiProFitCoaddExpDeVFitTask",
)

from abc import abstractmethod
import itertools
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from lsst.daf.butler import DeferredDatasetHandle
import lsst.gauss2d.fit as g2f
from lsst.multiprofit.componentconfig import (
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.fitting.fit_source import CatalogExposureSourcesABC, CatalogSourceFitterConfigData
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.pex.config import ConfigDictField, Field
from lsst.pipe.tasks.fit_coadd_multiband import (
    CatalogExposureInputs,
    CoaddMultibandFitConfig,
    CoaddMultibandFitConnections,
    CoaddMultibandFitTask,
)
from lsst.pipe.tasks.fit_coadd_psf import CoaddPsfFitConfig, CoaddPsfFitConnections, CoaddPsfFitTask

from .fit_coadd_multiband import (
    CachedBasicModelInitializer,
    MagnitudeDependentSizePriorConfig,
    MakeBasicInitializerAction,
    ModelInitializer,
    MultiProFitSourceTask,
    PsfComponentsActionBase,
    SourceTablePsfComponentsAction,
)
from .fit_coadd_psf import MultiProFitPsfTask
from .input_config import InputConfig

component_names_default = SimpleNamespace(
    point="point",
    gauss="gauss",
    exp="exp",
    deV="deV",
    sersic="sersic",
)

model_names_default = SimpleNamespace(
    point="Point",
    gauss="Gauss",
    exp="Exp",
    deV="DeV",
    sersic="Sersic",
    fixed_cen="FixedCen",
    shapelet_psf="ShapeletPsf",
)


class MultiProFitCoaddPsfFitConfig(
    CoaddPsfFitConfig,
    pipelineConnections=CoaddPsfFitConnections,
):
    """MultiProFit PSF fit task config."""

    def setDefaults(self):
        super().setDefaults()
        self.fit_coadd_psf.retarget(MultiProFitPsfTask)
        self.fit_coadd_psf.config_fit.eval_residual = False


class MultiProFitCoaddPsfFitTask(CoaddPsfFitTask):
    """MultiProFit PSF fit task."""

    ConfigClass = MultiProFitCoaddPsfFitConfig
    _DefaultName = "multiProFitCoaddPsfFit"


class MultiProFitCoaddObjectFitConnections(CoaddMultibandFitConnections):
    def __init__(self, *, config=None):
        super().__init__(config=config)
        for name, config_input in config.inputs_init.items():
            if hasattr(self, name):
                raise ValueError(
                    f"{config_input=} {name=} is invalid, due to being an existing attribute" f" of {self=}"
                )
            if config_input.is_multipatch or not config_input.is_multiband:
                raise ValueError(
                    f"Single-band and/or multipatch initialization config_input entries ({name})"
                    f" are not supported yet."
                )
            connection = config_input.get_connection(name)
            setattr(self, name, connection)


class MultiProFitCoaddObjectFitConfig(
    CoaddMultibandFitConfig,
    pipelineConnections=MultiProFitCoaddObjectFitConnections,
):
    """Generic MultiProFit source fit task config."""

    inputs_init = ConfigDictField(
        doc="Mapping of optional input dataset configs by name, for initialization",
        keytype=str,
        itemtype=InputConfig,
        default={},
    )

    # This needs to be set, ideally in setDefaults of subclasses
    name_model = Field[str](doc="The name of the model", default=None)

    def _get_source(self):
        return next(iter(self.fit_coadd_multiband.config_model.sources.values()))

    def _get_component_group(self, source: SourceConfig | None = None):
        if source is None:
            source = self._get_source()
        return next(iter(source.component_groups.values()))

    def add_point_source(self, name: str | None = None):
        """Add a point source component.

        Parameters
        ----------
        name
            The name of the component.
        """
        if name is None:
            name = component_names_default.point
        source = self._get_source()
        group = self._get_component_group(source=source)
        if name in group.components_gauss:
            raise RuntimeError(f"{name=} component already exists in {source=}")
        group.components_gauss[name] = self.make_point_source_component()
        self.connections.name_table += model_names_default.point

    def finalize(
        self,
        add_point_source: bool = False,
        fix_centroid: bool = False,
        use_shapelet_psf: bool = False,
        prior_axrat_stddev: float | str | None = None,
    ):
        """Apply runtime configuration changes to this config.

        Parameters
        ----------
        add_point_source
            Whether to add a point source component.
        fix_centroid
            Whether to fix the centroid.
        use_shapelet_psf
            Whether to initialize PSF parameters from prior shapelet fits.
        prior_axrat_stddev
            The standard deviation for the axis ratio prior. Ignored if None,
            otherwise it must be convertible to a float.
        """
        if add_point_source:
            self.add_point_source()
        if fix_centroid:
            self.fix_centroid()
        if use_shapelet_psf:
            self.use_shapelet_psf()
        if prior_axrat_stddev is not None:
            self.set_prior_axrat_stddev(float(prior_axrat_stddev))

    def fix_centroid(self):
        """Fix (freeze) the source centroid parameters."""
        group = self._get_component_group()
        centroids = group.centroids["default"]
        centroids.x.fixed = True
        centroids.y.fixed = True
        self.connections.name_table += model_names_default.fixed_cen

    @classmethod
    @abstractmethod
    def get_model_name_default(cls) -> str:
        """Return the default name for this model in table columns."""
        raise NotImplementedError("Subclasses must implement get_model_name_default")

    @classmethod
    def get_model_name_full(cls) -> str:
        """Return a longer, more descriptive name for the model."""
        return cls.get_model_name_default()

    @abstractmethod
    def make_default_model_config(self) -> ModelConfig:
        """Make a default configuration object for this model."""
        raise NotImplementedError("Subclasses must implement make_default_model_config")

    @staticmethod
    def make_point_source_component() -> GaussianComponentConfig:
        """Make a point source component config (zero-size Gaussian)."""
        return GaussianComponentConfig(
            size_x=ParameterConfig(value_initial=0.0, fixed=True),
            size_y=ParameterConfig(value_initial=0.0, fixed=True),
            rho=ParameterConfig(value_initial=0.0, fixed=True),
        )

    @staticmethod
    def make_sersic_component(**kwargs) -> SersicComponentConfig:
        """Make a default Sersic component config.

        Parameters
        ----------
        **kwargs
            Keyword arguments to pass to the SersicIndexParameterConfig.

        Returns
        -------
        config
            The default-initialized config.
        """
        return SersicComponentConfig(
            prior_axrat_stddev=1.0,
            prior_size_stddev=0.2,
            sersic_index=SersicIndexParameterConfig(**kwargs),
        )

    @staticmethod
    def make_single_model_config(group: ComponentGroupConfig) -> ModelConfig:
        """Make a default single-source, single component group config.

        Parameters
        ----------
        group
            The component group config for the single source.

        Returns
        -------
        config
            A model config with a single nameless source and component group.
        """
        return ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": group,
                    }
                )
            }
        )

    def set_prior_axrat_stddev(self, stddev: float) -> None:
        """Set the standard deviation for all axis ratio priors.

        Parameters
        ----------
        stddev
            The standard deviation.
        """
        for source in self.fit_coadd_multiband.config_model.sources.values():
            for group in source.component_groups.values():
                for comp in itertools.chain(
                    group.components_gauss.values(),
                    group.components_sersic.values(),
                ):
                    comp.prior_axrat_stddev = stddev

        group = self._get_component_group()
        centroids = group.centroids["default"]
        centroids.x.fixed = True
        centroids.y.fixed = True
        self.connections.name_table += model_names_default.fixed_cen

    def setDefaults(self):
        super().setDefaults()
        self.fit_coadd_multiband.retarget(MultiProFitSourceTask)
        self.fit_coadd_multiband.action_psf = PsfComponentsActionBase()
        self.fit_coadd_multiband.bands_fit = ("u", "g", "r", "i", "z", "y")

        self.fit_coadd_multiband.config_model = self.make_default_model_config()
        self.name_model = self.get_model_name_default()
        self.connections.name_table = self.name_model

    def use_shapelet_psf(self):
        """Reconfigure self to use prior shapelet PSF fit parameters."""
        self.fit_coadd_multiband.action_psf = SourceTablePsfComponentsAction()
        self.drop_psf_connection = True
        self.connections.name_table += model_names_default.shapelet_psf


class MultiProFitCoaddObjectFitTask(CoaddMultibandFitTask):
    """MultiProFit coadd object model fitting task."""

    ConfigClass = MultiProFitCoaddObjectFitConfig
    _DefaultName = "multiProFitCoaddObjectFit"

    def make_kwargs(self, butlerQC, inputRefs, inputs):
        inputs_init = {}
        for name, config in self.config.inputs_init.items():
            input_ = inputs[name][0]
            if isinstance(input_, DeferredDatasetHandle):
                parameters = None
                if config.needs_metadata:
                    parameters = {"strip_astropy_meta_yaml": False}
                input_ = input_.get(parameters=parameters)

            inputs_init[name] = (config, input_)

        kwargs = {}
        if inputs_init:
            kwargs["inputs_init"] = inputs_init

        return kwargs


class MultiProFitCoaddPointFitConfig(
    MultiProFitCoaddObjectFitConfig,
    pipelineConnections=MultiProFitCoaddObjectFitConnections,
):
    """MultiProFit single Sersic model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.point

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Point Source"

    def make_default_model_config(self) -> ModelConfig:
        config_group = ComponentGroupConfig()
        # This is a bit silly but add_point_source will look for the first
        # source so it must be added now. Perhaps add_point_source should
        # add to a config instance or only self by default
        self.fit_coadd_multiband.config_model = self.make_single_model_config(group=config_group)
        self.add_point_source()
        return self.fit_coadd_multiband.config_model


class MultiProFitCoaddSersicFitConfig(
    MultiProFitCoaddObjectFitConfig,
    pipelineConnections=MultiProFitCoaddObjectFitConnections,
):
    """MultiProFit single Sersic model fit task config."""

    def _rename_defaults(
        self,
        name_new: str,
        name_model: str | None = None,
        name_old: str | None = None,
        index_new: float | None = None,
        fix_index: bool = False,
    ):
        """Rename the default Sersic component to something more specific.

        This is intended for fixed index models such as exponential and
        deVaucouleurs.

        Parameters
        ----------
        name_new
            The new name for the component.
        name_model
            The new name of the model. Default is to capitalize name_new.
        name_old
            The old name of the component. Default is to set to
            component_names_default.sersic.
        index_new
            The initial value for the Sersic index.
        fix_index
            Whether the fix the index to the new value.
        """
        if name_old is None:
            name_old = component_names_default.sersic
        if name_model is None:
            name_model = name_new.capitalize()
        group = self._get_component_group()
        comps_sersic = group.components_sersic

        if name_new in comps_sersic:
            raise RuntimeError(f"{name_new=} is already in {comps_sersic=}")

        comp_sersic = comps_sersic[name_old]
        del comps_sersic[name_old]
        if index_new is not None:
            comp_sersic.sersic_index.value_initial = index_new
        if fix_index:
            comp_sersic.sersic_index.fixed = True
        comps_sersic[name_new] = comp_sersic

        if prior_old := self.fit_coadd_multiband.size_priors.get(name_old):
            self.fit_coadd_multiband.size_priors[name_new] = prior_old
            del self.fit_coadd_multiband.size_priors[name_old]

        self.name_model = name_model
        self.connections.name_table = name_model

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.sersic

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Sersic"

    def make_default_model_config(self) -> ModelConfig:
        config_group = ComponentGroupConfig(
            components_sersic={
                component_names_default.sersic: self.make_sersic_component(),
            },
        )
        return self.make_single_model_config(group=config_group)

    def setDefaults(self):
        super().setDefaults()
        # This is in pixels and based on DC2. See DM-46498 for details.
        self.fit_coadd_multiband.size_priors[component_names_default.sersic] = (
            MagnitudeDependentSizePriorConfig(
                intercept_mag=22.6,
                slope_median_per_mag=-0.15,
                slope_stddev_per_mag=0,
            )
        )


class MultiProFitCoaddSersicFitTask(MultiProFitCoaddObjectFitTask):
    """MultiProFit single Sersic model fit task."""

    ConfigClass = MultiProFitCoaddSersicFitConfig
    _DefaultName = "multiProFitCoaddSersicFit"


class MultiProFitCoaddGaussFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=MultiProFitCoaddObjectFitConnections,
):
    """MultiProFit single Gaussian model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.gauss

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Gaussian"

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(
            name_new=component_names_default.gauss,
            name_model=model_names_default.gauss,
            index_new=0.5,
            fix_index=True,
        )


class MultiProFitCoaddGaussFitTask(MultiProFitCoaddObjectFitTask):
    """MultiProFit single Gaussian model fit task."""

    ConfigClass = MultiProFitCoaddGaussFitConfig
    _DefaultName = "multiProFitCoaddGaussFit"


class MultiProFitCoaddExpFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=MultiProFitCoaddObjectFitConnections,
):
    """MultiProFit single exponential model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.exp

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Exponential"

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(
            name_new=component_names_default.exp,
            name_model=model_names_default.exp,
            index_new=1.0,
            fix_index=True,
        )
        # These are typical values from DC2 and could/should be switched to a
        # more data-driven prior (from HSC?)
        prior_size = self.fit_coadd_multiband.size_priors[component_names_default.exp]
        prior_size.intercept_mag = 23.4
        prior_size.slope_median_per_mag = -0.14


class MultiProFitCoaddExpFitTask(MultiProFitCoaddObjectFitTask):
    """MultiProFit single exponential model fit task."""

    ConfigClass = MultiProFitCoaddExpFitConfig
    _DefaultName = "multiProFitCoaddExpFit"


class MultiProFitCoaddDeVFitConfig(
    MultiProFitCoaddSersicFitConfig,
    pipelineConnections=MultiProFitCoaddObjectFitConnections,
):
    """MultiProFit single DeVaucouleurs model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return model_names_default.deV

    @classmethod
    def get_model_name_full(cls) -> str:
        return "de Vaucouleurs"

    def setDefaults(self):
        super().setDefaults()
        self._rename_defaults(
            name_new=component_names_default.deV,
            name_model=model_names_default.deV,
            index_new=4.0,
            fix_index=True,
        )
        # These are typical values from DC2 and could/should be switched to a
        # more data-driven prior (from HSC?). See DM-46498 for details.
        prior_size = self.fit_coadd_multiband.size_priors[component_names_default.deV]
        prior_size.intercept_mag = 21.2
        prior_size.slope_median_per_mag = -0.14


class MultiProFitCoaddDeVFitTask(MultiProFitCoaddObjectFitTask):
    """MultiProFit single DeVaucouleurs model fit task."""

    ConfigClass = MultiProFitCoaddDeVFitConfig
    _DefaultName = "multiProFitCoaddDeVFit"


class CachedChainedModelInitializer(CachedBasicModelInitializer):
    def get_centroid_and_shape(
        self,
        source: Mapping[str, Any],
        catexps: list[CatalogExposureSourcesABC],
        config_data: CatalogSourceFitterConfigData,
        values_init: Mapping[g2f.ParameterD, float] | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float, float]]:
        row_best = None
        chisq_red_min = math.inf
        for name, input_data in self.inputs.items():
            data = input_data.data
            index_row = input_data.id_index.get(source["id"])
            if index_row is not None:
                row = data[index_row]
                chisq_red = input_data.get_column("chisq_reduced", data=row)
                if chisq_red < chisq_red_min:
                    row_best = (row, input_data)
                    chisq_red_min = chisq_red
        if row_best is None:
            return super().get_centroid_and_shape(
                source=source,
                catexps=catexps,
                config_data=config_data,
                values_init=values_init,
            )
        row_best, input_data = row_best
        cen_x, cen_y, reff_x, reff_y, rho = (
            input_data.get_column(column, data=row_best)
            for column in (
                input_data.get_column("cen_x").name,
                input_data.get_column("cen_y").name,
                input_data.get_column(f"{input_data.size_column}_x").name,
                input_data.get_column(f"{input_data.size_column}_y").name,
                input_data.get_column("rho").name,
            )
        )
        return (cen_x, cen_y), (reff_x, reff_y, rho)


class MakeCachedChainedInitializerAction(MakeBasicInitializerAction):
    def _make_initializer(
        self,
        catalog_multi: Sequence,
        catexps: list[CatalogExposureInputs],
        config_data: CatalogSourceFitterConfigData,
    ) -> ModelInitializer:
        sources, priors = config_data.sources_priors
        return CachedChainedModelInitializer(config=self.config, priors=priors, sources=sources)


class MultiProFitCoaddExpDeVFitConfig(
    MultiProFitCoaddObjectFitConfig,
    pipelineConnections=MultiProFitCoaddObjectFitConnections,
):
    """MultiProFit single Exponential+DeVaucouleurs model fit task config."""

    @classmethod
    def get_model_name_default(cls) -> str:
        return f"{model_names_default.exp}{model_names_default.deV}"

    @classmethod
    def get_model_name_full(cls) -> str:
        return "Exponential + de Vaucouleurs"

    def make_default_model_config(self) -> ModelConfig:
        config_group = ComponentGroupConfig(
            components_sersic={
                component_names_default.exp: self.make_sersic_component(value_initial=1.0, fixed=True),
                component_names_default.deV: self.make_sersic_component(value_initial=4.0, fixed=True),
            },
        )
        return self.make_single_model_config(group=config_group)

    def setDefaults(self):
        super().setDefaults()
        self.fit_coadd_multiband.action_initializer = MakeCachedChainedInitializerAction()
        self.fit_coadd_multiband.config_model = self.make_default_model_config()
        self.name_model = self.get_model_name_default()
        self.connections.name_table = self.name_model

        size_priors = self.fit_coadd_multiband.size_priors
        size_priors[component_names_default.exp] = MagnitudeDependentSizePriorConfig(
            intercept_mag=23.3,
            slope_median_per_mag=-0.14,
        )
        size_priors[component_names_default.deV] = MagnitudeDependentSizePriorConfig(
            intercept_mag=21.2,
            slope_median_per_mag=-0.14,
        )


class MultiProFitCoaddExpDeVFitTask(MultiProFitCoaddObjectFitTask):
    """MultiProFit single ExpDeV model fit task."""

    ConfigClass = MultiProFitCoaddExpDeVFitConfig
    _DefaultName = "multiProFitCoaddExpDeVFit"
