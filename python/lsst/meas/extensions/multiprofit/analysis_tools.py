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
# but WITHOUT ANY WARRANTY; without e   ven the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from lsst.analysis.tools.atools.genericBuild import FluxConfig, MomentsConfig, SizeConfig
from lsst.analysis.tools.atools.sizeMagnitude import SizeMagnitudePlot
from lsst.analysis.tools.contexts import CoaddContext

moments_sersic = MomentsConfig(xx="reff_x", yy="reff_y", xy="rho")


class MultiProFitSizeMagnitudePlot(SizeMagnitudePlot):
    """A size-magnitude plot with default MultiProFit column names."""

    def _get_flags_default(self, name_model: str):
        """Get the default MultiProFit flags for a given model.

        Parameters
        ----------
        name_model
            The short name of the model in table columns.

        Returns
        -------
        flags_false
            The flags that must be false for the fit to be good.
        flags_true
            The flags that must be true for the fit to be good.
        """
        flags_false = [
            f"mpf_{name_model}_{flag}_flag" for flag in ("unknown", "is_parent", "not_primary", "psf_fit")
        ]
        flags_true = []
        return flags_false, flags_true

    def _set_model_defaults(
        self, name_model: str, label_model: str, name_component: str, label_component: str
    ) -> None:
        """Set default values for a given model.

        Parameters
        ----------
        name_model
            The short name of the model in table columns.
        label_model
            The label for the model in plots.
        name_component
            The shortname of the component in table columns.
        label_component
            The label for the component in plots.
        """
        flags_false, flags_true = self._get_flags_default(name_model)
        self.prep.selectors.flagSelector.selectWhenFalse = flags_false
        self.prep.selectors.flagSelector.selectWhenTrue = flags_true

        name_full = f"{name_model}_{name_component}"
        flux_config = FluxConfig(
            key_flux=f"mpf_{name_full}_{{band}}_flux",
            key_flux_error=f"mpf_{name_full}_{{band}}_flux_err",
            name_flux=label_model,
            name_flux_short=name_model,
        )
        size_config = SizeConfig(
            key_size=f"mpf_{name_full}_{{suffix}}",
            name_size=f"{name_component} {'$R_{eff}$'}",
        )
        self.fluxes = {name_full: flux_config}
        self.mag_x = name_full
        self.sizes = {name_full: size_config}
        self.size_y = name_full

    def setDefaults(self):
        super().setDefaults()
        self.applyContext(CoaddContext)
        self.size_type = "determinantRadius"
        self.is_covariance = False
        self.produce.plot.xLims = (17, 29)
        self.produce.plot.yLims = (-4, 3)


class MultiProFitExpDevSizeMagnitudePlot(MultiProFitSizeMagnitudePlot):
    """A size-magnitude plot for the MultiProFit Exp.Dev. model."""

    label_model: str = "MPF Exp+Dev"
    name_model: str = "expdev"

    def setDefaults(self):
        super().setDefaults()
        self.config_moments = moments_sersic


class MultiProFitExpDevBulgeSizeMagnitudePlot(MultiProFitExpDevSizeMagnitudePlot):
    """A size-magnitude plot for the bulge (de Vaucouleurs) component of a
    MultiProFit ExpDev model.
    """

    label_component: str = "de Vauc."
    name_component: str = "dev"

    def setDefaults(self):
        super().setDefaults()
        self._set_model_defaults(
            name_model=self.name_model,
            label_model=self.label_model,
            name_component=self.name_component,
            label_component=self.label_component,
        )


class MultiProFitExpDevDiskSizeMagnitudePlot(MultiProFitExpDevSizeMagnitudePlot):
    """A size-magnitude plot for the disk (exponential) component of a
    MultiProFit ExpDev model.
    """

    label_component: str = "Exponential"
    name_component: str = "exp"

    def setDefaults(self):
        super().setDefaults()
        self._set_model_defaults(
            name_model=self.name_model,
            label_model=self.label_model,
            name_component=self.name_component,
            label_component=self.label_component,
        )


class MultiProFitSersicSizeMagnitudePlot(MultiProFitSizeMagnitudePlot):
    """A size-magnitude plot for the MultiProFit Sersic model."""

    label_model: str = "MPF Sersic"
    name_model: str = "ser"
    label_component: str = "Sersic"
    name_component: str = "ser"

    def setDefaults(self):
        super().setDefaults()
        self.config_moments = moments_sersic
        self._set_model_defaults(
            name_model=self.name_model,
            label_model=self.label_model,
            name_component=self.name_component,
            label_component=self.label_component,
        )
