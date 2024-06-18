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

import os

import lsst.gauss2d.fit as g2f
import lsst.meas.extensions.multiprofit.fit_coadd_multiband as fitCMB
import lsst.meas.extensions.multiprofit.fit_coadd_psf as fitCP
import numpy as np
import pytest
from astropy.table import Table
from lsst.afw.image import ExposureF
from lsst.afw.table import SourceCatalog
from lsst.daf.butler.formatters.parquet import arrow_to_astropy
from lsst.multiprofit.componentconfig import (
    CentroidConfig,
    GaussianComponentConfig,
    ParameterConfig,
    SersicComponentConfig,
    SersicIndexParameterConfig,
)
from lsst.multiprofit.fit_psf import CatalogPsfFitterConfig
from lsst.multiprofit.modelconfig import ModelConfig
from lsst.multiprofit.sourceconfig import ComponentGroupConfig, SourceConfig
from lsst.pipe.tasks.fit_coadd_psf import CatalogExposurePsf

ROOT = os.environ.get("TESTDATA_CI_IMSIM_MINI", None)
has_files = (ROOT is not None) and os.path.isdir(ROOT)
filename_cat = os.path.join(ROOT, "data", "deepCoadd_meas_0_24_r_2k_ci_imsim.fits") if has_files else None
filename_exp = os.path.join(ROOT, "data", "deepCoadd_calexp_0_24_r_2k_ci_imsim.fits") if has_files else None

band = "r"
channel = g2f.Channel.get(band)
dataId = {"band": band}
do_exp_fixedcen = False
include_ps = False
n_test = 5


@pytest.fixture(scope="module")
def catalog():
    if not has_files:
        return None
    catalog = SourceCatalog.readFits(filename_cat)
    good = (catalog["detect_isPrimary"] == 1) & (catalog["merge_peak_sky"] == 0)
    good[np.where(good)[0][n_test:]] = False
    return catalog[good]


@pytest.fixture(scope="module")
def exposure():
    if not has_files:
        return None
    return ExposureF.readFits(filename_exp)


@pytest.fixture(scope="module")
def psf_fit_config():
    return fitCP.MultiProFitPsfConfig()


@pytest.fixture(scope="module")
def psf_fit_results(catalog, exposure, psf_fit_config):
    if not has_files:
        return None
    catexp = CatalogExposurePsf(dataId=dataId, catalog=catalog, exposure=exposure)
    task = fitCP.MultiProFitPsfTask(config=psf_fit_config)
    results = task.run(catexp).output
    return arrow_to_astropy(results)


@pytest.fixture(scope="module")
def source_fit_exp_fixedcen_config():
    config = fitCMB.MultiProFitSourceConfig(
        bands_fit=(band,),
        config_model=ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            centroids={
                                "default": CentroidConfig(
                                    x=ParameterConfig(fixed=True),
                                    y=ParameterConfig(fixed=True),
                                )
                            },
                            components_sersic={
                                "exp": SersicComponentConfig(
                                    sersic_index=SersicIndexParameterConfig(value_initial=1.0, fixed=True),
                                )
                            },
                        ),
                    }
                ),
            },
        ),
    )
    config.validate()
    return config


@pytest.fixture(scope="module")
def source_fit_ser_config():
    config = fitCMB.MultiProFitSourceConfig(
        bands_fit=(band,),
        config_model=ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            components_gauss={
                                "ps": GaussianComponentConfig(
                                    size_x=ParameterConfig(value_initial=0.0, fixed=True),
                                    size_y=ParameterConfig(value_initial=0.0, fixed=True),
                                    rho=ParameterConfig(value_initial=0.0, fixed=True),
                                )
                            }
                            if include_ps
                            else {},
                            components_sersic={
                                "ser": SersicComponentConfig(
                                    sersic_index=SersicIndexParameterConfig(value_initial=1.0),
                                )
                            },
                        ),
                    }
                ),
            },
        ),
    )
    config.validate()
    return config


@pytest.fixture(scope="module")
def source_fit_ser_config():
    config = fitCMB.MultiProFitSourceConfig(
        bands_fit=(band,),
        config_model=ModelConfig(
            sources={
                "": SourceConfig(
                    component_groups={
                        "": ComponentGroupConfig(
                            components_sersic={
                                "ser": SersicComponentConfig(
                                    sersic_index=SersicIndexParameterConfig(value_initial=1.0),
                                )
                            },
                        ),
                    }
                ),
            },
        ),
    )
    config.validate()
    return config


@pytest.fixture(scope="module")
def source_fit_exp_fixedcen_results(
    catalog,
    exposure,
    psf_fit_results,
    psf_fit_config,
    source_fit_exp_fixedcen_config,
):
    if not has_files:
        return None
    if not do_exp_fixedcen:
        return None
    catexp = fitCMB.CatalogExposurePsfs(
        dataId=dataId,
        catalog=catalog,
        exposure=exposure,
        table_psf_fits=psf_fit_results,
        channel=channel,
        config_fit=source_fit_exp_fixedcen_config,
    )
    task = fitCMB.MultiProFitSourceTask(config=source_fit_exp_fixedcen_config)
    results = task.run(catalog_multi=catalog, catexps=[catexp])
    return results.output.to_pandas()


@pytest.fixture(scope="module")
def source_fit_ser_results(catalog, exposure, psf_fit_results, psf_fit_config, source_fit_ser_config):
    if not has_files:
        return None
    catexp = fitCMB.CatalogExposurePsfs(
        dataId=dataId,
        catalog=catalog,
        exposure=exposure,
        table_psf_fits=psf_fit_results,
        channel=channel,
        config_fit=source_fit_ser_config,
    )
    task = fitCMB.MultiProFitSourceTask(config=source_fit_ser_config)
    results = task.run(catalog_multi=catalog, catexps=[catexp])
    return results.output.to_pandas()


@pytest.fixture(scope="module")
def source_fit_ser_shapelet_psf_results(
    catalog,
    exposure,
    psf_fit_results,
    psf_fit_config,
    source_fit_ser_config,
):
    if not has_files:
        return None
    table_psf = Table(
        meta=dict(config=CatalogPsfFitterConfig().toDict()),
    )
    catexp = fitCMB.CatalogExposurePsfs(
        dataId=dataId,
        catalog=catalog,
        exposure=exposure,
        table_psf_fits=table_psf,
        channel=channel,
        config_fit=source_fit_ser_config,
    )
    source_fit_ser_config.action_psf = fitCMB.SourceTablePsfComponentsAction()
    task = fitCMB.MultiProFitSourceTask(config=source_fit_ser_config)
    results = task.run(catalog_multi=catalog, catexps=[catexp])
    return results.output.to_pandas()


@pytest.fixture(scope="module")
def source_fits_all(
    source_fit_exp_fixedcen_results,
    source_fit_ser_results,
    source_fit_ser_shapelet_psf_results,
):
    return (
        source_fit_exp_fixedcen_results,
        source_fit_ser_results,
        source_fit_ser_shapelet_psf_results,
    )


def test_psf_fits(psf_fit_results):
    if psf_fit_results is not None:
        assert len(psf_fit_results) == n_test
        for column in psf_fit_results.columns:
            assert np.all(np.isfinite(psf_fit_results[column]))
        # TODO: Determine what checks can be done against previous values


def test_source_fits(source_fits_all):
    for results in source_fits_all:
        if results is not None:
            assert len(results) == n_test
            good = results[~results["mpf_unknown_flag"]]
            assert all(good.values.flat > -np.Inf)
            # TODO: Determine what checks can be done against previous values
