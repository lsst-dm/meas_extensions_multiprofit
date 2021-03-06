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

from collections import defaultdict, namedtuple
import copy
import gauss2d
import logging
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.meas.base.measurementInvestigationLib import rebuildNoiseReplacer
from lsst.meas.modelfit.display import buildCModelImages
from lsst.meas.modelfit.cmodel.cmodelContinued import CModelConfig
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
try:
    import modelling_research.make_cutout as mrCutout
    has_mrCutout = True
except Exception:
    has_mrCutout = False
import lsst.pipe.tasks.fit_multiband as fitMb
import matplotlib.pyplot as plt
import multiprofit.fitutils as mpfFit
import multiprofit.objects as mpfObj
from multiprofit.priors import get_hst_size_prior
import numpy as np
import os
import pprint
import time
import traceback
from typing import Iterable
from .utils import defaultdictNested, get_spanned_image, join_and_filter

pixel_scale_hst = 0.03
pixel_scale_hsc = 0.168


class FitFailedError(RuntimeError):
    pass


class MultiProFitConfig(fitMb.MultibandFitSubConfig):
    """Configuration for the MultiProFit profile fitter.

    Notes
    -----
    gaussianOrderSersic only has a limited number of valid values
    (those supported by multiprofit's MultiGaussianApproximationComponent).
    """
    backgroundPriorMultiplier = pexConfig.Field(dtype=float, default=0.,
                                                doc="Multiplier for background level prior sigma")
    backgroundSigmaAdd = pexConfig.Field(dtype=float, default=10,
                                         doc="Multiple of background level sigma to add to image for fits")
    bands_fit = pexConfig.ListField(dtype=str, default=[], doc="List of bandpass filters to fit",
                                    listCheck=lambda x: len(set(x)) == len(x))
    bboxDilate = pexConfig.Field(
        dtype=int,
        default=0,
        doc="Number of pixels to dilate (expand) source bounding boxes and hence fitting regions by",
    )
    computeMeasModelfitLikelihood = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to compute the log-likelihood of best-fit meas_modelfit parameters per model",
    )
    deblend = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to fit parents simultaneously with children",
    )
    deblendFromDeblendedFits = pexConfig.Field(
        dtype=bool, default=False, doc="Whether to fit parents simultaneously and linearly using prior "
                                       "fits to deblended children")
    deblendNonlinearMaxSources = pexConfig.Field(
        dtype=int, default=0, doc="Maximum children in blend to allow simultaneous non-linear fit")
    disableNoiseReplacer = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to disable replacement of nearby sources with noise"
            "when fitting deblended single sources",
    )
    estimateContiguousDenoisedMoments = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether models initiated from moments should estimate within contiguous positive pixels in a"
            " naively de-noised image",
    )
    field_localbg = pexConfig.Field(
        dtype=str,
        default='base_LocalBackground',
        doc="Field name (prefix to _instFlux) to read the local background level from if "
            "usePriorBackgroundLocalEstimate is True",
    )
    filenameOut = pexConfig.Field(dtype=str, optional=True, doc="Filename for output of FITS table")
    filenameOutDeblend = pexConfig.Field(
        dtype=str,
        optional=True,
        doc="Filename for output of FITS table with deblended fits",
    )
    fitBackground = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to fit a flat background level for each band",
    )
    fitCModel = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to perform a CModel (linear combo of exponential and deVaucouleurs) fit per source;"
            "necessitates doing exp. + deV. fits",
    )
    fitCModelExp = pexConfig.Field(dtype=bool, default=False,
                                   doc="Whether to perform an exponential fit with a fixed center (as "
                                       "CModel does in meas_modelfit) per source")
    fitGaussian = pexConfig.Field(dtype=bool, default=False,
                                  doc="Whether to perform a single Gaussian fit without PSF convolution")
    fitGaussianPsfConv = pexConfig.Field(dtype=bool, default=True,
                                         doc="Whether to perform a PSF-convolved single Gaussian fit")
    fitHstCosmos = pexConfig.Field(dtype=bool, default=False,
                                   doc="Whether to fit COSMOS HST F814W images instead of repo images")
    fitDevExpFromCModel = pexConfig.Field(dtype=bool, default=False,
                                          doc="Whether to perform a MG Sersic approximation Dev+Exp profile "
                                              "fit (initialized from previous exp./Dev. fits) per source")
    fitPrereqs = pexConfig.Field(dtype=bool, default=False, doc="Set fit(Model) flags for necessary "
                                                                "prerequisites even if not specified")
    fitSersic = pexConfig.Field(dtype=bool, default=True, doc="Whether to perform a MG Sersic approximation "
                                                              "profile fit per source")
    fitSersicFromCModel = pexConfig.Field(dtype=bool, default=False,
                                          doc="Whether to perform a MG Sersic approximation profile fit "
                                              "(initalized from previous exp./dev. fits) per source;"
                                              " ignored if fitCModel is False")
    fitSersicFromGauss = pexConfig.Field(dtype=bool, default=True,
                                         doc="Whether to perform a MG Sersic approximation profile fit "
                                             "(initalized from previous gauss fit) per source")
    fitSersicAmplitude = pexConfig.Field(dtype=bool, default=True,
                                         doc="Whether to perform a linear fit of the Gaussian"
                                             " amplitudes for the MG Sersic approximation profile fit per"
                                             " source; has no impact if fitSersic is False")
    fitSersicFromCModelAmplitude = pexConfig.Field(dtype=bool, default=True,
                                                   doc="Whether to perform a linear fit of the Gaussian"
                                                       " amplitudes for the MG Sersic approximation profile"
                                                       " fit (initialized from previous exp.Dev. fits) per"
                                                       " source; has no impact if fitSersicFromCModel is"
                                                       " False")
    fitSersicX2FromDevExp = pexConfig.Field(dtype=bool, default=False,
                                            doc="Whether to perform a MG Sersic approximation SersicX2 "
                                                "profile fit (initialized from previous devExp fit) per "
                                                "source; ignored if fitDevExpFromCModel is False")
    fitSersicX2DEAmplitude = pexConfig.Field(dtype=bool, default=False,
                                             doc="Whether to perform a linear fit of the Gaussian "
                                                 "amplitudes for the MG SersicX2 approximation profile fit "
                                                 "for each source; ignored if fitSersicX2FromDevExp is False")
    fitSersicX2FromSerExp = pexConfig.Field(dtype=bool, default=False,
                                            doc="Whether to perform a MG Sersic approximation SersicX2 "
                                                "profile fit (initialized from previous serExp fit) per "
                                                "source; ignored if fitSersicFromCModel is False")
    fitSersicX2SEAmplitude = pexConfig.Field(dtype=bool, default=False,
                                             doc="Whether to perform a linear fit of the Gaussian "
                                                 "amplitudes for the MG SersicX2 approximation profile fit "
                                                 "for each source; ignored if fitSersicX2FromSeRExp is False")
    gaussianOrderPsf = pexConfig.Field(dtype=int, default=2, doc="Number of Gaussians components for the PSF")
    gaussianOrderSersic = pexConfig.Field(dtype=int, default=8, doc="Number of Gaussians components for the "
                                                                    "MG Sersic approximation galaxy profile")
    gaussianSizePriorSigma = pexConfig.Field(
        dtype=float, default=0.2, doc="Std. dev. of the size (sigma) prior for the Gaussian model (pixels)")
    idx_begin = pexConfig.Field(dtype=int, default=0, doc="Initial row index to fit")
    idx_end = pexConfig.Field(dtype=int, default=-1, doc="Final row index to fit")
    intervalOutput = pexConfig.Field(dtype=int, default=100, doc="Number of sources to fit before writing "
                                                                 "output")
    isolatedOnly = pexConfig.Field(dtype=bool, default=False, doc="Whether to fit only isolated sources")
    maxParentFootprintPixels = pexConfig.Field(dtype=int, default=1000000,
                                               doc="Maximum number of pixels in a parent footprint allowed "
                                                   "before failing or reverting to child footprint")
    maxNChildParentFit = pexConfig.Field(dtype=int, default=25, doc="Maximum number of children allowed to "
                                                                    "fit a parent footprint")
    outputChisqred = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the reduced chi^2 of "
                                                                   "each model's best fit")
    outputLogLikelihood = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the log likelihood "
                                                                        "of each model's best fit")
    outputRuntime = pexConfig.Field(dtype=bool, default=True, doc="Whether to save the runtime of each model")
    pathCosmosGalsim = pexConfig.Field(
        dtype=str, optional=True,
        doc="Path containing COSMOST HST F814W-based real_galaxy_catalog_25.2.fits and "
            "real_galaxy_PSF_images_25.2_n[1-88].fits (https://zenodo.org/record/3242143)"
    )
    plot = pexConfig.Field(dtype=bool, default=False, doc="Whether to plot each source fit")
    plotOnly = pexConfig.Field(dtype=bool, default=False, doc="Whether to only attempt to plot existing "
                                                              "fits; requires resume=True")
    printTrace = pexConfig.Field(dtype=bool, default=False, doc="Print traceback for errors")
    priorCentroidSigma = pexConfig.Field(dtype=float, optional=True, doc="Centroid prior sigma")
    priorMagBand = pexConfig.Field(dtype=str, optional=True, doc="Band of the magnitude to use for priors")
    priorMagField = pexConfig.Field(
        dtype=str,
        default='base_PsfFlux_mag',
        doc="Magnitude field to use for priors",
    )
    psfHwhmShrink = pexConfig.Field(
        dtype=float, default=0.,
        doc="Length (pix) to subtract from PSF HWHM (x,y) in quadrature before fitting PSF-convolved models"
    )
    replaceDataByModel = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to replace the actual data to fit with a noisy realization of each model by using its"
            " initial parameters)"
    )
    resume = pexConfig.Field(dtype=bool, default=False, doc="Whether to resume from the previous output file")
    skipDeblendTooManyPeaks = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to skip fitting sources with the deblend_tooManyPeaks flag set",
    )
    useSpans = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use spans for source cutouts (i.e. the detected pixels) rather than the whole box",
    )
    useSdssShape = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use the baseSdssShape* moments to initialize Gaussian fits",
    )
    useParentFootprint = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use the parent's footprint when fitting deblended children",
    )
    usePriorShapeDefault = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to use the default shape prior",
    )
    usePriorBackgroundLocalEstimate = pexConfig.Field(
        dtype=bool, default=False,
        doc="Whether to use a local estimate of the background level to set the"
            " background prior mean/sigma; generally a bad idea",
    )

    def bands_read_only(self):
        return self.priorMagBand if self.usePriorShapeDefault else ()

    def getModelSpecs(self):
        """Get a list of dicts of model specifications for MultiProFit.

        Returns
        -------
        modelSpecs : `list` [`dict`]
            MultiProFit model specifications, as used by
            multiprofit.fitutils.fit_galaxy_exposures().
        """
        modelSpecs = []
        nameMG = f"mg{self.gaussianOrderSersic}"
        namePsfModel = f"gaussian:{self.gaussianOrderPsf}"
        nameSersicPrefix = f"mgsersic{self.gaussianOrderSersic}"
        nameSersicModel = f"{nameSersicPrefix}:1"
        nameSersicAmpModel = f"gaussian:{self.gaussianOrderSersic}+rscale:1"
        nameSersicX2Model = f"{nameSersicPrefix}:2"
        nameSersicX2AmpModel = f"gaussian:{2*self.gaussianOrderSersic}+rscale:2"
        allParams = "cenx;ceny;nser;sigma_x;sigma_y;rscale;rho"
        if self.fitPrereqs:
            prereqs = {
                'fitGaussianPsfConv': ['fitSersicFromGauss', 'fitCModel'],
                'fitSersic': ['fitSersicAmplitude'],
                'fitSersicFromCModel': ['fitSersicFromCModelAmplitude', 'fitSersicX2FromSerExp'],
                'fitCModel': ['fitSersicFromCModel', 'fitDevExpFromCModel'],
                'fitSersicX2FromDevExp': ['fitSersicX2DEAmplitude'],
                'fitDevExpFromCModel': ['fitSersicX2FromDevExp'],
                'fitSersicX2FromSerExp': ['fitSersicX2SEAmplitude'],
            }
            for req, depends in prereqs.items():
                dict_self = self.toDict()
                if (not dict_self[req]) and any([dict_self[dep] for dep in depends]):
                    self.update(**{req: True})
        defaults = {
            'psfmodel': namePsfModel,
            'psfpixel': True,
        }
        init_values = self.plotOnly or self.deblendFromDeblendedFits
        inittype_moments = "moments" if not init_values else "values"
        if self.fitGaussianPsfConv:
            modelSpecs.append(
                mpfFit.ModelSpec(
                    name="gausspx", model=nameSersicModel, fixedparams='nser', initparams="nser=0.5",
                    inittype=inittype_moments, **defaults
                )
            )
        if self.fitCModel:
            modelSpecs.extend([
                mpfFit.ModelSpec(
                    name=f"{nameMG}expgpx", model=nameSersicModel, fixedparams='nser', initparams="nser=1",
                    inittype="guessgauss2exp:gausspx" if not init_values else "values", **defaults
                ),
                mpfFit.ModelSpec(
                    name=f"{nameMG}devepx", model=nameSersicModel, fixedparams='nser', initparams="nser=4",
                    inittype=f"guessexp2dev:{nameMG}expgpx" if not init_values else "values", **defaults
                ),
                mpfFit.ModelSpec(
                    name=f"{nameMG}cmodelpx", model=f"{nameSersicPrefix}:2",
                    fixedparams="cenx;ceny;nser;sigma_x;sigma_y;rho", initparams="nser=4,1",
                    inittype=f"{nameMG}devepx;{nameMG}expgpx", **defaults
                ),
            ])
            if self.fitSersicFromCModel:
                modelSpecs.append(mpfFit.ModelSpec(
                    name=f"{nameMG}serbpx", model=nameSersicModel, fixedparams='', initparams='',
                    inittype="best" if not init_values else "values", **defaults
                ))
                if self.fitSersicFromCModelAmplitude:
                    modelSpecs.append(mpfFit.ModelSpec(
                        name=f"{nameMG}serbapx", model=nameSersicAmpModel, fixedparams=allParams,
                        initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}serbpx",
                        unlimitedparams="sigma_x;sigma_y", **defaults
                    ))
                if self.fitSersicX2FromSerExp:
                    modelSpecs.append(mpfFit.ModelSpec(
                        name=f"{nameMG}serx2sepx", model=nameSersicX2Model, fixedparams='', initparams='',
                        inittype=f"{nameMG}serbpx;{nameMG}expgpx" if not init_values else "values", **defaults
                    ))
                    if self.fitSersicX2SEAmplitude:
                        modelSpecs.append(mpfFit.ModelSpec(
                            name=f"{nameMG}serx2seapx", model=nameSersicX2AmpModel, fixedparams=allParams,
                            initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}serx2sepx",
                            unlimitedparams="sigma_x;sigma_y", **defaults
                        ))
            if self.fitDevExpFromCModel:
                modelSpecs.append(mpfFit.ModelSpec(
                    name=f"{nameMG}devexppx", model=nameSersicX2Model,
                    fixedparams='nser', initparams='nser=4,1',
                    inittype=f"{nameMG}devepx;{nameMG}expgpx" if not init_values else "values",
                    **defaults
                ))
                if self.fitSersicX2FromDevExp:
                    modelSpecs.append(mpfFit.ModelSpec(
                        name=f"{nameMG}serx2px", model=nameSersicX2Model, fixedparams='', initparams='',
                        inittype=f"{nameMG}devexppx" if not init_values else "values", **defaults
                    ))
                    if self.fitSersicX2DEAmplitude:
                        modelSpecs.append(mpfFit.ModelSpec(
                            name=f"{nameMG}serx2apx", model=nameSersicX2AmpModel, fixedparams=allParams,
                            initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}serx2px", **defaults
                        ))
        if self.fitCModelExp:
            modelSpecs.append(mpfFit.ModelSpec(
                name=f"{nameMG}expcmpx", model=nameSersicModel, fixedparams='cenx;ceny;nser',
                initparams="nser=1", inittype="moments", **defaults
            ))
        if self.fitSersicFromGauss:
            modelSpecs.append(mpfFit.ModelSpec(
                name=f"{nameMG}sergpx", model=nameSersicModel, fixedparams='', initparams="nser=1",
                inittype="guessgauss2exp:gausspx" if not init_values else "values", **defaults
            ))
        if self.fitSersic:
            modelSpecs.append(mpfFit.ModelSpec(
                name=f"{nameMG}sermpx", model=nameSersicModel, fixedparams='', initparams="nser=1",
                inittype=inittype_moments, **defaults
            ))
            if self.fitSersicAmplitude:
                modelSpecs.append(mpfFit.ModelSpec(
                    name=f"{nameMG}serapx", model=nameSersicAmpModel, fixedparams=allParams,
                    initparams="rho=inherit;rscale=modify", inittype=f"{nameMG}sermpx",
                    unlimitedparams="sigma_x;sigma_y", **defaults
                ))
        return modelSpecs


class MultiProFitTask(fitMb.MultibandFitSubTask):
    """Run MultiProFit on Exposure/SourceCatalog pairs in multiple bands.

    This task uses MultiProFit to fit the PSF and various analytic profiles to
    every source in an Exposure. The set of models to be fit can be configured,
    with the naming scheming indicating the model type and optionally
    additional info on how the model is initialized and/or fit.

    Notes
    -----
    See https://github.com/lsst-dm/multiprofit for more MultiProFit info, and
    https://github.com/lsst-dm/modelling_research for various investigations of
    its suitability for LSST data.
    """
    ConfigClass = MultiProFitConfig
    _DefaultName = "multibandProFit"
    meas_modelfit_models = ("dev", "exp", "cmodel")
    keynames = {'runtimeKey': 'time_total', 'failFlagKey': 'fail_flag'}
    ParamDesc = namedtuple('ParamInfo', ['doc', 'unit'])
    params_multiprofit = {
        'cenx': ParamDesc('Centroid x coordinate', 'pixel'),
        'ceny': ParamDesc('Centroid y coordinate', 'pixel'),
        'flux': ParamDesc('Total flux', 'Jy'),
        'nser': ParamDesc('Sersic index', ''),
        'rho': ParamDesc('Ellipse correlation coefficient', ''),
        'sigma_x': ParamDesc('Ellipse x half-light distance', 'pixel'),
        'sigma_y': ParamDesc('Ellipse y half-light distance', 'pixel'),
        'fluxFrac': ParamDesc('Flux fraction', ''),
    }
    params_extra = {'chisqred', 'loglike', 'time', 'nEvalFunc', 'nEvalGrad'}
    prefix = 'multiprofit_'
    prefix_psf = 'psf_'
    postfix_nopsf = '-nopsf'

    @property
    def schema(self):
        return self._schema

    def __init__(self, schema, modelSpecs=None, **kwargs):
        """Initialize the task with model specifications.

        Parameters
        ----------
        schema : `lsst.afw.table.Schema`
            Schema of the input catalog to extend.
        modelSpecs : iterable of `dict`, optional
            MultiProFit model specifications, as used by
            `multiprofit.fitutils.fit_galaxy_exposures`.
            Defaults to `self.config.getModelSpecs`().
        **kwargs
            Additional keyword arguments to pass to
            `lsst.pipe.base.Task.__init__`
        """
        super().__init__(schema=schema, **kwargs)
        self.log.info(pprint.pformat(self.config.toDict()))
        if modelSpecs is None:
            modelSpecs = self.config.getModelSpecs()
        self.modelSpecs = modelSpecs
        self.modelSpecsDict = {modelspec.name: modelspec for modelspec in modelSpecs}
        self.modeller = mpfObj.Modeller(None, 'scipy')
        self.models = {}
        self.mask_names_zero = ['BAD', 'EDGE', 'SAT', 'NO_DATA']
        self.bbox_ref = None
        # Initialize the schema, in case users need it before runtime
        img = gauss2d.make_gaussian_pixel(10, 10, 1, 3, 1, 0, 0, 20, 0, 20, 20, 20)
        exposurePsfs = [
            (mpfObj.Exposure(band=band, image=img, error_inverse=None), img)
            for band in self.config.bands_fit
        ]
        results = mpfFit.fit_galaxy_exposures(
            exposurePsfs, self.config.bands_fit, self.modelSpecs, skip_fit=True, skip_fit_psf=True,
            logger=logging.Logger(name='multiProFitTask_init', level=21)
        )
        self._schema, self.mapper, self.fields = self.__getSchema(self.config.bands_fit, results, schema)

    @staticmethod
    def _getMapper(schema):
        """Return a suitably configured schema mapper.

        Parameters
        ----------
        schema: `lsst.afw.table.Schema`
            A table schema to setup a mapper for and add basic keys to.

        Returns
        -------
        mapper: `lsst.afw.table.SchemaMapper`
            A mapper for `schema`.
        """
        mapper = afwTable.SchemaMapper(schema, True)
        mapper.addMinimalSchema(schema, True)
        mapper.editOutputSchema().disconnectAliases()
        return mapper

    def _getBBoxDilated(self, footprint):
        """Get a suitably dilated bounding box for a given footprint.

        Parameters
        ----------
        footprint : `lsst.afw.detection.Footprint`
            The footprint to optionally dilate and then return the bbox of.

        Returns
        -------
        bbox : `lsst.geom.Box2I`
            The bounding box of the footprint, dilated as per config
            parameters.
        dilate : `int`
            The number of pixels the footprint was actually dilated by.

        Notes
        -----
        `footprint` is modified as long as dilation is possible, which it may
        not be if it is right at the edge of the exposure.
        Dilation is always a fixed number of pixels in every direction
        (i.e. square, never rectangular).
        """
        bbox = footprint.getBBox()
        dilate = int(self.config.bboxDilate > 0)
        if dilate:
            begin = bbox.getBegin() - self.bbox_ref.getBegin()
            end = self.bbox_ref.getEnd() - bbox.getEnd()
            dilate = np.min((np.min((begin, end)), self.config.bboxDilate))
            if dilate > 0:
                footprint.dilate(dilate)
                bbox = footprint.getBBox()
        return bbox, dilate

    def _getExposurePsfs(
        self, exposures, source, extras, footprint=None,
        failOnLargeFootprint=False, idx_src=None, idx_mask=None,
    ):
        """Get exposure-PSF pairs as required by `multiprofit.fitutils`.

        Parameters
        ----------
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        source : `lsst.afw.table.SourceRecord`
            A deblended source to fit.
        extras : iterable of `lsst.meas.base.NoiseReplacer` or
                 `multiprofit.object.Exposure` or `lsst.afw.image.Mask`.
            Configuration-dependent extra data for each exposure.
        footprint : `lsst.afw.detection.Footprint`, optional
            The footprint to fit within. Default source.getFootprint().
        failOnLargeFootprint : `bool`, optional
            Whether to return a failure if the fallback (source) footprint
            dimensions also exceed `self.config.maxParentFootprintPixels`.
        idx_src : `int`, optional
            The index of the source in the original catalog.
        idx_mask : `int`, optional
            The index of the parent (blend) in the segmentation mask, if any.

        Returns
        -------
        exposurePsfs : `list` [`tuple`]
            A list of `multiprofit.objects.Exposure`, `multiprofit.objects.PSF`
            pairs.
        footprint : `lsst.afw.detection.Footprint`
            The final footprint used.
        bbox : `lsst.geom.Box2I`
            The bounding box containing the fit region/footprint.
        cen_hst : `list` [`float`]
            x, y pixel coordinates of `cens`.
        """
        if footprint is not None:
            if footprint.getBBox().getArea() > self.config.maxParentFootprintPixels:
                footprint = None
        if footprint is None:
            footprint = source.getFootprint()
        if idx_mask is None:
            idx_mask = idx_src
        fit_hst = self.config.fitHstCosmos
        bbox, dilate = self._getBBoxDilated(footprint)
        spans = None
        area = bbox.getArea()
        if not (area > 0):
            raise RuntimeError(f'Source bbox={bbox} has area={area} !>0')
        elif failOnLargeFootprint and (area > self.config.maxParentFootprintPixels):
            raise RuntimeError(f'Source footprint (fallback) area={area} pix exceeds '
                               f'max={self.config.maxParentFootprintPixels}')
        center = bbox.getCenter()
        # TODO: Implement multi-object fitting/deblending
        # peaks = footprint.getPeaks()
        # nPeaks = len(peaks)
        # isSingle = nPeaks == 1
        exposurePsfs = []
        # TODO: Check total flux first
        if fit_hst:
            wcs_src = next(iter(exposures.values())).getWcs()
            corners, cens = mrCutout.get_corners_src(source, wcs_src)
            exposure, cen_hst, psf = mrCutout.get_exposure_cutout_HST(
                corners, cens, extras, get_inv_var=True, get_psf=True)
            if np.sum(exposure.image > 0) == 0:
                raise RuntimeError('HST cutout has zero positive pixels')
            exposurePsfs.append((exposure, psf))
        else:
            # Try to get all PSFs first, in case it fails
            img_psfs = [exposure.getPsf().computeKernelImage(center) for exposure in exposures.values()]
            for extra, (band, exposure), img_psf in zip(extras, exposures.items(), img_psfs):
                bitmask = 0
                mask = exposure.mask.subset(bbox)
                for bitname in self.mask_names_zero:
                    bitval = mask.getPlaneBitMask(bitname)
                    bitmask |= bitval
                mask = (mask.array & bitmask) != 0

                addition = None
                if self.config.deblendFromDeblendedFits:
                    # To avoid using the noiseReplacer when deblending within
                    # dilated bboxes/footprints, we need to use the
                    # segmentation map to ensure that pixels from other blends
                    # are excluded
                    if dilate:
                        segmap = extra.subset(bbox)
                        mask &= (segmap == 0) | (segmap == idx_mask)
                else:
                    if self.config.disableNoiseReplacer:
                        sources_extra, meta_extra = extra[1], extra[2]
                        idx_extras = meta_extra['idx_add'][idx_src]
                        if idx_extras:
                            addition = afwImage.Image(bbox, dtype='F')
                            for idx_extra in idx_extras:
                                img_extra, bbox_extra = get_spanned_image(
                                    sources_extra[idx_extra].getFootprint()
                                )
                                addition.subset(bbox_extra).array += img_extra
                            addition = addition.array
                        # Unnecessary as long as images are copies, as below
                        #    meta_extra['img_bbox'] = (addition, bbox)
                        # else:
                        #    meta_extra['img_bbox'] = (None, None)
                    else:
                        extra.insertSource(source.getId())
                # TODO: Reassess the float cast is necessary
                img = np.float64(exposure.image.subset(bbox).array)
                if addition is not None:
                    img += addition
                err = np.sqrt(1. / np.float64(exposure.variance.subset(bbox).array))
                if self.config.useSpans:
                    mask_span = afwImage.Mask(bbox)
                    if spans is None:
                        spans = footprint.getSpans()
                    spans.setMask(mask_span, 2)
                    mask |= (mask_span.array == 0)

                err[mask] = 0

                exposure_mpf = mpfObj.Exposure(
                    band=band, image=img,
                    error_inverse=err, is_error_sigma=True, mask_inverse=~mask, use_mask_inverse=True
                )
                psf_mpf = mpfObj.PSF(
                    band, image=img_psf, engine="galsim")
                exposurePsfs.append((exposure_mpf, psf_mpf))

        return exposurePsfs, footprint, bbox, cen_hst if fit_hst else None

    @staticmethod
    def __addExtraField(extra, schema, prefix, name, doc, dtype=np.float64, unit=None, exists=False):
        """Add an extra field to a schema and store it by its short name.

        Parameters
        ----------
        extra : `dict` of `str`
            An input dictionary to store a reference to the new `Key` by its
            field name.
        schema : `lsst.afw.table.Schema`
            An existing table schema to add the field to.
        prefix : `str`
            A prefix for field full name.
        name : `str`
            A short name for the field, which serves as the key for `extra`.
        doc : `str`
            A brief documentation string for the field.
        unit : `str`
            A string convertible to an astropy unit.
        exists : `bool`
            Check if the field already exists and validate it instead of adding
            a new one.

        Notes
        -------
        The new field is added to `schema` and a reference to it is stored in
        `extra`.
        """
        if doc is None:
            doc = ''
        if unit is None:
            unit = ''
        fullname = join_and_filter('_', [prefix, name])
        if exists:
            item = schema.find(fullname)
            field = item.field
            if field.dtype != dtype or field.getUnits() != unit:
                raise RuntimeError(f'Existing field {field} has dtype {field.dtype}!={dtype} and/or units'
                                   f'{field.getUnits()}!={unit}')
            key = item.key
        else:
            key = schema.addField(fullname, type=dtype, doc=doc, units=unit)
        extra[name] = key

    def __addExtraFields(self, extra, schema, prefix=None, exists=False):
        """Add all extra fields for a given model.

        Parameters
        ----------
        extra : `dict` of `str`
            An input dictionary to store reference to the new `Key`s by their
            field names.
        schema : `lsst.afw.table.Schema`
            An existing table schema to add the field to.
        prefix : `str`, optional
            A string such as a model name to prepend to each field name;
            default None.
        exists : `bool`
            Check if the fields already exist and validate them instead of
            adding new ones.

        Returns
        -------
        The new fields are added to `schema` and reference to them are stored
        in `extra`.
        """
        if self.config.outputChisqred:
            self.__addExtraField(extra, schema, prefix, 'chisqred', 'reduced chi-squared of the best fit',
                                 exists=exists)
        if self.config.outputLogLikelihood:
            self.__addExtraField(extra, schema, prefix, 'loglike', 'log-likelihood of the best fit',
                                 exists=exists)
        if self.config.outputRuntime:
            self.__addExtraField(extra, schema, prefix, 'time', 'model runtime excluding setup', unit='s',
                                 exists=exists)
        self.__addExtraField(extra, schema, prefix, 'nEvalFunc', 'number of objective function evaluations',
                             exists=exists)
        self.__addExtraField(extra, schema, prefix, 'nEvalGrad', 'number of Jacobian evaluations',
                             exists=exists)

    @staticmethod
    def __fitModel(model, exposurePsfs, modeller=None, sources=None, resetPsfs=False, **kwargs):
        """Fit a moments-initialized model to sources in exposure-PSF pairs.

        Parameters
        ----------
        model : `multiprofit.objects.Model`
            A MultiProFit model to fit.
        exposurePsfs : `iterable` of tuple of
                (`multiprofit.objects.Exposure`, `multiprofit.objects.PSF`)
            An iterable of exposure-PSF pairs to fit.
        modeller : `multiprofit.objects.Modeller`, optional
            A MultiProFit modeller to use to fit the model; default creates a
            new modeller.
        sources : `list` [`dict`]
            A list of sources specified as a dict of values by parameter name,
            with flux a dict by filter.
        resetPsfs : `bool`, optional
            Whether to set the PSFs to None and thus fit a model with no PSF
            convolution.
        kwargs
            Additional keyword arguments to pass to
            `multiprofit.fitutils.fit_model`.

        Returns
        -------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_model, if no error
            occurs.
        """
        if sources is None:
            sources = [{}]
        # Set the PSFs to None in each exposure to skip convolution
        exposures_no_psf = {}
        for exposure, _ in exposurePsfs:
            if resetPsfs:
                exposure.psf = None
            exposures_no_psf[exposure.band] = [exposure]
        model.data.exposures = exposures_no_psf
        params_free = [src.get_parameters(free=True) for src in model.sources]
        n_sources = len(sources)
        if n_sources == 1 and ('sigma_x' not in sources[0]):
            fluxes, sources[0], _, _, _ = mpfFit.get_init_from_moments(
                exposurePsfs, cenx=sources[0].get('cenx', 0), ceny=sources[0].get('ceny', 0))
            sources[0]['flux'] = fluxes
        if n_sources != len(params_free):
            raise ValueError(f'len(sources)={n_sources} != len(model.sources)={len(model.sources)}')
        for values, params in zip(sources, params_free):
            fluxes = values.get('flux', {})
            for param in params:
                value = fluxes.get(param.band, 1) if isinstance(param, mpfObj.FluxParameter) else \
                    values.get(param.name)
                if param.name.startswith('cen'):
                    param.limits.upper = exposurePsfs[0][0].image.shape[param.name[-1] == 'x']
                if value is not None:
                    param.set_value(value)
        result, _ = mpfFit.fit_model(model=model, modeller=modeller, **kwargs)
        return result

    @pipeBase.timeMethod
    def __fitSource(self, source, exposures, extras, children_cat=None,
                    footprint=None, failOnLargeFootprint=False, row=None,
                    usePriorShapeDefault=False, priorCentroidSigma=None, mag_prior=None,
                    backgroundPriors=None, fields=None, idx_src=None, children_src=None, skip_fit=False,
                    kwargs_moments=None,
                    **kwargs):
        """Fit a single deblended source with MultiProFit.

        Parameters
        ----------
        source : `lsst.afw.table.SourceRecord`
            A deblended source to fit.
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        extras : iterable of `lsst.meas.base.NoiseReplacer`
                 or `multiprofit.object.Exposure`
            An iterable of NoiseReplacers that will insert the source into
            every exposure, or a tuple of HST exposures if fitting HST data.
        children_cat : iterable of `lsst.afw.table.SourceRecord`
            Child sources with existing measurements to use for deblending.
        footprint : `lsst.afw.detection.Footprint`, optional
            The footprint to fit within. Default source.getFootprint().
        failOnLargeFootprint : `bool`, optional
            Whether to return a failure if the fallback (source) footprint
            dimensions also exceed `self.config.maxParentFootprintPixels`.
        usePriorShapeDefault : `bool`, optional
            Whether to use the default MultiProFit shape prior.
        priorCentroidSigma : `float`, optional
            The sigma on the Gaussian centroid prior.
        mag_prior: `float`, optional
            The magnitude for setting magnitude-dependent priors.
            A None value disables such priors.
        backgroundPriors: `dict` [`str`, `tuple`], optional
            Dict by band of 2-element tuple containing background level prior
            mean and sigma.
        children_src : iterable of `lsst.afw.table.SourceRecord`
            Child sources used to determine original bounding boxes used in
            fitting deblended sources and for dilation.
        skip_fit : `bool`
            Whether to skip fitting, setting up and returning a dummy result.
        kwargs_moments : `dict`
            Additional arguments to pass to the kwargs_moments param of
            `multiprofit.fitutils.fit_galaxy_exposures`.
        **kwargs
            Additional keyword arguments to pass to
            `multiprofit.fitutils.fit_galaxy_exposures`.

        Returns
        -------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures,
            if no error occurs.
        error : `Exception`
            The first exception encountered while fitting, if any.
        deblended : `bool`
            Whether the method inserted the source using the provided
            noiseReplacers.

        Notes
        -----
        For deblending in typical usage, children_src should be derived from
        the input catalog used for initial deblended source fits
        ("deepCoadd_ref") whereas children_cat should be the resume catalog
        with MultiProFit measurements, which likely contains band-dependent
        footprints that may not match those from the deepCoadd_ref.
        """
        if kwargs_moments is None:
            kwargs_moments = {}
        has_children = children_src is not None
        n_children = len(children_cat) if has_children else 0
        deblend_no_init = self.config.deblend and has_children
        if deblend_no_init and len(self.modelSpecs) > 0:
            raise RuntimeError("Can only deblend with gausspx-nopsf model")
        fit_hst = self.config.fitHstCosmos
        pixel_scale = pixel_scale_hst if fit_hst else pixel_scale_hsc
        deblended = False
        if children_src is None:
            children_src = children_cat

        try:
            exposurePsfs, footprint, bbox, cen_hst = self._getExposurePsfs(
                exposures, source, extras, footprint=footprint, failOnLargeFootprint=failOnLargeFootprint,
                idx_src=idx_src,
            )
            if not self.config.deblendFromDeblendedFits and not self.config.fitHstCosmos:
                deblended = True
            cen_src = source.getCentroid()
            begin = bbox.getBegin()
            cens = cen_hst if fit_hst else cen_src - begin
            if self.config.fitGaussian:
                rho_min, rho_max = -0.9, 0.9
                if fit_hst and deblend_no_init:
                    # TODO: Use wcs_hst/src instead
                    # wcs_hst = exposure.meta['wcs']
                    scale_x = pixel_scale_hsc / pixel_scale_hst
                    scales = np.array([scale_x, scale_x])
                sources = [{}]
                if deblend_no_init or self.config.useSdssShape:
                    for child in (children_cat if deblend_no_init else [source]):
                        cen_child = ((cen_hst + scales*(child.getCentroid() - cen_src)) if fit_hst else (
                            child.getCentroid() - begin)) if deblend_no_init else cens
                        ellipse = {'cenx': cen_child[0], 'ceny': cen_child[1]}
                        cov = child['base_SdssShape_xy']
                        if np.isfinite(cov):
                            sigma_x, sigma_y = (np.sqrt(child[f'base_SdssShape_{ax}']) for ax in ['xx', 'yy'])
                            if sigma_x > 0 and sigma_y > 0:
                                ellipse['rho'] = np.clip(cov / (sigma_x * sigma_y), rho_min, rho_max)
                                ellipse['sigma_x'], ellipse['sigma_y'] = sigma_x, sigma_y
                        sources.append(ellipse)

            bands = [item[0].band for item in exposurePsfs]
            params_prior = {}
            if usePriorShapeDefault:
                size_mean, size_stddev = get_hst_size_prior(mag_prior if np.isfinite(mag_prior) else np.Inf)
                size_mean_stddev = (size_mean - np.log10(pixel_scale), size_stddev)
                params_prior['shape'] = {
                    True: {
                        'size_mean_std': (self.config.psfHwhmShrink, self.config.gaussianSizePriorSigma),
                        'size_log10': False,
                        'axrat_params': (-0.1, 0.5, 1.1),
                    },
                    False: {
                        'size_mean_std': size_mean_stddev,
                        'size_log10': True,
                        'axrat_params': (-0.3, 0.2, 1.2),
                    }
                }
            if priorCentroidSigma is not None:
                if not priorCentroidSigma > 0:
                    raise ValueError(f'Invalid priorCentroidSigma={priorCentroidSigma} !>0')
                for coord in ('cenx', 'ceny'):
                    params_prior[coord] = {
                        True: {'stddev': priorCentroidSigma},
                        False: {'stddev': priorCentroidSigma},
                    }
            if backgroundPriors:
                priors_background = {}
                for idx_band, (band, (bg_mean, bg_sigma)) in enumerate(backgroundPriors.items()):
                    if bg_sigma is None:
                        exposure = exposurePsfs[idx_band][0]
                        if exposure.band != band:
                            raise RuntimeError(f'exposure.band={exposure.band}!=band={band} setting bg prior')
                        err = exposure.error_inverse
                        pix_good = err > 0
                        bg_sigma = np.nanmedian(np.sqrt(1/err[pix_good]))/np.sqrt(np.sum(pix_good))
                    if not bg_sigma > 0:
                        raise ValueError(f'Non-positive bg_sigma={bg_sigma}')
                    priors_background[band] = {'mean': bg_mean, 'stddev': bg_sigma}
                params_prior['background'] = priors_background

            deblend_from_fits = self.config.deblendFromDeblendedFits and children_cat is not None
            if not deblend_from_fits:
                skip_fit = skip_fit or self.config.plotOnly
                # No good reason to allow default centroids
                kwargs_moments_in = kwargs_moments.copy()
                kwargs_moments_in['cenx'], kwargs_moments_in['ceny'] = cens
                results = mpfFit.fit_galaxy_exposures(
                    exposurePsfs, bands, self.modelSpecs, plot=self.config.plot, print_exception=True,
                    kwargs_moments=kwargs_moments_in, fit_background=self.config.fitBackground,
                    psf_shrink=self.config.psfHwhmShrink, prior_specs=params_prior,
                    skip_fit=skip_fit, skip_fit_psf=skip_fit, background_sigma_add=(
                        self.config.backgroundSigmaAdd if self.config.fitBackground else None),
                    replace_data_by_model=self.config.replaceDataByModel, loggerPsf=kwargs.get('logger'),
                    **kwargs,
                )
                if (not self.config.plotOnly) and self.config.fitGaussian:
                    n_sources = len(sources)
                    name_modeltype = f'gausspx{self.postfix_nopsf}'
                    name_model_full = f'{name_modeltype}_{n_sources}'
                    if name_model_full in self.models:
                        model = self.models[name_model_full]
                    else:
                        bands = [x[0].band for x in exposurePsfs]
                        model = mpfFit.get_model(
                            {band: 1 for band in bands}, "gaussian:1", (1, 1), slopes=[0.5],
                            engine='galsim',
                            engineopts={
                                'use_fast_gauss': True,
                                'drawmethod': mpfObj.draw_method_pixel['galsim']
                            },
                            name_model=name_model_full
                        )
                        for _ in range(n_sources-1):
                            model.sources.append(copy.deepcopy(model.sources[0]))
                        # Probably don't need to cache anything more than this
                        if n_sources < 10:
                            self.models[name_model_full] = model
                    self.modeller.model = model
                    result = self.__fitModel(
                        model, exposurePsfs, modeller=self.modeller, sources=sources, resetPsfs=True,
                        plot=self.config.plot and len(self.modelSpecs) == 0,
                    )
                    results['fits']['galsim'][name_modeltype] = mpfFit.ModelFits(
                        fits=[result], modeltype='gaussian:1',
                    )
                    results['models']['gaussian:1'] = model
            else:
                results = {}
                kwargs_fit = {
                    'do_linear_only': n_children > self.config.deblendNonlinearMaxSources,
                    'replace_data_by_model': self.config.replaceDataByModel,
                }
                skip_fit_psf = True
                # Check if any PSF fit failed (e.g. because a large blend was
                # skipped entirely) and redo if so
                values_init_psf = self.modelSpecs[0].values_init_psf
                for band, params_psf in values_init_psf.items():
                    for name_param, value_param in params_psf:
                        if not np.isfinite(value_param):
                            skip_fit_psf = False
                            break
                # Get the child footprints once instead of for each model
                exposurePsfs_cs = [None] * n_children
                bbox_cs = [None] * n_children
                for idx_c, child_src in enumerate(children_src):
                    exposurePsfs_cs[idx_c], _, bbox_cs[idx_c], *_ = self._getExposurePsfs(
                        exposures, child_src, extras,
                        footprint=source.getFootprint() if self.config.useParentFootprint else None,
                        failOnLargeFootprint=failOnLargeFootprint
                    )
                # Check if a model fit failed (e.g. as above) and zero init
                # params. Needs update if background model becomes non-linear
                for modelSpec in self.modelSpecs:
                    values_init_model = modelSpec.values_init
                    if not skip_fit_psf:
                        del modelSpec.values_init_psf
                    for name_param, value_param in values_init_model:
                        if not np.isfinite(value_param):
                            values_init_model = [
                                (name_p, self.__getParamValueDefault(name_p))
                                for name_p, _ in values_init_model
                            ]
                            modelSpec.values_init = values_init_model
                            break
                kwargs_moments_in = kwargs_moments.copy()
                kwargs_moments_in['cenx'], kwargs_moments_in['ceny'] = cens
                results_parent = mpfFit.fit_galaxy_exposures(
                    exposurePsfs, bands, [self.modelSpecs[0]], plot=False, print_exception=True,
                    kwargs_moments=kwargs_moments_in, fit_background=self.config.fitBackground,
                    psf_shrink=self.config.psfHwhmShrink, skip_fit=True, skip_fit_psf=skip_fit_psf,
                    loggerPsf=kwargs.get('logger'),
                    **kwargs
                )
                filters = [x[0].band for x in exposurePsfs]
                if not skip_fit_psf:
                    self.__setFieldsPsf(
                        results_parent, fields["psf"], fields["psf_extra"], row, filters, key_index=1,
                    )
                fields_base = fields["base"]
                values_init = {}
                # Iterate through each model and do a simultaneous re-fit
                for modelSpec in self.modelSpecsDict.values():
                    name_modeltype = modelSpec.model
                    name_model = modelSpec.name
                    fields_base_model = fields_base[name_model]

                    inittype_split = modelSpec.inittype.split(';')
                    # TODO: Come up with a better way to determine if
                    # initializing from other models
                    if inittype_split[0] in self.modelSpecsDict:
                        modelSpecs = [self.modelSpecsDict[name_init] for name_init in inittype_split]
                        modelSpecs.append(modelSpec)
                    else:
                        modelSpecs = [modelSpec]

                    _, models = mpfFit.fit_galaxy(
                        exposurePsfs, modelSpecs, fit_background=self.config.fitBackground, skip_fit=True,
                        background_sigma_add=(self.config.backgroundSigmaAdd if self.config.fitBackground
                                              else None),
                    )
                    model = models[name_modeltype]
                    sources_bg = []
                    for source_model in model.sources:
                        if isinstance(source_model, mpfObj.Background):
                            if not self.config.fitBackground:
                                raise TypeError("Model should not have Background objects with "
                                                "fitBackground=False")
                            else:
                                sources_bg.append(source_model)
                    model.sources = []
                    params_free = {}
                    values_init[name_model] = {}
                    n_good = 0

                    for idx_c, child_cat in enumerate(children_cat):
                        is_good = True
                        values_init_model = []
                        for name_field, key in fields_base_model:
                            value_field = child_cat[key]
                            if not np.isfinite(value_field):
                                is_good = False
                                values_init_model = None
                                break
                            values_init_model.append((name_field, value_field))
                        values_init[name_model][idx_c] = values_init_model
                        # Re-initialize the child model and transfer its
                        # sources to the full model
                        if is_good:
                            n_good += 1
                            exposurePsfs_input = []
                            for (exposure_p, psf_p), (exposure_c, psf_c) in zip(
                                    exposurePsfs, exposurePsfs_cs[idx_c]):
                                exposure_c.psf = exposure_p.psf
                                exposurePsfs_input.append((exposure_c, psf_p))

                            for specs in modelSpecs:
                                specs['values_init'] = values_init[specs['name']][idx_c]

                            _, models = mpfFit.fit_galaxy(
                                exposurePsfs_input, modelSpecs, fit_background=self.config.fitBackground,
                                skip_fit=True
                            )
                            # Another double entendre; it never ends.
                            model_child = models[name_modeltype]
                            bbox_min = bbox.getMin()
                            bbox_min_c = bbox_cs[idx_c].getMin()
                            # All centroid parameters need adjusting, even if
                            # they're fixed
                            if bbox_min != bbox_min_c:
                                offset_x, offset_y = bbox_min_c - bbox_min
                                params_all = model_child.get_parameters(free=True, fixed=True)
                                params_free_i = []
                                for param in params_all:
                                    if param.name.startswith('cen'):
                                        is_x = param.name[3] == 'x'
                                        offset = offset_x if is_x else offset_y
                                        if offset != 0:
                                            param.limits.upper = bbox.width if is_x else bbox.height
                                            value = param.get_value()
                                            param.set_value(value + offset)
                                    if not param.fixed:
                                        params_free_i.append(param)
                                params_free[child_cat] = params_free_i
                            else:
                                params_free[child_cat] = model_child.get_parameters(free=True, fixed=False)

                            # Ensure no duplicate background sources added
                            for source_mpf in models[name_modeltype].sources:
                                if isinstance(source_mpf.modelphotometric.components[-1], mpfObj.Background):
                                    if not self.config.fitBackground:
                                        raise TypeError("Model should not have Background objects with "
                                                        "fitBackground=False")
                                else:
                                    model.sources.append(source_mpf)
                        else:
                            params_free[child_cat] = None

                    # Only proceed if at least one child actually has good
                    # (all finite) parameter values.
                    # TODO: consider if it should be n_good > 1
                    if n_good > 0:
                        if self.config.plotOnly:
                            result_model = None
                            if self.config.plot:
                                model.evaluate(plot=True)
                        else:
                            result_model, _ = mpfFit.fit_model(model, plot=self.config.plot,
                                                               kwargs_fit=kwargs_fit)
                            self.__setExtraFields(fields["base_extra"][name_model], source, result_model)

                            for child, params_free_c in params_free.items():
                                if params_free_c is not None:
                                    if len(params_free_c) != len(fields_base_model):
                                        raise RuntimeError(
                                            f'len(params_free_c={params_free_c})={len(params_free_c)} != '
                                            f'len(fields_base_model={fields_base_model})'
                                            f'={len(fields_base_model)}'
                                        )
                                    for param, (name, key) in zip(params_free_c, fields_base_model):
                                        child[key] = param.get_value()
                    else:
                        result_model = None
                    results[name_model] = result_model
                results = {'fits': {'galsim': results}}
            if self.config.plot:
                plt.show()
            return results, None, deblended
        except Exception as e:
            if self.config.printTrace:
                traceback.print_exc()
            if self.config.plot:
                n_exposures = len(exposures)
                if n_exposures > 1:
                    fig, axes = plt.subplots(1, n_exposures)
                    for idx, (band, exposure) in enumerate(exposures.items()):
                        axes[idx].imshow(exposure.image)
                        axes[idx].set_title(f'{band} [{idx}/{n_exposures}]')
                else:
                    plt.figure()
                    band, exposure = list(exposures.items())[0]
                    plt.imshow(exposure.image)
                    plt.title(band)
            return None, e, False

    def __getSchema(self, filters, results, schema):
        """Get a catalog and a dict containing the keys of extra fields.

        Parameters
        ----------
        filters : iterable of `str`
            Names of bandpass filters for filter-dependent fields.
        results : `dict`
            Results structure as returned by `__fitSource`.
        schema : `lsst.afw.table.Schema`

        Returns
        -------
        schema : `lsst.afw.table.Schema`
            A new Schema with extra fields.
        fields : `dict` [`str`, `dict`]
            A dict of dicts, keyed by the field type. The values may contain
            further nested dicts e.g. those keyed by filter for PSF fit-related
            fields.
        """
        resume = self.config.resume

        if not resume:
            mapper = MultiProFitTask._getMapper(schema)
        keys_extra = {
            'runtimeKey': {'name': f'{self.prefix}{self.keynames["runtimeKey"]}', 'dtype': np.float64,
                           'doc': 'Source fit CPU runtime', 'unit': 's'},
            'failFlagKey': {'name': f'{self.prefix}{self.keynames["failFlagKey"]}', 'dtype': 'Flag',
                            'doc': 'MultiProFit general failure flag'},
        }
        fields_attr = {}
        for name_attr, specs in keys_extra.items():
            name_field = specs.get('name', f'{self.prefix}{name_attr}')
            self.__addExtraField(
                fields_attr, schema if resume else mapper.editOutputSchema(),
                prefix=None, name=name_field, doc=specs.get('doc', ''),
                dtype=specs.get('dtype'), unit=specs.get('unit'), exists=resume)
            setattr(self, name_attr, fields_attr[name_field])
        if not resume:
            schema = mapper.getOutputSchema()

        fields = {key: {} for key in ["base", "base_extra", "psf", "psf_extra", "measmodel"]}
        # Set up the fields for PSF fits, which are independent per filter
        for idxBand, band in enumerate(filters):
            prefix = f'{self.prefix}{self.prefix_psf}{band}'
            resultsPsf = results['psfs'][idxBand]['galsim']
            fields["psf"][band] = {}
            fields["psf_extra"][band] = defaultdictNested()
            for name, fit in resultsPsf.items():
                fit = fit['fit']
                namesAdded = defaultdict(int)
                keyList = []
                for nameParam in fit['name_params']:
                    namesAdded[nameParam] += 1
                    fullname, doc, unit = self.__getParamFieldInfo(
                        f'{nameParam}{"Frac" if nameParam == "flux" else ""}',
                        f'{prefix}_c{namesAdded[nameParam]}_')
                    if resume:
                        key = schema.find(fullname).key
                    else:
                        key = schema.addField(fullname, doc=doc, units=unit, type=np.float64)
                    keyList.append(key)
                fields["psf"][band][name] = keyList
                self.__addExtraFields(fields["psf_extra"][band][name], schema, prefix, exists=resume)

        # Setup field names for source fits, which may have fluxes in multiple
        # bands if run in multi-band. Either way, flux parameters should
        # contain a band name.
        for name, result in results['fits']['galsim'].items():
            prefix = f'{self.prefix}{name}'
            fit = result.fits[0]
            namesAdded = defaultdict(int)
            keyList = []
            bands = [f'{x.band}_' if hasattr(x, 'band') else '' for x, fixed in zip(
                fit['params'], fit['params_allfixed']) if not fixed]
            for nameParam, postfix in zip(fit['name_params'], bands):
                nameParamFull = f'{nameParam}{postfix}'
                namesAdded[nameParamFull] += 1
                fullname, doc, unit = self.__getParamFieldInfo(
                    nameParam, f'{prefix}_c{namesAdded[nameParamFull]}_{postfix}')
                if resume:
                    key = schema.find(fullname).key
                else:
                    key = schema.addField(fullname, doc=doc, units=unit, type=np.float64)
                keyList.append(key)
            fields["base"][name] = keyList
            fields["base_extra"][name] = defaultdictNested()
            self.__addExtraFields(fields["base_extra"][name], schema, prefix, exists=resume)

        if self.config.computeMeasModelfitLikelihood:
            for name in self.meas_modelfit_models:
                self.__addExtraField(fields["measmodel"], schema, f"{self.prefix}measmodel_loglike", name,
                                     f'MultiProFit log-likelihood for meas_modelfit {name} model',
                                     exists=resume)

        return schema, mapper, fields

    @staticmethod
    def __getFieldParamName(field):
        """Get the standard MultiProFit parameter name for a catalog field.

        Parameters
        ----------
        field : `str`
            The name of the catalog field.

        Returns
        -------
        name : `str`
            The standard MultiProFit parameter name.
        """
        field_split = field.split('_')
        final = field_split[-1]
        if final.endswith('Flux') or final.endswith('Flux'):
            return 'flux'
        elif final == 'x' or final == 'y':
            if (len(field_split) > 1) and (field_split[-2] == 'sigma'):
                return f'sigma_{final}'
        return final

    @staticmethod
    def __getParamFieldInfo(nameParam, prefix=None):
        """Return standard information about a MultiProFit parameter by name.

        Parameters
        ----------
        nameParam : `str`
            The name of the parameter.
        prefix : `str`
            A prefix for the full name of the parameter; default None ('').

        Returns
        -------
        name_full : `str`
            The full name of the parameter including prefix, remapping 'flux'
            to 'instFlux'.
        doc : `str`
            The default docstring for the parameter, if any; '' otherwise.
        unit : `str`
            The default unit of the parameter, if any; '' otherwise.
        """
        name_full = f'{prefix if prefix else ""}{"instFlux" if nameParam == "flux" else nameParam}'
        doc, unit = MultiProFitTask.params_multiprofit.get(nameParam, ('', ''))
        return name_full, doc, unit

    @staticmethod
    def __getParamValueDefault(name_param):
        """Get a reasonable default value for a parameter by its name.

        Parameters
        ----------
        name_param : `str`:
            The name of the parameter.

        Returns
        -------
        value : `default`
            The MultiProFit-specified default value.
        """
        return mpfFit.get_param_default(
            MultiProFitTask.__getFieldParamName(name_param),
            return_value=True,
            is_value_transformed=True
        )

    @staticmethod
    def _getSegmentationMap(bbox, sources):
        """Make an exposure segmentation map for each blend in a catalog.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            A bounding box to build the segmentation map with.
        sources : `lsst.afw.table.SourceCatalog`
            The source catalog to fill segments for.

        Returns
        -------
        mask : `lsst.afw.image.Mask`
            A segmentation map where 0 = sky and each source from `sources`
            has its segment value as 1 + its row index in `sources`.
        """
        mask = afwImage.Mask(bbox)
        for idx, src in enumerate(sources):
            if src['deblend_nChild'] > 0:
                spans = src.getFootprint().getSpans()
                spans.setMask(mask, idx + 1)
        return mask

    def getNamePsfModel(self):
        return f'gaussian:{self.config.gaussianOrderPsf}_pixelated'

    @staticmethod
    def __setExtraField(extra, row, fit, name, nameFit=None, index=None):
        """Set the value of an extra field for a given row.

        Parameters
        ----------
        extra : container [`str`]
            An input container permitting retrieval of values by string keys.
        row : container [`str`]
            An output container permitting assignment by string keys.
        fit : `dict` [`str`]
            A fit result containing a value for `name` or `nameFit`.
        name : `str`
            The name of the field in `row`.
        nameFit : `str`, optional
            The name of the field in `fit`; default `name`.
        index : `int`, optional
            The index of the value in `fit`, if it is a container.
            Ignored if None (default).
        Returns
        -------
        None
        """
        if nameFit is None:
            nameFit = name
        value = fit[nameFit]
        if index is not None:
            value = value[index]
        row[extra[name]] = value

    def _parseCatalogFields(self, catalog, validate_psf=True, add_keys=False):
        """Get the keys for MultiProFit fields in an existing catalog.

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            The catalog to parse.
        validate_psf : `bool`, optional
            Whether to validate the retrieved PSF fields against expected.
        add_keys: `bool`, optional
            Whether to assign keys stored individually as attributes.

        Returns
        -------
        fields_mpf: `dict` [`str`, `list` [`str`]]
            A dictionary containing a list of parameter tuples, keyed by type.
            Each tuple contains:
            - name [`str`] : the short field name
            - key [] : the catalog (schema) field key

        Notes
        -----
        The return value is similar to that from __getCatalog.
        """
        fields_mpf = {f'{pre}{post}': defaultdict(dict if is_extra else list) for pre in ('base', 'psf')
                      for post, is_extra in (('', False), ('_extra', True))}
        prefixes = (self.prefix, f'{self.prefix}{self.prefix_psf}')
        prefix_psf = prefixes[True]
        schema = catalog.schema

        for name_field in schema.getOrderedNames():
            if name_field.startswith(self.prefix):
                is_psf = name_field.startswith(prefix_psf)
                prefix = prefixes[is_psf]
                name_field = name_field.split(prefix, 1)[1]
                key, name_field = name_field.split('_', 1)
                suffix = name_field.split('_')[-1]
                type_field = "psf" if is_psf else "base"
                is_extra = suffix in self.params_extra
                if is_extra:
                    type_field = f"{type_field}_extra"
                key_cat = schema.find(f'{prefix}{key}_{name_field}').key
                if is_extra:
                    fields_mpf[type_field][key][name_field] = key_cat
                else:
                    fields_mpf[type_field][key].append((name_field, key_cat))

        if validate_psf:
            fields_psf = ['c1_cenx', 'c1_ceny']
            params_shape = ('sigma_x', 'sigma_y', 'rho')
            for idx_comp in range(1, 1 + self.config.gaussianOrderPsf):
                if idx_comp != self.config.gaussianOrderPsf:
                    fields_psf.append(f'c{idx_comp}_fluxFrac')
                fields_psf.extend([f'c{idx_comp}_{name_param}' for name_param in params_shape])
            for band, fields_found in fields_mpf["psf"].items():
                fields_found_names = [field[0] for field in fields_found]
                if fields_found_names != fields_psf:
                    raise RuntimeError(f'PSF fields_found={fields_found_names} != expected fields_psf='
                                       f'{fields_psf}')

        name_psfmodel = self.getNamePsfModel()
        for band in fields_mpf["psf"].keys():
            fields_mpf["psf"][band] = {name_psfmodel: fields_mpf["psf"][band]}
            fields_mpf["psf_extra"][band] = {name_psfmodel: fields_mpf["psf_extra"][band]}

        if add_keys:
            for name_attr, name_field in self.keynames.items():
                setattr(self, name_attr, schema.find(f'{self.prefix}{name_field}').key)

        return fields_mpf

    def __setExtraFields(self, extra, row, fit):
        """Set the values of extra fields, like fit metadata.

        Parameters
        ----------
        extra : container [`str`]
            An input container permitting retrieval of values by string keys.
        row : container [`str`]
            An output container permitting assignment by string keys.
        fit : `dict` [`str`]
            A fit result containing the required values.

        Returns
        -------
        None
        """
        if self.config.outputChisqred:
            self.__setExtraField(extra, row, fit, 'chisqred')
        if self.config.outputLogLikelihood:
            self.__setExtraField(extra, row, fit, 'loglike', nameFit='likelihood', index=0)
        if self.config.outputRuntime:
            self.__setExtraField(extra, row, fit, 'time')
        self.__setExtraField(extra, row, fit, 'nEvalFunc', nameFit='n_eval_func')
        self.__setExtraField(extra, row, fit, 'nEvalGrad', nameFit='n_eval_grad')

    def __setFieldsSource(self, results, fieldsBase, fieldsExtra, row):
        """Set fields for a source's fit parameters.

        Parameters
        ----------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures,
            if no error occurs.
        fieldsBase : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for base fields (i.e. fit
            parameters), keyed by model name.
        fieldsExtra : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for extra fields, keyed by
            model name.
        row : container [`str`]
            An output container permitting assignment by string keys.

        Returns
        -------
        None
        """
        for idxfit, (name, result) in enumerate(results['fits']['galsim'].items()):
            if result is not None:
                fits = result.fits
                if fits is not None and len(fits) > 0:
                    fit = fits[-1]
                    if isinstance(fit, tuple):
                        if not (len(fit) > 0):
                            raise RuntimeError(f'Model={name} fit result tuple has len<1')
                        elif not isinstance(fit[0], Exception):
                            raise RuntimeError(f'Model={name} type(fit[0])={type(fit[0])} unrecognized')
                        else:
                            raise FitFailedError(f'Model={name} fit failed: {fit[0]}')
                    elif not isinstance(fit, dict):
                        raise RuntimeError(f'Model={name} fit type(fit)={type(fit)} not recognized')
                    values = [x for x, fixed in zip(fit['params_bestall'], fit['params_allfixed'])
                              if not fixed]
                    for value, key in zip(values, fieldsBase[name]):
                        row[key] = value
                    self.__setExtraFields(fieldsExtra[name], row, fit)

    def __setFieldsPsf(self, results, fieldsBase, fieldsExtra, row, key_index=None):
        """Set fields for a source's PSF fit parameters.

        Parameters
        ----------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures,
            if no error occurs.
        fieldsBase : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for base fields (i.e. fit
            parameters), keyed by filter name.
        fieldsExtra : `dict` [`str`, `dict` [`str`]]
            A dict of dicts of field keys by name for extra fields, keyed by
            filter name.
        row : container [`str`]
            An output container permitting assignment by string keys.

        Returns
        -------
        None
        """
        results_psfs = results.get('psfs', None)
        if results_psfs:
            for idxBand, band in enumerate(self.config.bands_fit):
                resultsPsf = results_psfs[idxBand]['galsim']
                for name, fit in resultsPsf.items():
                    fit = fit['fit']
                    values = [x for x, fixed in zip(fit['params_bestall'], fit['params_allfixed'])
                              if not fixed]
                    for value, key in zip(values, fieldsBase[band][name]):
                        row[key[key_index] if key_index is not None else key] = value
                    self.__setExtraFields(fieldsExtra[band][name], row, fit)

    def __setFieldsMeasmodel(self, exposures, model, source, fieldsMeasmodel, row):
        """Set fields for a source's meas_modelfit-derived fields.

        These fields include likelihoods for the meas_modelfit models
        parameters as computed by MultiProFit, if
        `self.config.computeMeasModelfitLikelihood` is True.

        Parameters
        ----------
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        model : `multiprofit.objects.Model`
            A MultiProFit model to compute likelihoods with.
        source : `lsst.afw.table.SourceRecord`
            A deblended source to build meas_modelfit models for.
        fieldsMeasmodel : `dict` [`str`, container [`str`]]
            A dict of output fields, keyed by model name.
        row : container [`str`]
            An output container permitting assignment by string keys.

        Returns
        -------
        None
        """
        configMeasModelfit = CModelConfig()
        measmodels = {key: {} for key in self.meas_modelfit_models}
        for band, exposure in exposures.items():
            _, measmodels['dev'][band], measmodels['exp'][band], measmodels['cmodel'][band] = \
                buildCModelImages(exposure, source, configMeasModelfit)
        # Set the values of meas_modelfit model likelihood fields
        for measmodel_type, measmodel_images in measmodels.items():
            likelihood = 0
            for band, exposure in exposures.items():
                likelihood += model.get_exposure_likelihood(
                    model.data.exposures[band][0], measmodel_images[band].array)[0]
            likelihood = {measmodel_type: likelihood}
            self.__setExtraField(fieldsMeasmodel, row, likelihood, measmodel_type)

    def __setRow(self, results, fields, row, exposures, source):
        """Set all necessary field values for a given source's row.

        Parameters
        ----------
        results : `dict`
            The results returned by multiprofit.fitutils.fit_galaxy_exposures,
            if no error occurs.
        fields : `dict` [`str`, `dict`]
            A dict of dicts of field keys by name for base fields (i.e. fit
            parameters), keyed by filter name.
        row : container [`str`]
            An output container permitting assignment by string keys.
        exposures : `dict` [`str`, `lsst.afw.image.Exposure`]
            A dict of Exposures to fit, keyed by filter name.
        source : `lsst.afw.table.SourceRecord`
            A deblended source to build meas_modelfit models for.
        """
        self.__setFieldsPsf(results, fields["psf"], fields["psf_extra"], row)
        self.__setFieldsSource(results, fields["base"], fields["base_extra"], row)
        if self.config.computeMeasModelfitLikelihood:
            model = results['models'][self.modelSpecs[0].model]
            self.__setFieldsMeasmodel(exposures, model, source, fields["measmodel"], row)

    def fit(
        self,
        catexps: Iterable[fitMb.CatalogExposure],
        cat_ref: afwTable.SourceCatalog,
        logger: logging.Logger = None,
        **kwargs,
    ) -> afwTable.SourceCatalog:
        """Fit a catalog of sources with MultiProFit.

        Each source has its PSF fit with a configureable Gaussian mixture PSF
        model and then fits a sequence of different models, some of which can
        be configured to be initialized from previous fits.

        Parameters
        ----------
        catexps : `Iterable` [`lsst.pipe.tasks.fit_multiband.CatalogExposure`]
            A list of CatalogExposures to fit.
        logger : `logging.Logger`, optional
            A Logger to log output; default logging.getLogger(__name__).
        cat_ref : `lsst.afw.table.SourceCatalog`, optional
            A source catalog to override filter-specific catalogs provided in
            `data`, e.g. deepCoadd_ref. Default None.
        **kwargs
            Additional keyword arguments to pass to `__fitSource`.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            A new catalog containing all of the fields from `sources` and those
            generated by MultiProFit.
        results : `dict`
            A results structure as returned by mpfFit.fit_galaxy_exposures()
            for the first successfully fit source

        Notes
        -----
        See `MultiProFitConfig` for information on model configuration
        parameters.

        Plots can be generated on success (with MultiProFit defaults), or on
        failure, in which case only the images themselves are shown. Tracebacks
        are suppressed on failure by default.
        """
        logger_is_none = logger is None
        if logger_is_none:
            logging.basicConfig()
            logger = logging.getLogger(__name__)
            logger.level = logging.INFO
        if (self.config.resume or self.config.deblendFromDeblendedFits) and \
                not os.path.isfile(self.config.filenameOut):
            raise FileNotFoundError(f"Can't resume or deblendFromDeblendedFits from non-existent file"
                                    f"={self.config.filenameOut}")
        if self.config.resume and self.config.deblendFromDeblendedFits and not os.path.isfile(
                self.config.filenameOutDeblend):
            raise FileNotFoundError(f"Can't resume deblendFromDeblendedFits from non-existent file"
                                    f"={self.config.filenameOutDeblend}")
        if self.config.plotOnly and not self.config.resume:
            raise ValueError("Can't set plotOnly=True without resume=True")
        data = {}
        data_prior = {}
        if self.config.usePriorShapeDefault:
            data_prior[self.config.priorMagBand] = None
        for catexp in catexps:
            band = catexp.band
            if band in data:
                raise ValueError(f'catexp band={band} already found; duplicates are not supported')
            in_prior = band in data_prior
            in_fit = band in self.config.bands_fit
            if not (in_fit or in_prior):
                raise ValueError(f'catexp band={band} not in self.config.bands_fit={self.config.bands_fit}'
                                 f' or bands_read={self.config.get_bands_read()}')
            if in_fit:
                data[band] = catexp
            if in_prior:
                data_prior[band] = catexp

        if self.config.usePriorShapeDefault:
            catexp = data_prior.get(self.config.priorMagBand)
            if catexp is None:
                raise RuntimeError(
                    f"self.config.priorMagBand={self.config.priorMagBand} not found in any catexp"
                    f"; bands:{[catexp.band for catexp in catexps]}")
            mags_prior = catexp.exposure.getPhotoCalib().calibrateCatalog(catexp.catalog)[
                self.config.priorMagField]
        else:
            mags_prior = None

        bands = list(data.keys())
        if set(bands) != set(self.config.bands_fit):
            raise RuntimeError(f'bands={bands} not identical to config.bands={self.config.bands_fit}')
        exposures = {}
        bbox_ref = None
        for band in self.config.bands_fit:
            exposure = data[band].exposure
            if bbox_ref is None:
                bbox_ref = exposure.image.getBBox()
            else:
                bbox_compare = exposure.image.getBBox()
                if bbox_compare != bbox_ref:
                    raise ValueError(
                        f"Non-matching bboxes: {band}={bbox_compare} != ref {bands[0]}:{bbox_ref}")
            exposures[band] = exposure
        self.bbox_ref = bbox_ref

        filenameOut = (self.config.filenameOut if not self.config.deblendFromDeblendedFits else
                       self.config.filenameOutDeblend)

        if self.config.pathCosmosGalsim:
            tiles = mrCutout.get_tiles_HST_COSMOS()
            ra_corner, dec_corner = mrCutout.get_corners_exposure(next(iter(data.values()))['exposure'])
            extras = mrCutout.get_exposures_HST_COSMOS(
                ra_corner, dec_corner, tiles, self.config.pathCosmosGalsim,
            )
            # TODO: Generalize this for e.g. CANDELS
            bands = [extras[0].band]
        elif not self.config.deblendFromDeblendedFits:
            if self.config.disableNoiseReplacer:
                extras = tuple(
                    (datum.exposure, datum.catalog, {'idx_add': [None]*len(cat_ref)})
                    for datum in data.values()
                )
                for exposure_extra, sources_extra, meta_extra in extras:
                    idx_add = meta_extra['idx_add']
                    for idx_extra, source_extra in enumerate(sources_extra):
                        parent_extra = cat_ref[idx_extra]['parent']
                        has_parent = parent_extra > 0
                        is_blended = (cat_ref.find(parent_extra) if has_parent else source_extra)[
                            'deblend_nChild'] >= 1
                        # Add parents back in when fitting, except if they have
                        # only one child. Parents with one child don't need to
                        # be subtracted or added back in since they're not
                        # blended
                        idx_add[idx_extra] = [] if not is_blended else (
                            [idx_extra] if has_parent else [
                                int(idx_child) for idx_child in np.where(
                                    cat_ref['parent'] == source_extra['id'])[0]
                            ]
                        )
                        if has_parent and is_blended:
                            img_deblend, bbox_src = get_spanned_image(source_extra.getFootprint())
                            # Can happen if something is wrong with the
                            # footprint, e.g. zero-area bbox
                            if img_deblend is None:
                                idx_add[idx_extra] = []
                            else:
                                exposure_extra.image.subset(bbox_src).array -= img_deblend
            else:
                extras = tuple(rebuildNoiseReplacer(datum.exposure, datum.catalog) for datum in data.values())
        else:
            if self.config.bboxDilate > 0:
                extras = []
                for datum in data.values():
                    extras.append(
                        MultiProFitTask._getSegmentationMap(
                            datum.exposure.getBBox(), datum.catalog,
                        )
                    )
            else:
                extras = [None] * len(data)
        timeInit = time.time()
        processTimeInit = time.process_time()
        resultsReturn = None
        toWrite = bool(filenameOut)
        nFit = 0
        numSources = len(cat_ref)
        idx_begin, idx_end = self.config.idx_begin, self.config.idx_end
        if idx_begin < 0:
            idx_begin = 0
        if idx_end > numSources or idx_end < 0:
            idx_end = numSources
        numSources = idx_end - idx_begin

        if self.config.fitGaussian:
            if len(self.config.bands_fit) > 1:
                raise ValueError(f'Cannot fit Gaussian (no PSF) model with multiple filters ({bands})')
            self.models['gausspx_no_psf_1'] = mpfFit.get_model(
                {band: 1 for band in bands}, "gaussian:1", (1, 1), slopes=[0.5], engine='galsim',
                engineopts={'use_fast_gauss': True, 'drawmethod': mpfObj.draw_method_pixel['galsim']},
                name_model='gausspx_no_psf_1'
            )

        flags_failure = ['base_PixelFlags_flag_saturatedCenter', 'deblend_skipped']

        if self.config.skipDeblendTooManyPeaks:
            flags_failure.append('deblend_tooManyPeaks')
        backgroundPriorMultiplier = self.config.backgroundPriorMultiplier
        backgroundPriors = {}

        if backgroundPriorMultiplier > 0 and np.isfinite(backgroundPriorMultiplier):
            for band in bands:
                backgroundPriors[band] = None
        catalog_in = afwTable.SourceCatalog.readFits(self.config.filenameOut) if (
            self.config.resume or self.config.deblendFromDeblendedFits) else cat_ref

        init_from_cat = self.config.plotOnly or self.config.deblendFromDeblendedFits
        if init_from_cat:
            fields_in = self._parseCatalogFields(catalog_in, add_keys=True)
            if fields_in != self.fields:
                raise RuntimeError(f"fields_in={fields_in} don't match self.fields={self.fields}")
        fields = self.fields
        catalog = catalog_in if self.config.deblendFromDeblendedFits else None

        kwargs_moments = {
            'sigma_min': np.max((1e-2, self.config.psfHwhmShrink)),
            'denoise': self.config.estimateContiguousDenoisedMoments,
            'contiguous': self.config.estimateContiguousDenoisedMoments,
        }

        if not init_from_cat:
            catalog = afwTable.SourceCatalog(self.schema)
            catalog.extend(catalog_in, mapper=self.mapper)
        if catalog.schema != self.schema:
            raise RuntimeError(f'catalog.schema={catalog.schema} != self.schema={self.schema} from init')

        deblend = self.config.deblend or self.config.deblendFromDeblendedFits

        for idx in range(np.max([idx_begin, 0]), idx_end):
            src = cat_ref[idx]
            results = None
            id_parent = src['parent']
            n_child = src['deblend_nChild']
            is_parent = n_child > 0
            runtime = 0
            deblended = False
            skipped, succeeded = False, False
            if self.config.deblendFromDeblendedFits and not is_parent:
                skipped = True
                error = 'deblendFromDeblendedFits and not is_parent'
            else:
                flags_failed = {flag: src[flag] for flag in flags_failure}
                skipped = any(flags_failed.values())
                if skipped:
                    error = f'{[key for key, fail in flags_failed.items() if fail]} flag(s) set'
            if not skipped:
                errors = []
                is_child = id_parent != 0
                id_src = src['id']
                # Scarlet/meas_extensions_scarlet has isolated children;
                # meas_deblender does not
                isolated = (cat_ref.find(id_parent)['deblend_nChild'] == 1) if is_child else not is_parent

                if self.config.isolatedOnly and not isolated:
                    errors.append('not isolated')
                if not self.config.deblendFromDeblendedFits and (
                        is_parent and n_child > self.config.maxNChildParentFit):
                    errors.append(f'is_parent and n_child={n_child} > max={self.config.maxNChildParentFit}')
                if errors:
                    error = ' & '.join(errors)
                else:
                    if not is_parent or not deblend:
                        children_cat, children_src = (None, None)
                    else:
                        children_idx = [int(x) for x in np.where(cat_ref['parent'] == id_src)[0]]
                        children_cat, children_src = (
                            [catalog[x] for x in children_idx], [cat_ref[x] for x in children_idx]
                        )
                    footprint = cat_ref.find(id_parent).getFootprint() if (
                        self.config.useParentFootprint and is_child) else None

                    for band in backgroundPriors:
                        if self.config.usePriorBackgroundLocalEstimate:
                            cat_band = data[band]['sources']
                            bg_mean = 0. if cat_band[f'{self.config.field_localbg}_flag'] else \
                                cat_band[f'{self.config.field_localbg}_instFlux']
                            bg_sigma = self.config.backgroundPriorMultiplier * \
                                cat_band[f'{self.config.field_localbg}_instFluxErr']
                        else:
                            bg_mean, bg_sigma = 0, None
                        backgroundPriors[band] = (bg_mean, bg_sigma)

                    if init_from_cat:
                        values_init_psf = {}
                        row_in = catalog_in[idx]
                        name_psfmodel = self.getNamePsfModel()
                        for band, fields_band in fields['psf'].items():
                            values_init_psf[band] = [
                                (name_field, row_in[key]) for name_field, key in fields_band[name_psfmodel]
                            ]
                        for modelspec in self.modelSpecs:
                            modelspec.values_init = [
                                (name_field, row_in[key])
                                for name_field, key in fields['base'][modelspec['name']]
                            ]
                            modelspec.values_init_psf = values_init_psf
                    results, error, deblended = self.__fitSource(
                        src, exposures, extras, children_cat=children_cat,
                        footprint=footprint, failOnLargeFootprint=is_parent,
                        row=catalog[idx] if self.config.deblendFromDeblendedFits else None,
                        usePriorShapeDefault=self.config.usePriorShapeDefault,
                        priorCentroidSigma=self.config.priorCentroidSigma,
                        kwargs_moments=kwargs_moments,
                        mag_prior=mags_prior[idx] if mags_prior is not None else None,
                        backgroundPriors=backgroundPriors,
                        children_src=children_src,
                        results=results, fields=fields, idx_src=idx,
                        logger=None if logger_is_none else logger,
                        **kwargs)
                    succeeded = error is None
                    runtime = (self.metadata["__fitSourceEndCpuTime"]
                               - self.metadata["__fitSourceStartCpuTime"])
            # Preserve the first successful result to return at the end
            if resultsReturn is None and succeeded:
                resultsReturn = results
            # Fill in field values if successful, or save just the runtime to
            # enter later otherwise
            if not self.config.plotOnly:
                row = catalog[idx]
                row[self.failFlagKey] = not succeeded
                row[self.runtimeKey] = runtime
                if succeeded:
                    if not ((self.config.deblend and is_parent) or self.config.deblendFromDeblendedFits):
                        try:
                            self.__setRow(results, fields, row, exposures, src)
                        except FitFailedError as error_fit:
                            error = error_fit
                            succeeded = False

            # Returns the image to pure noise
            if deblended:
                if self.config.disableNoiseReplacer:
                    # This isn't actually necessary as long as the exposures
                    # are float64 copies, as they are now
                    for exposure_extra, _, meta_extra in extras if False else []:
                        img_added, bbox_added = meta_extra['img_bbox']
                        if img_added is not None:
                            exposure_extra.image.subset(bbox_added).array -= img_added
                else:
                    for noiseReplacer in extras:
                        noiseReplacer.removeSource(id_src)
            errorMsg = '' if succeeded else f" {'skipped' if skipped else 'failed'}: {error}"
            # Log with a priority just above info, since MultiProFit itself
            # will generate a lot of info logs per source.
            nFit += 1
            if not self.config.plotOnly:
                logger.info(
                    f"Fit src {idx} ({nFit}/{numSources}) id={src['id']} in {runtime:.3f}s "
                    f"(total time {time.time() - timeInit:.2f}s "
                    f"process_time {time.process_time() - processTimeInit:.2f}s) {errorMsg}"
                )
                if toWrite and (nFit % self.config.intervalOutput) == 0:
                    catalog.writeFits(filenameOut)
        if not self.config.plotOnly:
            if toWrite:
                catalog.writeFits(filenameOut)
            # Return the exposures to their original state
            if (not self.config.fitHstCosmos and not self.config.deblendFromDeblendedFits
                    and not self.config.disableNoiseReplacer):
                for noiseReplacer in extras:
                    noiseReplacer.end()
        return catalog, resultsReturn

    @pipeBase.timeMethod
    def run(
        self,
        catexps: Iterable[fitMb.CatalogExposure],
        cat_ref: afwTable.SourceCatalog,
        **kwargs,
    ) -> pipeBase.Struct:
        """Run the MultiProFit task on a set of catalog-exposure pairs.

        This function is currently a simple wrapper that calls self.fit().

        Parameters
        ----------
        catexps : `iterable` [`lsst.pipe.tasks.fit_multiband.CatalogExposure`]
            A list of CatalogExposures to fit.
        cat_ref: `lsst.afw.table.SourceCatalog`
            A catalog containing deblended sources with footprints
        **kwargs
            Additional keyword arguments to pass to self.fit.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            A new catalog containing all of the fields from `sources` and
            those generated by MultiProFit.
        results : `dict`
            A results structure as returned by mpfFit.fit_galax y_exposures()
            for the first successfully fit source.
        """
        catalog, *_ = self.fit(catexps, cat_ref, **kwargs)
        return pipeBase.Struct(output=catalog)
