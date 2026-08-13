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

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

import lsst.afw.detection as afwDetect
import lsst.afw.image as afwImage
import lsst.geom as geom


def defaultdictNested():
    """Get a nested defaultdict with defaultdict default value.

    Returns
    -------
    defaultdict : `defaultdict`
        A `defaultdict` with `defaultdict` default values.
    """
    return defaultdict(defaultdictNested)


def get_all_subclasses(cls, children_first: bool = True):
    """Return all subclasses of a class recursively.

    Parameters
    ----------
    cls
        The class to get subclasses for.
    children_first
        If true, return child (direct subclasses) first, followed by their
        children (recursively). Otherwise, return each direct child followed
        by its own descendants first.
    """
    subclasses = {c: None for c in cls.__subclasses__()}
    subclasses_return = subclasses.copy() if children_first else {}
    for subclass in subclasses:
        if children_first:
            subclasses_return[subclass] = None
        for subsubclass in get_all_subclasses(subclass):
            subclasses_return[subsubclass] = None
    return list(subclasses_return)


# TODO: Allow addition to existing image
def get_spanned_image(
    exposure: afwImage.Exposure,
    footprint: afwDetect.Footprint = None,
    bbox: geom.Box2I | None = None,
    spans: np.ndarray | None = None,
    get_sig_inv: bool = False,
    calibrate: bool = True,
) -> tuple[np.ndarray, geom.Box2I, np.ndarray]:
    """Get an image masked by its spanset.

    Parameters
    ----------
    exposure
        An exposure to extract arrays from.
    footprint
        The footprint to get spans/bboxes from. Not needed if both of
        `bbox` and `spans` are provided.
    bbox
        The bounding box to subset the exposure with.
        Defaults to the footprint's bbox.
    spans
        A spanset array (inverse mask/selection).
        Defaults to the footprint's spans.
    get_sig_inv
        Whether to get the inverse variance and return its square root.
    calibrate
        Whether to calibrate the image; set to False if already calibrated.

    Returns
    -------
    image
        The image array, with masked pixels set to zero.
    bbox
        The bounding box used to subset the exposure.
    sig_inv
        The inverse sigma array, with masked pixels set to zero.
        Set to None if `get_sig_inv` is False.
    """
    bbox_is_none = bbox is None
    if bbox_is_none:
        bbox = footprint.getBBox()
    if not (bbox.getHeight() > 0 and bbox.getWidth() > 0):
        return None, bbox
    if spans is None:
        spans = footprint.getSpans().asArray()
    sig_inv = afwImage.ImageF(bbox) if get_sig_inv else None
    img = afwImage.ImageF(bbox)
    img.array[:] = np.nan
    if footprint is None:
        maskedIm = exposure.maskedImage.subset(bbox)
        if not calibrate:
            img = maskedIm.image.array
            sig_inv.array[spans] = 1 / np.sqrt(maskedIm.variance.array[spans])
    else:
        img.array[spans] = footprint.getImageArray()
        if get_sig_inv:
            # footprint.getVarianceArray() returns zeros
            variance = exposure.variance[bbox]
            if not calibrate:
                sig_inv.array[spans] = 1 / np.sqrt(variance.array[spans])
        if calibrate:
            # Have to calibrate with the original image
            maskedIm = afwImage.MaskedImageF(
                image=exposure.image[bbox],
                variance=variance if get_sig_inv else None,
            )
    if calibrate:
        maskedIm = exposure.photoCalib.calibrateImage(maskedIm)
        if footprint is None:
            img = maskedIm.image.array
        else:
            # Apply the calibration to the deblended footprint
            # ... hopefully it's multiplicative enough
            img.array[spans] *= maskedIm.image.array[spans] / exposure.image[bbox].array[spans]
            img = img.array
        if get_sig_inv:
            sig_inv.array[spans] = 1 / np.sqrt(maskedIm.variance.array[spans])
            # Should not happen but does with footprints having nans
            sig_inv.array[~(sig_inv.array >= 0)] = 0

    return np.array(img, dtype="float64"), bbox, np.array(sig_inv.array, dtype="float64")


def join_and_filter(separator: str, items: Iterable[str], exclusion: str | None = None) -> str:
    """Join an iterable of items by a separator, filtering out an exclusion.

    Parameters
    ----------
    separator
        The separator to join items by.
    items
        Items to join.
    exclusion
        The pattern to exclude.

    Returns
    -------
    joined
        The joined string.
    """
    return separator.join(filter(exclusion, items))
