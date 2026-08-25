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

import pytest

# analysis_tools is not and should not be a dependency of this package, so
# only run these tests if it is set up.
try:
    from lsst.meas.extensions.multiprofit.analysis_tools import MultiProFitSersicSizeMagnitudePlot

    has_analysis_tools = True
except ImportError:
    has_analysis_tools = False

if has_analysis_tools:

    @pytest.fixture(scope="module")
    def kwargs_plot():
        """Return sensible kwargs for size-magnitude plots."""
        kwargs = dict(
            xLims=(18, 25),
            yLims=(-3, 4),
        )
        return kwargs

    @pytest.fixture(scope="module")
    def tool_sersic(kwargs_plot):
        """Return a finalized Sersic size-magnitude plot action."""
        atool = MultiProFitSersicSizeMagnitudePlot()
        atool.finalize()
        return atool

    @pytest.fixture(scope="module")
    def data_sersic(tool_sersic):
        """Return the schema for the Sersic size-magnitude plot action."""
        schema = tool_sersic.getInputSchema()
        data = {key: [] for key in schema}
        return data

    def test_psf_fits(tool_sersic, data_sersic):
        """Test that the tool and its schema are useable."""
        assert tool_sersic is not None
        assert len(data_sersic) > 0
