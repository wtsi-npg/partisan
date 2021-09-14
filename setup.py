# -*- coding: utf-8 -*-
#
# Copyright © 2020, 2021 Genome Research Ltd. All rights reserved.
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# @author Keith James <kdj@sanger.ac.uk>

from pathlib import Path

from setuptools import setup

project_root = Path(__file__).parent
long_description = (project_root / "README.md").read_text()

setup(
    name="partisan",
    packages=["partisan"],
    url="https://github.com/kjsanger/partisan",
    license="GPL3",
    author="Keith James",
    author_email="kdj@sanger.ac.uk",
    description="A Python API for iRODS using the baton.py iRODS client",
    long_description=long_description,
    long_description_content_type="text/markdown",
    use_scm_version=True,
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=["structlog"],
    tests_require=["pytest", "pytest-it"],
    scripts=[],
)
