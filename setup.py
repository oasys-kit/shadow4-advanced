# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
# Copyright (c) 2019-2020 Lawrence Berkeley National Laboratory
# Copyright (c) 2020-now European Synchrotron Radiation Facility
# Copyright (c) 2023-now Argonne National Laboratory
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

#
# Memorandum: 
#
# Install from sources: 
#     git clone https://github.com/oasys-kit/shadow4-hybrid
#     cd shadow4
#     python -m pip install -e . --no-deps --no-binary :all:
#
# Upload to pypi (when uploading, increment the version number):
#     python setup.py register (only once, not longer needed)
#     python setup.py sdist
#     python -m twine upload dist/...
#          
# Install from pypi:
#     pip install shadow4-hybrid
#

__authors__ = ["M Sanchez del Rio, Luca Rebuffi, Xianbo shi"]
__license__ = "MIT"
__date__ = "2016-now"

from setuptools import setup

PACKAGES = [
    "shadow4_hybrid",
]

INSTALL_REQUIRES = (
    'setuptools',
    'numpy',
    'scipy',
    'syned>=1.0.30',
    'srxraylib',
    'wofryimpl>=1.0.26',
    'shadow4>=0.1.24',
    'shadow-hybrid-methods>=1.0.4',
)

setup(name='shadow4-hybrid',
      version='0.0.7',
      description='shadow/hybrid implementation in python',
      author='Manuel Sanchez del Rio, Luca Rebuffi, Xianbo Shi',
      author_email='lrebuffi@anl.gov',
      url='https://github.com/oasys-kit/shadow4-hybrid/',
      packages=PACKAGES,
      install_requires=INSTALL_REQUIRES,
     )

