# -*- coding: utf-8 -*-
"""
@author: Yusen Ye
"""

import os
import sys
import shutil
from subprocess import call
from warnings import warn
from setuptools import setup

setup(name='CIRCLET',
version='1.0',
package_dir={'': 'src'},
packages=['CIRCLET'],
package_data={
        # And include any *.msg files found in the 'hello' package, too:
        'CIRCLET': ['DATA/*','*.txt','DATA/RNA-seq/*','DATA/Hi-Cmaps/*','DATA/Nagano et al/*'],
    },

include_package_data=True
)

