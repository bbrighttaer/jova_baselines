import sys
from distutils.core import setup
from distutils.extension import Extension

import numpy as np

sys.argv[1:] = ['build_ext', '--inplace']

ext_modules = [Extension(
    name="swapped",
    sources=["ext_src/swapped.c", "ext_src/c_swapped.c"],
    language="c", include_dirs=[np.get_include()])
]

setup(
    name='RLScore',
    ext_modules=ext_modules,
)
