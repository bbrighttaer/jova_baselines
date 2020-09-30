# Author: bbrighttaer
# Project: jova
# Date: 5/23/19
# Time: 10:30 AM
# File: __init__.py.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.cuda import is_available

allow_cuda = True
cuda = is_available() and allow_cuda

import jova.splits as splits
import jova.data as data
import jova.feat as feat
import jova.metrics as metrics
import jova.nn as nn
import jova.utils as utils
import jova.trans as trans
