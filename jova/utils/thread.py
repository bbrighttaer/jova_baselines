# Author: bbrighttaer
# Project: jova
# Date: 11/22/19
# Time: 11:33 AM
# File: thread.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import threading
import time


class UnboundedProgressbar(threading.Thread):
    def __init__(self, delay=.3):
        super(UnboundedProgressbar, self).__init__(target=self.print)
        self._stop_called = False
        self.delay = delay

    def stop(self):
        self._stop_called = True

    def print(self):
        start = time.time()
        while not self._stop_called:
            sys.stdout.write('Computing: [')
            sys.stdout.flush()
            for i in range(100):
                time.sleep(self.delay)
                sys.stdout.write('=')
                sys.stdout.flush()
            sys.stdout.write(']\n')
            sys.stdout.flush()
        duration = time.time() - start
        print('Task duration: {:.0f}m {:.0f}s'.format(duration // 60, duration % 60))

# pb = UnboundedProgressbar()
# pb.start()
# time.sleep(3)
# pb.stop()
# pb.join()
# print('The end!')
