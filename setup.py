# Author: bbrighttaer
# Project: IVPGAN for DTI
# Date: 07/06/2019
# Time: 
# File: setup.py.py


from setuptools import setup

setup(
    name='jova',
    version='0.0.1',
    packages=['jova', 'jova.nn', 'jova.nn.tests', 'jova.utils', 'jova.utils.tests',
              'jova.metrics'],
    url='',
    license='MIT',
    author='Brighter Agyemang',
    author_email='brighteragyemang@gmail.com',
    description='',
    install_requires=['torch', 'numpy', 'scikit-optimize', 'pandas', 'matplotlib', 'seaborn', 'soek', 'biopython',
                      'joblib', 'scikit-learn']
)
