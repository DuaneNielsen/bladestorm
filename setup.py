#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='bladestorm',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym>=0.2.3',
                        'torch',
                        'ray',
                        'opencv-python',
                        'numpy',
                        'tensorboardX',
                        'peewee',
                        'redis',
                        'psycopg2-binary'],
      extras_require={
          'dev': [
              'pytest'
          ],
      }
      )
