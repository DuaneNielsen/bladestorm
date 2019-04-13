#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='bladestorm',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym[classic_control]>=0.2.3', 'gym[box2d]>=0.2.3', 'gym[atari]>=0.2.3',
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
