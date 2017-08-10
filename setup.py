try:
    from setuptools import setup
except ImportError:
    from distutils.core	import setup

""" To install package pyipm and its dependencies, run:                                      
                                                                                             
        python setup.py install                                                              
                                                                                             
"""

setup(name='pyipm',
      version='0.1',
      description='Interior-point method for solving nonlinear programs',
      url='https://github.com/jkaardal/pypm/',
      author='Joel T. Kaardal',
      license='MIT',
      py_modules=['pyipm', 'unit_tests'],
      install_requires=[
          'numpy>=1.7.1',
          'scipy>=0.11',
          'theano>=0.8.2',
      ])
