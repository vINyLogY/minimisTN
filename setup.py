#!/usr/bin/env python2
# coding: utf-8

import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='minimisTN',
    version='0.0.1',
    description='A small program solving multidimensional problem',
    long_description=long_description,
    url='https://github.com/vINyLogY/minimisTN',
    author='Xinxian Chen',
    author_email='vinylogy9@gmail.com',
    license='GNU GPLv3',
    classifiers=[
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    install_requires=['matplotlib', 'numpy', 'scipy', 'sympy', 'future', 'six'],
)
