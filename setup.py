import os
import shutil

from setuptools import find_packages, setup, distutils

LONG_DESCRIPTION = u"""
==============================================
py_stm -- Structural Topic Model in Python
==============================================

Features
---------

Installation
------------

Documentation
-------------

Citing gensim
-------------
"""

class CleanExt(distutils.cmd.Command):
    description = 'Remove C sources, C++ sources and binaries for gensim extensions'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for root, dirs, files in os.walk('gensim'):
            files = [
                os.path.join(root, f)
                for f in files
                if os.path.splitext(f)[1] in ('.c', '.cpp', '.so')
            ]
            for f in files:
                self.announce('removing %s' % f, level=distutils.log.INFO)
                os.unlink(f)

        if os.path.isdir('build'):
            self.announce('recursively removing build', level=distutils.log.INFO)
            shutil.rmtree('build')

cmdclass = {'clean_ext': CleanExt}

install_requires = [
    'numpy >= 1.18.5',
    'scipy >= 1.7.0',
    'smart_open >= 1.8.1',
    'gensim >= 4.3.0',
]

setup_requires = [
    'numpy >= 1.18.5',
    'gensim >= 4.3.0'
]

setup(
    name='py_stm',
    version='1.0.0',
    description='Python framework for structural topic modeling',
    long_description=LONG_DESCRIPTION,

    cmdclass=cmdclass,
    packages=find_packages(),

    author=u'Tyler Holston, Umberto Mignozetti',
    author_email='tholston@ucsd.edu',

    project_urls={
        'Source': 'https://github.com/tpholston/stm',
    },

    keywords='Latent Dirichlet Allocation, LDA, Structural Topic Model, '
        'Structured Topic Model, STM, Structured Topic Modeling',

    platforms='any',

    zip_safe=False,

    classifiers=[  # from https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],

    test_suite="py_stm.test",
    python_requires='>=3.8',
    setup_requires=setup_requires,
    install_requires=install_requires,

    include_package_data=True,
)