# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from setuptools import setup, find_packages
import mlxtend

VERSION = mlxtend.__version__

setup(name='mlxtend',
      version=VERSION,
      description='Machine Learning Library Extensions',
      author='Sebastian Raschka',
      author_email='mail@sebastianraschka.com',
      url='https://github.com/rasbt/mlxtend',
      packages=find_packages(),
      package_data={'': ['LICENSE-BSD3.txt',
                         'LICENSE-CC-BY.txt',
                         'README.md',
                         'requirements.txt']
                    },
      include_package_data=True,
      install_requires=['numpy>=1.10.4', 'scipy>=0.17'],
      extras_require={'testing': ['nose'],
                      'docs': ['mkdocs']},
      license='BSD 3-Clause',
      platforms='any',
      classifiers=[
             'License :: OSI Approved :: BSD License',
             'Development Status :: 5 - Production/Stable',
             'Operating System :: Microsoft :: Windows',
             'Operating System :: POSIX',
             'Operating System :: Unix',
             'Operating System :: MacOS',
             'Programming Language :: Python :: 2',
             'Programming Language :: Python :: 2.7',
             'Programming Language :: Python :: 3',
             'Programming Language :: Python :: 3.3',
             'Programming Language :: Python :: 3.4',
             'Programming Language :: Python :: 3.5',
             'Topic :: Scientific/Engineering',
             'Topic :: Scientific/Engineering :: Artificial Intelligence',
             'Topic :: Scientific/Engineering :: Information Analysis',
             'Topic :: Scientific/Engineering :: Image Recognition',
      ],
      long_description="""

A library of Python tools and extensions for data science.


Contact
=============

If you have any questions or comments about mlxtend,
please feel free to contact me via
eMail: mail@sebastianraschka.com
or Twitter: https://twitter.com/rasbt

This project is hosted at https://github.com/rasbt/mlxtend

The documentation can be found at http://rasbt.github.io/mlxtend/

""")
