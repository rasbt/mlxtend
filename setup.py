from setuptools import setup, find_packages
import sys

requirements = [
          'matplotlib>=3.0.0',
          'joblib>=0.13.2'
          'dlib>=19.24.0'
]

py36_requirements = [
       'scikit-learn~=0.20.3',
       'numpy<1.20', # last version supported by python 3.6  
       'pandas~=0.23.0',
       'scipy~=1.1.0',
]

py39_requirements = [
       'scikit-learn==1.1.1',
       'numpy==1.23.0', # require a version which is compatible with py3.9 and is > 1.20 due to API changes  
       'pandas~=1.4.3',
       'scipy~=1.9.0',
]

if sys.version_info[0] == 3 and sys.version_info[1] == 6:
    requirements += py36_requirements
else:
    requirements += py39_requirements


setup(name='mlxtend',
      version='0.0.2',
      description='Machine Learning Library Extensions',
      url='https://github.com/pelucid/mlxtend',
      packages=find_packages(),
      install_requires=requirements,
      setup_requires=["pytest-runner"],
      tests_require=['pytest~=6.2', 'pytest-cov', 'tomli==1.2.2', 'coverage==6.2']
      )