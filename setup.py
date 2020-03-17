from setuptools import setup, find_packages

setup(name='mlxtend',
      version='0.0.1',
      description='Machine Learning Library Extensions',
      url='https://github.com/pelucid/mlxtend',
      packages=find_packages(),
      install_requires=[
          "scipy>=1.2.1",
          "numpy>=1.16.2",
          "pandas>=0.24.2",
          "scikit-learn>=0.20.3",
          "matplotlib>=3.0.0",
          "joblib>=0.13.2"
      ],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"]
      )
