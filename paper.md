---
title: "MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack"
tags:
- machine learning
- data science
- association rule mining
- ensemble learning
- feature selection
authors:
- name: Sebastian Raschka
  orcid: 0000-0001-6989-4493
  affiliation: 1
affiliations:
- name: Michigan State University
  index: 1
date: 15 March 2018
bibliography: paper.bib
---

# Summary

MLxtend is a library that implements a variety of core algorithms and utilities for machine learning and data mining. The primary goal of MLxtend is to make commonly used tools accessible to researchers in academia and data scientists in industries focussing on user-friendly and intuitive APIs and compatibility to existing machine learning libraries, such as scikit-learn, when appropriate. While MLxtend implements a large variety of functions, highlights include sequential feature selection algorithms [@pudil1994floating], implementations of stacked generalization [@wolpert1992stacked] for classification and regression, and algorithms for frequent pattern mining [@agrawal1994fast]. The sequential feature selection algorithms cover forward, backward, forward floating, and backward floating selection and leverage scikit-learn's cross-validation API [@pedregosa2011scikit] to ensure satisfactory generalization performance upon constructing and selecting feature subsets. Besides, visualization functions are provided that allow users to inspect the estimated predictive performance, including performance intervals, for different feature subsets. The ensemble methods in MLxtend cover majority voting, stacking, and stacked generalization, all of which are compatible with scikit-learn estimators and other libraries as XGBoost [@chen2016xgboost]. In addition to feature selection, classification, and regression algorithms, MLxtend implements model evaluation techniques for comparing the performance of two different models via McNemar's test and multiple models via Cochran's Q test. An implementation of the 5x2 cross-validated paired t-test [@dietterich1998approximate] allows users to compare the performance of machine learning algorithms to each other. Furthermore, different flavors of the Bootstrap method [@efron1994introduction], such as the .632 Bootstrap method [@efron1983estimating] are implemented to compute confidence intervals of performance estimates. All in all, MLxtend provides a large variety of different utilities that build upon and extend the capabilities of Python's scientific computing stack.

# Acknowledgements

I would like to acknowledge all of the contributors and users of mlxtend, who helped with valuable feedback, bug fixes, and additional functionality to further improve the library: James Bourbeau, Reiichiro Nakano, Will McGinnis, Guillaume Poirier-Morency, Colin Carrol, Zach Griffith, Anton Loss, Joshua Goerner, Eike Dehling, Gilles Armand, Adam Erickson, Mathew Savage, Pablo Fernandez, Alejandro Correa Bahnsen, and many others. A comprehensive list of all contributors to mlxtend is available at https://github.com/rasbt/mlxtend/graphs/contributors.

# References
