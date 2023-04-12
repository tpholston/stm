# stm: A Python Package for the Structural Topic Model

Website: www.structuraltopicmodel.com

Authors: [Tyler Holston](tholston@ucsd.edu), [Umberto Mignozetti](umbertomig@ucsd.edu)

Please email all comments/questions to: tholston@ucsd.edu

[![PyPI Version](https://img.shields.io/pypi/v/your-package-name)](https://pypi.org/project/your-package-name/)
[![PyPI Downloads](https://img.shields.io/pypi/v/your-package-name)](https://pypi.org/project/your-package-name/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/your-package-name/blob/master/LICENSE)

## Summary

This repository is a Python package for implementing the Structural Topic Model (STM), a statistical method for analyzing topic patterns in text data. It provides tools for ingesting and manipulating text data, estimating Structural Topic Models, calculating covariate effects on latent topics with uncertainty, estimating topic correlations, and creating visualizations for analysis.

### Installation Instructions

To install your package from PyPI, you can use pip:

```
pip install py_stm
```

### Getting Started
After installing the package, you can import it in your Python code and use the STM class to estimate the model. For example:

```
from py_stm import STM
```

### Usage
```
# Create an instance of STM
model = STM()

# Fit the model to your data
model.fit(your_data)

# Access the estimated parameters and results
estimated_topics = model.topics
covariate_effects = model.covariate_effects
# ... more results

# Perform inference and prediction
inference_results = model.inference()
prediction_results = model.predict(your_new_data)
```

### Contribution
We welcome contributions to improve and expand this package. Please fork the repository, make your changes, and submit a pull request. We appreciate your help in making this package better for the community!
