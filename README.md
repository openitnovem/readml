# ReadML : Easier Explainability in python

![Logo](/doc/source/_static/logo_readml.png)

[![Code style: black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue)](https://www.python.org/downloads/release/python-360/)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-2.4-orange)](https://www.tensorflow.org/)
![readml](https://img.shields.io/badge/readml-v0.1-green)

**readml** is a Python library for easy explainability using  machine learning and deep learning interpretability techniques. 
It supports regression and classification models.

## Overview

This package incorporates machine learning (and deep learning) **explainability techniques** under one roof. 

With this package, you can understand and explain your model's global behavior **global interpretability**, understand the reasons behind individual predictions **local interpretability** or both.

The techniques available in this package are:
## Global explainability
- Partial Dependecy Plots
- Individual Conditional Expectation Plots 
- Accumulated Local Effects Plots
- SHAP feature importance and summary plots
## Local explainability
- SHAP local explanation plots for non TreeBased model (model agnostic)
- SHAP local explanation plots for TreeBased model (XGBoost, LightGBM, CatBoost, Pyspark & most tree-based models in scikit-learn)
- SHAP local explanation plots for DL model on tabular data (using Deep SHAP DeepExplainer)
- GRAD-CAM for DL models on image data.



## Dependencies
**readml** 0.1 requires : 
- [Python 3.6+](https://www.python.org/)
- [click](https://pypi.org/project/click/)
- [Pandas](https://pypi.org/project/pandas/)
- [plotly](https://pypi.org/project/plotly/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [Tensorflow 2](https://pypi.org/project/tensorflow/)
- [shap](https://pypi.org/project/shap/)


## Installation instructions

First, setup a clean Python environment for your project with at least Python 3.6 using [venv](https://docs.python.org/3/library/venv.html), [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env") or [virtualenv](https://virtualenv.pypa.io/en/latest/).

Then, you can install readml using pip:

    pip install readml


## Pre-requisites
**Optional for some usages** (See the [Quickstart](#Quickstart) section)

Update required configuration variables located in `readml/config/config_local.cfg` 


### Supported techniques 

| Interpretability Technique | Interpretability Type | Model Type | Data Type |
| --- | --- | ---| ---|
|Partial Dependecy Plots|Global|Model Agnostic|Tabular Data|
|Individual Conditional Expectation Plots|Global|Model Agnostic|Tabular Data|
|Accumulated Local Effects Plots|Global|Model Agnostic|Tabular Data|
|ShapTreeExplainer - Feature Importance|Global|Model Specific : Tree Based models|Tabular Data|
|ShapKernelExplainer - Feature Importance|Global|Model Agnostic: Non Tree Based models|Tabular Data|ML|
|ShapKernelExplainer - force plot|Local|Model Agnostic: Non Tree Based models|Tabular Data|
|ShapTreeExplainer - force plot|Local|Model Specific : Tree Based models|Tabular Data|
|ShapDeepExplainer - force plot|Local|Model Specific : DL models on tabular & text data|Tabular or Text Data|
|GRAD-CAM heatmaps|Local|Model Specific : DL models on image data|Image Data|



## Quickstart


### Usage as external module  [Python package] 

One way of using the package is to run the `interpret` function which takes care of explaining model behaviour .

You need to update required configuration variables located in `readml/config/config_local.cfg` before.

For instance , using **partial dependency plots** for global interpretability on ML model:
```python
from readml.main import interpret
interpret(interpret_type="global", use_pdp_ice=True, use_ale=False, use_shap=False)
```
### Usage without filling in the config file (by passing data and model directly)

You can also use the package without filling in the configuration file by using the `ExplainML` class which contains 
many methods to explain globally or locally any ML model, or the `ExplainDL` class to explain locally any DL model 
when applied to tabular, textual or image data.

For instance , using **accumulated local effect plots** for global interpretability of a tree based classification model (`global_ale` method):
```python
from readml.explainers.ml.explain_ml import ExplainML
exp = ExplainML(
        model=xgb_model,
        task_name="classification",
        tree_based_model=True,
        features_name=["f1", "f2", "f3", "f4", "f5"],
        features_to_interpret=["f1", "f2"],
        target_col="target",
        out_path="outputs_ml/",
    )
exp.global_ale(df_train)
```

Here is an other example of using **GRAD-CAM** for local interpretability of a tensorflow DL model applied to images:
```python
from readml.explainers.dl.explain_dl import ExplainDL
from tensorflow.keras.applications.vgg16 import VGG16
exp = ExplainDL(
        model=VGG16(),
        out_path="outputs_dl/",
    )
exp.explain_image(
            image_dir="readml/data_factory/inputs/data/image_data",
            size=(224, 224),
            color_mode="rgb",
        )
```
### Usage of a particular explainer : icecream module for PDP, ICE & ALE plots

**icecream** is a module that aims at explaining how a machine learning model works by drawing Partial Dependency Plots,
 Individual Conditional Expectation and Accumulated Local Effects. 
 
For instance , using **partial dependency plots** for global interpretability:
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from readml.icecream import icecream
# load data and adapt for binary classification
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
df['label'] = (df.species == 'setosa') * 1
df = df.drop('species', axis=1)
# train a classification model
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
model = LogisticRegression(solver='lbfgs').fit(df[features], df['label'])
# optionally customize icecream options
icecream.options.default_number_bins = 20
# create partial dependencies and draw plots
pdp = icecream.IceCream(
        data=df[features],
        feature_names=features,
        bins={'sepal_length': 10},
        model=model,
        targets=df['label'],
        use_classif_proba=True,
        use_ale= False
    )
pdp.draw(kind='pdp', show=True)
# create 2D partial dependencies and draw plots
pdp2d = icecream.IceCream2D(
        data=df[features],
        feature_x='petal_length',
        feature_y='sepal_width',
        bins_x=10,
        bins_y=10,
        model=model,
        targets=df['label'],
        use_classif_proba=True,
    )
pdp2d.draw(kind='hist', show=True)
```
Other explainers located in `readml/explainers/` can also be used directly.


### Usage via command line interface [CLI] 
The most straightforward way is to use the **interpret** click command , that wraps around most of the functionality in the module.

You need to update required configuration variables located in `readml/config/config_local.cfg`

A basic usage example is shown below :

```bash src
python readml/main.py 

```

Supported parameters are (not used for DL):

```bash src
python readml/main.py --help

Usage: main.py [OPTIONS]

Options:
  --interpret-type   Interpretability type: Choose global, local or mix. Not
                     needed for DL  [default: mix]

  --use-ale          Computes and plots ALE. Not needed for DL  [default:
                     True]

  --use-pdp-ice      Computes and plots PDP & ICE. Not needed for DL
                     [default: True]

  --use-shap         Computes and plots shapely values for global & local
                     explanation. Not needed for DL  [default: True]

  --help             Show this message and exit.

```

## Documentation

* [API Documentation](https://openitnovem.github.io/readml/)
* [Examples](to_complete)


## Linting 
For future development, please use black for code formatting  

## References 
- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
- [SHAP](https://github.com/slundberg/shap)


## Copyright 
DSE Team - Data IA Factory


