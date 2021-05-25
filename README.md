# fbd-interpreter Package
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Code style: black](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)
[![Python 3.6+](https://img.shields.io/badge/python-3.6-blue)](https://www.python.org/downloads/release/python-360/)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-2.2-orange)](https://www.tensorflow.org/)
![Fbd interpreter](https://img.shields.io/badge/fbd__interpreter-v1-green)
![Maintainer](https://img.shields.io/badge/Maintainer-Soumaya%20IHHI-green)
![Reviewed by](https://img.shields.io/badge/Reviewed%20by-Guido%20INTRONATI-green)

The interpreter package of the Fab Big Data (SNCF).

Its purpose is to make state-of-the-art machine learning and deep learning interpretability techniques easy to use. 


## Requirements 
**fbd_interpreter** 1.0 requires : 
* Python 3.6+

Before creating a virtual environment, make sure Python3.6 is installed. If not, install it with `sudo apt-get install python3.6-dev`.

## Dependencies

- [click](https://pypi.org/project/click/)
- [Pandas](https://pypi.org/project/pandas/)
- [plotly](https://pypi.org/project/plotly/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [shap](https://pypi.org/project/shap/)
- [Tensorflow](https://pypi.org/project/tensorflow/)

## Installation instructions

Create a virtual environment : `python3 -m venv interpret-env`.

Source the virtual environment with `source .venv/bin/activate`.

Install Python dependencies with `pip install -r requirements/requirements.txt`

Install project modules with `pip install -e .`, this command runs the `setup.py` script to make the package `fbd_interpreter` available in the environment.

## Pre-requisites
**Optional for some usages** (See the Quickstart section)

Update required configuration variables located in `fbd_interpreter/config/config_local.cfg` 

## Overview

This package incorporates state-of-the-art machine learning (and deep learning) **interpretability techniques** under one roof. 

With this package, you can understand and explain your model's global behavior **global interpretability**, understand the reasons behind individual predictions **local interpretability** or both (mix).

The techniques available in this package are:
## Global interpretability
- Partial Dependecy Plots (from icecream) 
- Individual Conditional Expectation Plots (from icecream)
- Accumulated Local Effects Plots (from icecream)
- SHAP feature importance and summary plots (from SHAP)
## Local interpretability
- SHAP local explanation plots for non TreeBased model (model agnostic)
- SHAP local explanation plots for TreeBased model (XGBoost, LightGBM, CatBoost, Pyspark & most tree-based models in scikit-learn).
- SHAP local explanation plots for DL model on text or tabular data (using Deep SHAP)
- GRAD-CAM for DL models on image data

Detailed infos are available [here](https://wiki-big-data-ia.intranet.itnovem.com/index.php/REaDME_:_Extraire_la_logique_m%C3%A9tier)

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

### Usage via command line interface [CLI] 
The most straightforward way is to use the **interpret** click command , that wraps around most of the functionality in the module.

You need to update required configuration variables located in `fbd_interpreter/config/config_local.cfg`

A basic usage example is shown below :

```bash src
python fbd_interpreter/main.py 

```

Supported parameters are (not used for DL):

```bash src
python fbd_interpreter/main.py --help

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

### Usage as external module  [Python package] 

One way of using the package is to run the `interpret` function which takes care of explaining model behaviour .

You need to update required configuration variables located in `fbd_interpreter/config/config_local.cfg` before.

For instance , using **partial dependency plots** for global interpretability on ML model:
```python
from fbd_interpreter.main import interpret
interpret(interpret_type="global", use_pdp_ice=True, use_ale=False, use_shap=False)
```
### Usage without filling in the config file (by passing data and model directly)

You can also use the package without filling in the configuration file by using the `ExplainML` class which contains 
many methods to explain globally or locally any ML model, or the `ExplainDL` class to explain locally any DL model 
when applied to tabular, textual or image data.

For instance , using **accumulated local effect plots** for global interpretability of a tree based classification model:
```python
from fbd_interpreter.explainers.ml.explain_ml import ExplainML
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

Here is an other example of using **GRAD-CAM** for local interpretability of a DL model applied to images:
```python
from fbd_interpreter.explainers.dl.explain_dl import ExplainDL
from tensorflow.keras.applications.vgg16 import VGG16
exp = ExplainDL(
        model=VGG16(),
        out_path="outputs_dl/",
    )
exp.explain_image(
            image_dir="fbd_interpreter/data_factory/inputs/data/image_data",
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
from fbd_interpreter.icecream import icecream
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
Other explainers located in `fbd_interpreter/explainers/` can also be used directly.

## Documentation

To run documentation, go to the `doc` folder and run:

`make html`

Documentation will be available in `build/html/` folder.

## Test

To run tests and generate html report use:

`pytest --cov=fbd_interpreter tests/ --cov-report=html`


## Deployment 

In order to configure deployment environment , one may create environment variable `INTERPRET_ENV`
to specify deployment env , two  modes are supported :
- Deploy in dev env
- Deploy in prod env 

By default , `INTERPRET_ENV = "local"` 

To create new deployment modes :
- Update `INTERPRET_ENV` :  ```export INTERPRET_ENV = $deploy_env ```
- Create new configuration file named `config_{deploy_env}.cfg` based on existing templates 
- Copy configuration file  in `config/` directory 


## Git hooks

Copy hooks file from config/hooks/ in .git/hooks and make them executable
```
cp config/hooks/pre-commit .git/hooks/
cp config/hooks/post-commit .git/hooks/
chmod 775 .git/hooks/pre-commit
chmod 775 .git/hooks/post-commit
```
At each commit, those hooks make a clean copy (without outputs) of all notebooks from jupyter/ to notebook_git/ and commit them. If you need to use the notebook of another user, copy it from notebook_git to your folder in jupyter/ before modifying it.

## Upgrade dependencies and freeze (if necessary)

Upgrade Python dependencies with `pip install -r requirements/requirements-to-freeze.txt --upgrade`

Then freeze dependencies with the command `pip freeze | grep -v "pkg-resources" > config/requirements.txt` (the `grep` part deals with a bug specific to Ubuntu 16.04, see https://github.com/pypa/pip/issues/4022)



## Linting 
For future development, please use black for code formatting  

## References 
- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
- [SHAP](https://github.com/slundberg/shap)


## Copyright 
DSE Team - Big Data Fab

## Author
* **Soumaya IHIHI** - *Data scientist*
