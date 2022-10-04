.. SPDX-FileCopyrightText: 2022 GroupeSNCF 
..
.. SPDX-License-Identifier: Apache-2.0

.. _install:

Installation
------------

Before creating a virtual environment, make sure at least **Python3.6** is installed. If not, install it with::

    sudo apt-get install python3.7-dev

Create a virtual environment ::

    python3 -m venv interpret-env

Source the virtual environment with ::

    source .venv/bin/activate

Install Python dependencies with ::

    pip install -r requirements/requirements.txt

Install project modules with ::

    pip install -e .

This command runs the ``setup.py`` script to make the package **readml** available in the environment.

Python version support
----------------------

Officially Python 3.6 and above.

Quickstart
----------

Note that for the **first two usages**, you will need to update required configuration variables located in ``readml/config/config_local.cfg``
If you don't want to clone the repository, you can jump to the last two usages.


Usage via command line interface [CLI]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The most straightforward way is to use the **interpret** click command , that wraps around most of the functionality in the module.
The command parameters allow you to choose the interpretability method. These parameters are not used for DL models since there is only one method for each data type.

.. note:: You need to update required configuration variables located in ``readml/config/config_local.cfg``

A basic usage example is shown below ::

    python readml/main.py

Supported parameters are (not used for DL)::

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


Usage as external module [Python package]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One way of using the package is to run the ``interpret`` function which takes care of explaining model behaviour.

.. note:: You need to update required configuration variables located in ``readml/config/config_local.cfg`` before.

For instance , using **partial dependency plots** for global interpretability on ML model::

    from readml.main import interpret
    interpret(interpret_type="global", use_pdp_ice=True, use_ale=False, use_shap=False)

The function parameters are not used for DL models since there is only one method for each data type.

Usage without filling in the config file [Python package]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can also use the package without filling in the configuration file by using the ``ExplainML`` class which contains
many methods to explain globally or locally any ML model, or the ``ExplainDL`` class to explain locally any DL model
when applied to tabular, textual or image data.

For instance , using **accumulated local effect plots** for global interpretability of a tree based classification model::

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

Here is an other example of using **Guided GRAD-CAM** for local interpretability of a DL model applied to images::

    from readml.explainers.dl.explain_dl import ExplainDL
    from tensorflow.keras.applications.vgg16 import VGG16
    exp = ExplainDL(
            model=VGG16(),
            task_name="classification",
            out_path="outputs_dl/",
        )
    exp.explain_image(image_dir= "readml/data_factory/inputs/data/image_data/")

Usage of a particular explainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Example : icecream module for PDP, ICE & ALE plots**

**icecream** is a module that aims at explaining how a machine learning model works by drawing Partial Dependency Plots, Individual Conditional Expectation and Accumulated Local Effects.

For instance , using **partial dependency plots** for global interpretability::

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

.. note:: Other explainers located in ``readml/explainers/`` can also be used directly.


Deployment
----------

In order to configure deployment environment, one may create environment variable ``INTERPRET_ENV``
to specify deployment env, two  modes are supported :

- Deploy in dev env
- Deploy in prod env

By default , ``INTERPRET_ENV = "local"``

To create new deployment modes :

- Update ``INTERPRET_ENV`` ::

    export INTERPRET_ENV = $deploy_env
- Create new configuration file named ``config_{deploy_env}.cfg`` based on existing templates
- Copy configuration file  in ``config/`` directory

