.. SPDX-FileCopyrightText: 2022 GroupeSNCF 
..
.. SPDX-License-Identifier: Apache-2.0

Global interpretability in a nutshell
=====================================

Global model interpretability helps to understand the distribution of the target outcome based on the features.
It allows us to know the impact of each feature on the model behaviour.
The Global interpretability techniques available in this package are:

- Partial Dependecy Plots (from :doc:`icecream` module)
- Individual Conditional Expectation Plots (from :doc:`icecream` module)
- Accumulated Local Effects Plots (from :doc:`icecream` module)
- SHAP feature importance and summary plots (from SHAP)

PDP & ICE Plots (from icecream)
-------------------------------

Partial Dependency and Individual Conditional Expectation plots are implemented in the :doc:`icecream` module.

How it works
~~~~~~~~~~~~

To explain how a feature influence a model using PDP & ICE plots:

  - we create fake data based on its values in the dataset
  - we make new predictions on these fake data
  - directly plotting these new predictions creates **ice plots**
  - plotting aggregated values of these new predictions creates **pdplots**

Suppose we have this dataset, containing 3 features and 3 examples:

=== === ===
 A   B   C
=== === ===
 A0  B0  C0
 A1  B1  C1
 A2  B2  C2
=== === ===

We want to study the influence of feature `A` on the model. We create, for each possible value of feature `A`, a dataframe identical to the original dataset, except that feature `A` will be replaced by one of its value. Then we make predictions on all these dataframes. Here predictions are called `Y`:

=== === === ===
 A   B   C   Y
=== === === ===
 A0  B0  C0 Y00
 A0  B1  C1 Y01
 A0  B2  C2 Y02
=== === === ===

=== === === ===
 A   B   C   Y
=== === === ===
 A1  B0  C0 Y10
 A1  B1  C1 Y11
 A1  B2  C2 Y12
=== === === ===

=== === === ===
 A   B   C   Y
=== === === ===
 A2  B0  C0 Y20
 A2  B1  C1 Y21
 A2  B2  C2 Y22
=== === === ===

These fake predictions allow us to assess the influence of feature `A` with the following methods:

- For **pdplots**: we compute the mean `Y_i` of `Y_ij` for each dataframe, and plot `f(A_i) = Y_i`. This creates 1 line plot.
- For pure theoretical **ice plots**: we plot `f(A_i) = Y_ij` directly. This creates several line plots, as much as there are value of `j`.

  - *Note:* this method generates many line plots, which is too heavy for Plotly if the dataset has more than a few tens of rows. As most relevant datasets are much bigger than that, we chose to aggregate/cluster the lines and create specific lighter ice plots that show relevant information.

- For **ice box plots**: we make a box plot of the values `Y_ij` for each value `A_i`.

This package integrates discretization functions for usage on features that take a high number of different values (such as continuous features), to make computations not too expensive and results easy to interpret. For example discretizing feature `A` in `N` bins would require creating only `N` new dataframes.

This method allows better assessment of feature effect than simple predictions aggregations. For example in the case of correlated features, the plots will show which feature is actually used by the model and how. Interpreting **pdplots** is generally straightforward, whereas **ice plots** give more information.

*Note:* drawing **ice plots** for a classifier is only relevant if using prediction probability (and not simply prediction) as the `Y` output.

To study the influence of 2 features `A` and `B` simultaneously with interactions, we follow the same process, this time generating and aggregating `N x M` dataframes of predictions (features `A` and `B` take respectively `N` and `M` different values). We can draw **pdplots** as heatmaps, however ice plots are not possible.

Example
~~~~~~~

Here is an example of using the icecream module (directly from ``readml.icecream``) to draw PDP ans ICE plots ::

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
    # optionally customize icecream options
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
    # draw PDP
    pdp.draw(kind='pdp', show=True)
    # draw ICE plots
    pdp.draw(kind='ice', show=True)
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


Accumulated Local Effects Plots (from icecream)
-----------------------------------------------

Accumulated Local Effects plots are implemented in the :doc:`icecream` module.

How it works
~~~~~~~~~~~~

ALE plots show how the model predictions change in a small “window” of the feature around a certain grid value v for data instances in that window.

To estimate local effects, we divide the feature into many intervals and compute the average of the differences in the predictions.

- we first divide into intervals
- for each interval (i), we filter the dataset in order to keep only observations within this interval
- for the data instances (points) in an interval (i), we calculate the difference in the prediction when we replace the feature with the upper and lower limit of the interval
- these differences are later accumulated and centered, resulting in the ALE curve.

The value of the ALE can be interpreted as the main effect of the feature at a certain value compared to the average prediction of the data.
For example, an ALE estimate of -2 at x_j = 3 means that when the j-th feature has value 3, then the prediction is lower by 2 compared to the average prediction.

Imagine that we want to get the Accumulated Local Effects for a machine learning model that predicts the value of a house depending on the number of rooms and the size of the living area.
For the effect of living area at 30 m2:
- the ALE method uses all houses with about 30 m2 (between 29 m2 and 31 m1)
- it gets the model predictions pretending these houses were 31 m2 minus the predictions pretending they were 29 m2.
- then averages these differences

This gives us the pure effect of the living area and is not mixing the effect with the effects of correlated features (number of rooms). The use of differences blocks the effect of other features.
In addition to that, by filtering the dataset and keeping only data instances in the interval, we avoid generating unrealistic instances

ALE are different than PDP because:

* We only generate fake data on instances that are already within the interval
* We average the changes of predictions, not the predictions itself (the change is the differences in the predictions over an interval).
* ALE are centred at zero. This makes their interpretation easier, a positive value means a positive effect and a negative value can be interpreted as a negative effect.


Example
~~~~~~~
Here is an example of using the icecream module (directly from ``readml.icecream``) to draw ALE plots ::

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
    # create accumulated local effects and draw plots
    ale = icecream.IceCream(
        data=df[features],
        feature_names=features,
        bins={'sepal_length': 10},
        model=model,
        targets=df['label'],
        use_classif_proba=True,
        use_ale= True
    )
    ale.draw(kind='ale', show=True)


SHAP feature importance and summary plots
-----------------------------------------

SHAP (SHapley Additive exPlanations) by Lundberg and Lee (2016) is a game theoretic approach to explain the output of any machine learning model.
It connects optimal credit allocation with local explanations using Shapley values from game theory and their related extensions.
(see `repo <https://github.com/slundberg/shap>`_ and `paper <http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf>`_ for details)

The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each feature to the prediction.
The SHAP explanation method computes Shapley values from coalitional game theory. The feature values of a data instance act as players in a coalition.
Shapley values tell us how to fairly distribute the "payout" (= the prediction) among the features.
A player can be an individual feature value, e.g. for tabular data. A player can also be a group of feature values.
One innovation that SHAP brings to the table is that the Shapley value explanation is represented as an additive feature attribution method, a linear model.
That view connects LIME and Shapley Values.

SHAP has explainers for every type of model, we use 2 among them to explain globally ML models (we will discuss Deep SHAP in local interpretability section):

- **KernelSHAP:** uses a specially-weighted local linear regression to estimate SHAP values for any model. But it is slower than other Explainers and it offers an approximation rather than exact Shap values. (Note that to speed up computations, we summarize the train set with a set of weighted kmeans, each weighted by the number of points they represent.)
- **TreeSHAP:** a variant of SHAP for tree-based machine learning models such as decision trees, random forests and gradient boosted trees. TreeSHAP was introduced as a fast, model-specific alternative to KernelSHAP.

SHAP Feature Importance
~~~~~~~~~~~~~~~~~~~~~~~

The idea behind SHAP feature importance is simple: Features with large absolute Shapley values are important.
Since we want the global importance, we average the absolute Shapley values per feature across the data.
Next, we sort the features by decreasing importance and plot them.

SHAP feature importance is an alternative to permutation feature importance.
There is a big difference between both importance measures: Permutation feature importance is based on the decrease in model performance.
SHAP is based on magnitude of feature attributions.

The feature importance plot is useful, but contains no information beyond the importance. For a more informative plot, we need to look at the summary plot.

SHAP Summary Plot
~~~~~~~~~~~~~~~~~
The summary plot combines feature importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance.
The position on the y-axis is determined by the feature and on the x-axis by the Shapley value. The color represents the value of the feature from low to high. Overlapping points are jittered in y-axis direction, so we get a sense of the distribution of the Shapley values per feature. The features are ordered according to their importance.

Example
~~~~~~~
Here is an example of using ``ExplainML`` class form ``readml.explainers.ml.explain_ml`` to compute SHAP feature importance and summary plots.

An html report containing feature importance and summary plots (with a breve description) will be stored in out_path. ::

    from readml.explainers.ml.explain_ml import ExplainML
    exp = ExplainML(
            model=xgb_model,
            task_name="classification",
            tree_based_model=True,
            features_name=["f1", "f2", "f3", "f4", "f5"],
            features_to_interpret=["f1", "f2", "f3", "f4", "f5"], # not used for SHAP
            target_col="target",
            out_path="outputs_ml/",
        )
    exp.global_shap(df_train)

