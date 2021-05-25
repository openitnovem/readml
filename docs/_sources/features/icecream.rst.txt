Icecream
========

🍨 Individual Conditional Expectation Charts by REAdme teaM 🍦

**icecream** is a module that aims at explaining how a machine learning model works. It offers ways of assessing the influence of features on a model using methods such as Partial Dependency Plots [1]_, Individual Conditional Expectation [2]_ and Accumulated Local Effects [3]_ . For a specific feature, it can draw the influence of its values on the output of the model. It helps:

- identify which features influence the model
- how they influence the model (what values)
- compare how the model reacts versus how the target reacts (for example some features would show an influence on the target but are actually not used by the model)

This module can be applied to any supervised learning model, however it presents the greatest value on "black box" models. The implemented methods are superior to simple feature importance assessment methods.

This module offers two classes ``IceCream`` and ``IceCream2D``, it can be imported from ``fbd_interpreter.icecream``

Plots are created with Plotly_.

Recommandations
~~~~~~~~~~~~~~~

It is recommended to use this package on well performing machine learning models. Indeed, there is not much interest in explaining unreliable models. However, the visualizations can be drawn using training, test or even unlabeled data, the only condition being that the data is representative of its normal distribution.

If the dataset is too large, sampling may be needed before using this package (a dataset of 1 million rows is a good limit). What is important is that the distribution of the studied features is representative of what is seen globally in the data.

Regression and binary classification models are supported. Multiclass models are not supported yet, it is recommended to use a "one versus all" approach to explain a multiclass model.

Continuous features are automatically discretized before computations are applied. The API allows specifying the number of bins or directly bin edges (allowing for non-uniform bin width). Discretization can also be quantile-based so that all bins contain the same number of examples.

The discretization and aggregations abilities of this package are deeply tied to those of pandas_ and its ``DataFrame`` object. And the prediction abilities are tied to the scikit-learn_ supervised learning API. Thus, this package requires a dataset stored as a dataframe, and the model must have a prediction method so that ``model.predict(dataframe)`` returns the predictions as an iterable. The ``predict_proba()`` method available for some classification models can be used too.

If preprocessing is applied on data before modelling, it is advised to build a scikit-learn ``Pipeline`` instance containing all preprocessing and modelling steps. Such pipeline object would have prediction methods that can be used on the original intelligible data.

If the model is not available to the user, this package can also directly use predictions to plot aggregated values. However results will not be as explanatory as when using the model because, when available, the model is used to make new specific predictions.

We also recommend using Accumulated Local Effects plots in case of numerical features because ALE plots are unbiased when features are correlated.

.. _pandas : http://pandas.pydata.org/
.. _scikit-learn: https://scikit-learn.org/stable/

Using icecream
~~~~~~~~~~~~~~

Explaining a model using **icecream** consists in two steps:

1. Creation of a ``IceCream`` or ``IceCream2D`` instance given data, model, bin definitions, targets and options. All computations and aggregations are conducted on instance creation.
2. Plot drawing and saving using method ``draw()`` of created instance. Several plots are available (PDP, ICE & ALE plots):

  - for ``IceCream``, 1 chart is created per feature:

    - **pdp**: aggregated targets and predictions (mean or median) for each feature value, light in resources.
    - **ale**: mean of differences in predictions between min and max of each bin, it shows how the model predictions change in a small “window” of the feature values.
    - **ice**: clustered lines showing predictions, clustering method can be heavy in CPU, but final plot is light in resources; available clustering methods:

      - **kmeans**: KMeans clustering of predictions, lines are clusters centers, line widths are number of predictions for each cluster; heaviest in CPU but generally most revealing
      - **quantiles**: division of predictions in quantiles, lines are quantiles limits, lighter in CPU and still revealing
      - **random**: random selection of predictions, lightest in CPU but not reliable

    - **box**: distributions of predictions as boxes for each feature value, heavier in resources.

  - for ``IceCream2D``, all charts are created using the 2 given features:

    - **hist**: heatmap of aggregated targets and predictions for feature values with histograms of feature values on the sides
    - **scatter**: heatmap of aggregated targets and predictions for feature values with overlaid scatter plot of number of feature values

3. The created ``IceCream`` instance contains all discretizations, aggregations, predictions and samples that are used to create the charts.

*Notes:*

  - Number of lines in **ice** plots should not be too high, because Plotly becomes heavy with too many traces. 30 lines would be the recommended maximum.
  - If using **quantiles** method, number of lines should be odd so that a median line is drawn.

Options
~~~~~~~

**icecream** functions use custom options stored in a global configuration object called ``options``. These options mostly affect the automatic discretization functions and the chart attributes. Calling ``icecream.options`` shows available options and current values. Values can be customized with statement:

    icecream.options.<field_name> = <value>


References
----------

.. [1] Hastie, T., Tibshirani, R. and Friedman, J. (2009). The Elements of Statistical Learning Ed. 2. New York: Springer.
.. [2] Goldstein, A., Kapelner, A., Bleich, J., Pitkin, E. (2013). Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation. https://arxiv.org/abs/1309.6392
.. [3] Molnar, Christoph. (2019). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. https://christophm.github.io/interpretable-ml-book/
.. _Plotly: https://plot.ly/python/