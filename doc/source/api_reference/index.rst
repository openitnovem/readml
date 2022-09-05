=============
API reference
=============

This page gives an overview of all public classes, functions and methods.

Classes
-------
Many classes can be used to interpret any model.

**For ML interpretability:**

.. currentmodule:: readml.explainers.ml

.. autosummary::
   :toctree: generated/

   explain_ml.ExplainML
   shap_tree_explainer.ShapTreeExplainer
   shap_kernel_explainer.ShapKernelExplainer

You can also use the icecream module classes for custom visualizations.

.. currentmodule:: readml.icecream

.. autosummary::
   :toctree: generated/

   IceCream
   IceCream2D

**For DL interpretability:**

.. currentmodule:: readml.explainers.dl

.. autosummary::
   :toctree: generated/

   explain_dl.ExplainDL
   tabular_deep_explainer.TabularExplainer
   text_deep_explainer.TextExplainer
   image_deep_explainer.VisualExplainer
   grad_cam.GradCAM

Functions
---------

.. currentmodule:: readml.main

.. autosummary::
   :toctree: generated/

   interpret

.. currentmodule:: readml.explainers.core

.. autosummary::
   :toctree: generated/

   interpret_dl
   interpret_ml

.. warning::

    You should know some python to use the library.
