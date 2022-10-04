# SPDX-FileCopyrightText: 2022 GroupeSNCF 
#
# SPDX-License-Identifier: Apache-2.0

SECTIONS_HTML = {
    "COMMUN": [
        "Global interpretability consists in understanding and explaining the learning of the model. This step is essential because it allows the model to be debugged, biases to be detected, and the decisions of the model to be trusted once in production...\n",
        "Partial dependency plots (PDP), ICE plots and ALE plots are a way of quantifying the impact of one (or two) features on a machine learning model. They allow to plot, feature by feature, the influence of a variation of this feature on the model.\n",
        "In the case of correlated features, it is advisable to use ALE plots instead of PDPs.\n",
        "SHAP plots (global explanation) allow to know the feature importance but also how the feature values have impacted the model learning. It is preferable to have a good quality model before launching these plots. Indeed, there is little point in explaining an unreliable model.\n",
        "These plots can be drawn from training data, test data or even unlabelled data; the only condition being that these data are representative of the normal distribution of the data.\n"
    ],
    "PDP": [
        "Partial dependence plots (PDPs) show the marginal effect that one or two features have on the predictions of a machine learning model. A PDP is the average of the lines in an ICE plot. \n",
        "In the case of feature non-correlation, the interpretation is simple: the PDP plots show how the average prediction changes when the n-th feature changes. We vary a feature and visualise the changes that this variation causes in the predictions. We are therefore measuring a causal relationship between the feature and the prediction.\n",
    ],
    "ICE": [
        "An ICE (Individual Conditional Expectation) plot visualises the dependence of the prediction on a feature for each observation separately, giving a curve per instance, as opposed to an overall curve in partial dependence plots. A PDP is the average of the lines in an ICE plot.\n",
        "ICE curves are more intuitive and easier to interpret than PDPs. Each curve represents the predictions for a given value if the feature in question is varied.\n",
        "In practice, it is advisable to use PDPs and ICE plots together for better interpretation.\n",
    ],
    "ALE": [
        "ALEs (Accumulated Local Effects) describe on average - based on the conditional distribution of features - the influence of features on the prediction of a Machine Learning model.\n",
        "This method provides model-agnostic global explanations for classification and regression models on tabular data. ALE plots are a faster and less biased alternative to PDP plots.\n",
        "If we take the example of predicting house prices, to estimate the effect of a house size of 30 m², the ALE method will take into account all houses that have an area of about 30 m² and calculate the difference between the price predictions of houses that have an area of 31 m² and the price predictions of houses that have an area of 29 m². This gives us the pure effect of the feature area and does not mix this effect with the effect of the other features that are correlated with it. Using the difference between the predictions eliminates the effect of the other features and makes the method more robust to feature correlation.\n",
    ],
    "SHAP": [
        "SHAP values are used to show how the values of each feature contribute (positively or negatively) to the target variable. SHAP allows two important feature plots to be drawn, the 'Summary bar plot' and the 'Summary bee-swarm plot'.\n",
        'The summary bar plot lists the most significant/impacting features in descending order.\u200b For each feature, the absolute average SHAP value for each feature is calculated, allowing the impact of the features on the model output to be compared.\n',
        'The "Summary bee-swarm plot" uses the SHAP values to show the distribution of impacts that each feature has on the model output. This plot aggregates several pieces of information:\u200b\n',
        "- Feature importance: features are ranked in descending order according to their impact on the model (the sum of SHAP values over the whole dataset).\u200b\n",
        "- Impact: The horizontal axis shows if the effect of a value is positive or negative on the prediction, it represents for each feature the sum of the SHAP values on the whole dataset.\u200b\n",
        "- The values of each feature: the colours show if the feature values are high (blue) or low (red).\u200b\n",
    ],
}
