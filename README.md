# MacroRandomForest


---------------------------------------------------------------------------------------------------------------------------------
*"Machine Learning is useful for macroeconomic forecasting but not so useful for macroeconomics" - Philippe Goulet Coulombe*
---------------------------------------------------------------------------------------------------------------------------------

Ever wanted the power of a random forest with the interpretability of a linear regression model? Well now you can...

This code base is the Python implementation of "The Macroeconomy as a Random Forest (MRF)" by Philippe Goulet Coulombe. 

MRF has at its core a basic linear regression equation that is intended to express a macroeconomic relationship. Unlike a regular linear regression though, in MRF our predictors are not manually specified. Instead, they are chosen by a random forest. 

Bringing together the linear macro equation with the random forest ML algorithm means that our linear coefficient then nests important time series information. This parameter can provide 1. a time-varying variable importance measure, 2. information about the optimal splitting point, and 3. an indication of the presence of regime-switching and structural breaks in the time series. 

Thus being powerful and directly interpretable, the model can provide value not only to economic forecasters, but also to macroeconomic policy makers.

The full paper corresponding to the implementation can be found here: https://arxiv.org/abs/2006.12724. 
