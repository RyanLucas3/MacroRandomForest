# MacroRandomForest


---------------------------------------------------------------------------------------------------------------------------------
*"Machine Learning is useful for macroeconomic forecasting but not so useful for macroeconomics" - Philippe Goulet Coulombe*
---------------------------------------------------------------------------------------------------------------------------------

Ever wanted the power of a random forest with the interpretability of a linear regression model? Well now you can...

This code base is the Python implementation of "The Macroeconomy as a Random Forest (MRF)" by Philippe Goulet Coulombe. 

MRF has at its core a basic linear regression equation that is intended to express a macroeconomic relationship. Unlike a regular linear regression though, in MRF the variable and splitting point (or window size) selections are not manually specified. Instead, these choices are made by a random forest. Bringing together the linear macro equation with the random forest ML algorithm means that our linear coefficient then nests time-varying indications of variable importance, information about the optimal splitting point and (by implication) regime-switching and structural breaks. 

Being directly interpretable, the model can provide value not only to economic forecasters, but also to macroeconomic policy makers.

The full paper corresponding to the implementation can be found here: https://arxiv.org/abs/2006.12724. 
