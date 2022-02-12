# MacroRandomForest

*"Machine Learning is very useful for macroeconomic forecasting but not so useful for macroeconomics" - Phillipe Goulet Coulombe*
---------------------------------------------------------------------------------------------------------------------------------

Ever wanted the power of a random forest with the interpretability of a linear regression model? Well now you can...

This code base is the Python implementation of "The Macroeconomy as a Random Forest (MRF)" by Philippe Goulet Coulombe. MRF has, at it's core, a linear macro equation. Unlike a regular regression though, in MRF variable and window size selection are made via a random forest. 

Bringing together the linear macro equation with the random forest ML algorithm means that our linear coefficient then nests information about variable selection, time-variation, regime-switching and structural breaks. This parameter is directly interpretable and can provide value not only to economic forecasters, but also to economic policy makers.


The full paper corresponding to the implementation can be found here: https://arxiv.org/abs/2006.12724. 
