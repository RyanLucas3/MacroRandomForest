# MacroRandomForest


---------------------------------------------------------------------------------------------------------------------------------
*"Machine Learning is useful for macroeconomic forecasting but not so useful for macroeconomics" - Philippe Goulet Coulombe*
---------------------------------------------------------------------------------------------------------------------------------
![MRF_v9 (1)-1](https://user-images.githubusercontent.com/55145311/156152978-a76976b6-d03c-43e4-8e5c-7047c5b6a0ea.png)

Ever wanted the power of a random forest with the interpretability of a linear regression model? Well now you can...

This code base is the Python implementation of "The Macroeconomy as a Random Forest (MRF)" by Philippe Goulet Coulombe. 

MRF has at its core a basic linear regression equation that is intended to express a macroeconomic relationship. Unlike a regular linear regression though, in MRF our predictors are not entirely manually specified. Instead, they are chosen from our state variables by a random forest. 

The beauty is that, bringing together the linear macro equation with the random forest ML algorithm means that our linear coefficient then nests important time series information. This parameter can provide a transparent variable importance measure while at the same time accounting for complex non-linearities in the data. 

Thus being both powerful and directly interpretable, the model can provide value not only to economic forecasters, but also to macroeconomic policy makers.

The full paper corresponding to the implementation can be found here: https://arxiv.org/abs/2006.12724. 



