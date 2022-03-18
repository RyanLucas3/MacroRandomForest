# MacroRandomForest


---------------------------------------------------------------------------------------------------------------------------------
*"Machine Learning is useful for macroeconomic forecasting but not so useful for macroeconomics" - Philippe Goulet Coulombe*
-------------------------------------------------------------------------------------------------------------------------------
![MRF_logo_2](https://user-images.githubusercontent.com/55145311/156574873-e72ef942-6979-4639-9089-9b2e06f7a80e.svg)

Ever wanted the power of a Random Forest with the interpretability of a Linear Regression model? Well now you can...

This code base is the official open-source Python implementation of "The Macroeconomy as a Random Forest (MRF)" by Philippe Goulet Coulombe. MRF is a time series modification of the canonical Random Forest Machine Learning algorithm. It uses a Random Forest to flexibly model time-varying parameters in a linear macro equation. This means that, unlike most Machine Learning methods, MRF is directly interpretable via its main output - what are known as Generalised Time Varying Parameters (GTVPs). 
  
The model has also shown forecasting gains over numerous alternatives and across many time series datasets. It is well suited to macroeconomic forecasting, but there are also many possible extensions to quantitative finance, or any other field of science with time series data. The full paper corresponding to the implementation can be found here: https://arxiv.org/abs/2006.12724

