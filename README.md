# MacroRandomForest

We documented this package in much more detail at: https://mrf-web.readthedocs.io/en/latest/index.html.

---------------------------------------------------------------------------------------------------------------------------------
*"Machine Learning is useful for macroeconomic forecasting but not so useful for macroeconomics" - Philippe Goulet Coulombe*
-------------------------------------------------------------------------------------------------------------------------------
![MRF_logo_2](https://user-images.githubusercontent.com/55145311/156574873-e72ef942-6979-4639-9089-9b2e06f7a80e.svg)

# How it works

Random forest is an extremely popular algorithm because it allows for complex nonlinearities, handles high-dimensional data, bypasses overfitting, and requires little to no tuning. However, while random forest gladly delivers gains in prediction accuracy (and ergo a conditional mean closer to the truth), it is much more reluctant to disclose its inherent model. 

MRF shifts the focus of the forest away from predicting $y_t$ into modelling $\beta_t$, which are the economically meaningful coefficients in a time-varying linear macro equation. More formally:

$$y_t = X_t \beta_t  + \varepsilon_t$$

$$\beta_t = \mathcal{F}(S_t)$$

Where $S_t$ are the state variables governing time variation and $\mathcal{F}$ is a forest. $X_t$ is typically a subset of $S_t$ which we want to emphasize and for which associated coefficients may be of economic interest. There are interesting special cases. For instance, $X_t$ could use lags of $y_t$ -- an autoregressive random forest (ARRF) – which will outperform RF when applied to persistent time series. Typically $X_t \subset S_t$ is rather small (and focused) compared to $S_t$. 

The new algorithm comes with some benefits. First, it can be interpreted. Its main output, Generalized Time-Varying Parameters (GTVPs) is a versatile device nesting many popular nonlinearities (threshold/switching, smooth transition, structural breaks/change). In the end, we simply get a linear equation with time-varying coefficients following a very general law of motion. The latter is powered by a large data set, and an algorithm particularly apt with complex nonlinearities and high-dimensionality. 

By striking an appealing balance of efficiency and flexibility, it forecasts better. Most ML algorithms are designed for large cross-sectional data sets, whereas macroeconomics is characterized by short dependent time series. If persistence (or any other linear relationship) is pervasive, important efficiency gains ensue from modeling them directly. When measured against econometric approaches, MRF can again perform better, but now by being less rigid about $\beta_t$'s law of motion and avoiding overfitting. 
