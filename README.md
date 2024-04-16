# MacroRandomForest

We documented this package in much more detail at: https://mrf-web.readthedocs.io/en/latest/index.html.

---------------------------------------------------------------------------------------------------------------------------------
*"Machine Learning is useful for macroeconomic forecasting but not so useful for macroeconomics" - Philippe Goulet Coulombe*
-------------------------------------------------------------------------------------------------------------------------------
![MRF_logo_2](https://user-images.githubusercontent.com/55145311/156574873-e72ef942-6979-4639-9089-9b2e06f7a80e.svg)

Ever wanted the power of a Random Forest with the interpretability of a Linear Regression model? Well now you can...

This code base is the official open-source Python implementation of "The Macroeconomy as a Random Forest (MRF)" by Philippe Goulet Coulombe. MRF is a time series modification of the canonical Random Forest Machine Learning algorithm. It uses a Random Forest to flexibly model time-varying parameters in a linear macro equation. This means that, unlike most Machine Learning methods, MRF is directly interpretable via its main output - what are known as Generalised Time Varying Parameters (GTVPs). 
  
The model has also shown forecasting gains over numerous alternatives and across many time series datasets. It is well suited to macroeconomic forecasting, but there are also many possible extensions to quantitative finance, or any other field of science with time series data. The full paper corresponding to the implementation can be found here: https://arxiv.org/abs/2006.12724

# How it works


How it works
============

This is a simple explanation of MRF, how it works and why it's useful. The algorithm is described in more detail in https://arxiv.org/abs/2006.12724.

Setup
--------

Within the modern ML canon, random forest is an extremely popular algorithm because it allows for complex nonlinearities, handles high-dimensional data, bypasses overfitting, and requires little to no tuning. However, while random forest gladly delivers gains in prediction accuracy (and ergo a conditional mean closer to the truth), it is much more reluctant to disclose its inherent model. 

MRF shifts the focus of the forest away from predicting :math:`y_t` into modelling :math:`\beta_t`, which are the economically meaningful coefficients in a time-varying linear macro equation. More formally:

.. math::

    \begin{equation*}
    \begin{aligned}
    y_t = X_t \beta_t  + \varepsilon_t
    \end{aligned}
    \end{equation*}
   
.. math::

   \begin{equation}
   \beta_t = \mathcal{F}(S_t)
   \end{equation}


Where :math:`S_t` are the state variables governing time variation and :math:`\mathcal{F}` is a forest. :math:`X_t` is typically a subset of :math:`S_t` which we want to emphasize and for which associated coefficients may be of economic interest. There are interesting special cases. For instance, :math:`X_t` could use lags of :math:`y_t` -- an autoregressive random forest (ARRF) – which will outperform RF when applied to persistent time series. Typically :math:`X_t \subset S_t` is rather small (and focused) compared to :math:`S_t`. 

The new algorithm comes with some benefits. First, it can be interpreted. Its main output, Generalized Time-Varying Parameters (GTVPs) is a versatile device nesting many popular nonlinearities (threshold/switching, smooth transition, structural breaks/change). In the end, we simply get a linear equation with time-varying coefficients following a very general law of motion. The latter is powered by a large data set, and an algorithm particularly apt with complex nonlinearities and high-dimensionality. 

By striking an appealing balance of efficiency and flexibility, it forecasts better. Most ML algorithms are designed for large cross-sectional data sets, whereas macroeconomics is characterized by short dependent time series. If persistence (or any other linear relationship) is pervasive, important efficiency gains ensue from modeling them directly. When measured against econometric approaches, MRF can again perform better, but now by being less rigid about :math:`\beta_t`’s law of motion and avoiding overfitting. 

Random Forest
--------

For those unfamiliar with random forests, the general fitting procedure involves firstly bootstrapping the data to create a random sub-sample of observations. In time series, this will be a set of time indices :math:`l` that becomes the parent node for our tree-splitting procedure. 

After randomising over rows, we then take a random subset of the predictors, call it :math:`\mathcal{J}^-`. MRF then performs a search for the optimal predictor and optimal splitting point. For each tree, we  implement least squares optimisation with a ridge penalty over :math:`j \in \mathcal{J}^{-}` and :math:`c \in \mathbb{R}`, where c is the splitting point. Mathematically, this becomes:

.. math::

    \begin{equation*}
    \begin{aligned}
    \begin{aligned}\label{OLS}
    (j^*, c^*) = \min _{j \in \mathcal{J}^{-}, \; c \in \mathbb{R}} &\left[\min _{\beta_{1}} \sum_{\left\{t \in l \mid S_{j, t} \leq c\right\}}\left(y_{t}-X_{t} \beta_{1}\right)^{2}+\lambda\left\|\beta_{1}\right\|_{2}\right.\\
     &\left.+\min _{\beta_{2}} \sum_{\left\{t \in l \mid S_{j, t}>c\right\}}\left(y_{t}-X_{t} \beta_{2}\right)^{2}+\lambda\left\|\beta_{2}\right\|_{2}\right] 
    \end{aligned}
    \end{aligned} \label{a} \tag{1}
    \end{equation*} 

Practically, optimisation over :math:`c` happens by sampling empirical quantiles of the predictor to be split. These become the possible options for the splits and we evaluate least squares repeatedly to find the optimal splitting point for a given predictor :math:`j`. In an outer loop, we take the minimum to find :math:`j^* \in \mathcal{J}^{-}` and :math:`c^* \in \mathbb{R}`.

This process is, in principle, a greedy search algorithm. A greedy algorithm makes locally optimal decisions, rather than finding the globally optimal solution.

.. image:: /images/Greedy_v_true.svg

However, various properties of random forests reduce the extent to which this is a problem in practice. First, each tree is grown on a bootstrapped sample, meaning that we are selecting many observation triplets :math:`[y_t, X_t, S_t]` for each tree that is fit. This means the trees are diversified by being fit on many different random subsamples. By travelling down a wide array of optimization routes, the forest safeguards against landing at a sub-optimal solution.

This problem is further alleviated in our context by growing trees semi-stochastically. In Equation :math:`\ref{a}`, this is made operational by using :math:`\mathcal{J}^{-} \in \mathcal{J}` rather than :math:`\mathcal{J}`. This means that at each step of the recursion, a different subsample of regressors is drawn to constitute candidates for the split. This prevents the greedy algorithm from always embarking on the same optimization route. As a result, trees are further diversified and computing time reduced.

Random Walk Regularisation
--------------------------

Equation :math:`\ref{a}` uses Ridge shrinkage which implies that each time-varying coefficient (:math:`\beta_t`) is implicitly shrunk to 0 at every point in time. This can be an issue if a process is highly persistent, since shrinking the first lag heavily to 0 can incur serious bias. :math:`\beta_i = 0` is a natural stochastic constraint in a cross-sectional setting, but its time series translation :math:`\beta_t = 0` can easily be suboptimal. The traditional regularisation employed in macro is rather the random walk:

.. math::
   
   \begin{equation*}
   \begin{aligned}
   \begin{aligned}
   \beta_t = \beta_{t-1} + u_t
   \end{aligned}
   \end{aligned} 
   \end{equation*} 

Thus it is desirable to transform Equation :math:`\ref{a}` so that that coefficients evolve smoothly, which entails shrinking :math:`\beta_t` to be in the neighborhood of :math:`\beta_{t-1}` and :math:`\beta_{t+1}` rather than 0. This is in line with the view that economic states last for at least a few consecutive periods.

This regularisation is implemented by taking the rolling-window view of time-varying parameters. That is, the tree, instead of solving a plethora of small ridge problems, will rather solve many weighted least squares (WLS) problems, which includes close-by observations. The latter are in the neighborhood (in time) of observations in the current leaf. They are included in the estimation, but are allocated a smaller weight. For simplicity and to keep the computational demand low, the kernel used by WLS is a simple symmetric 5-step Olympic podium.

Informally, the kernel puts a weight of 1 on observation  :math:`t`, a weight of :math:`\zeta < 1` for observations :math:`t-1` and :math:`t+1` and a weight of :math:`\zeta^2` for observations :math:`t-2` and :math:`t+2`. Since some specific :math:`t`'s will come up many times (for instance if observations :math:`t` and :math:`t+1` are in the same leaf), MRF takes the maximal weight allocated to :math:`t` as the final weight :math:`w(t; \zeta)`.

Formally, define :math:`l_{-1}` as the lagged version of the leaf :math:`l`. In other words :math:`l_{-1}` is a set containing each observation from :math:`l`, with all of them lagged one step. :math:`l_{+1}` is the "forwarded" version. :math:`l_{-2}` and :math:`l_{+2}` are two-steps equivalents. For a given candidate subsample :math:`l`, the podium is:

.. math::
   
   w(t ; \zeta)=\left\{\begin{array}{ll}
   1, & \text { if } t \in l \\
   \zeta, & \text { if } t \in\left(l_{+1} \cup l_{-1}\right) / l \\
   \zeta^{2}, & \text { if } t \in\left(l_{+2} \cup l_{-2}\right) /\left(l \cup\left(l_{+1} \cup l_{-1}\right)\right) \\
   0, & \text { otherwise }
   \end{array}\right.

Where :math:`\zeta < 1` is the tuning parameter guiding the level of time-smoothing. Then, it is only a matter of how to include those additional (but down weighted) observations in the tree search procedure. The usual candidate splitting sets: 

.. math::
   
   \begin{equation*}
   \begin{aligned}
   \begin{aligned}
   l_{1}(j, c) \equiv\left\{t \in l \mid S_{j, t} \leq c\right\} \quad \text { and } \quad l_{2}(j, c) \equiv\left\{t \in l \mid S_{j, t}>c\right\}
   \end{aligned}
   \end{aligned} 
   \end{equation*} 

are expanded to include all observations of relevance to the podium:

.. math::
   
   \begin{equation*}
   \begin{aligned}
   \begin{aligned}
   \text { for } i=1,2: \quad l_{i}^{RW}(j, c) \equiv l_{i}(j, c) \cup l_{i}(j, c)_{-1} \cup l_{i}(j, c)_{+1} \cup l_{i}(j, c)_{-2} \cup l_{i}(j, c)_{+2}
   \end{aligned}
   \end{aligned} 
   \end{equation*} 

The splitting rule then becomes:

.. math::
   
   \begin{equation*}
   \begin{aligned}
   \begin{aligned}
   (j^*, c^*) = \min _{j \in \mathcal{J}^{-}, c \in \mathbb{R}} & {\left[\min _{\beta_{1}} \sum_{t \in l_{1}^{R W}(j, c)} w(t ; \zeta)\left(y_{t}-X_{t} \beta_{1}\right)^{2}+\lambda\left\|\beta_{1}\right\|_{2}\right.} \\
   &\left.+\min _{\beta_{2}} \sum_{t \in l_{2}^{ RW}(j, c)} w(t ; \zeta)\left(y_{t}-X_{t} \beta_{2}\right)^{2}+\lambda\left\|\beta_{2}\right\|_{2}\right] 
   \end{aligned}
   \end{aligned} \label{b} \tag{2}
   \end{equation*} 
