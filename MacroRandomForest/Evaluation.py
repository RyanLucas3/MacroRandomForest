import numpy as np


def collect_errors(oos_pos, actual, forecasts, k=1):
    '''
    Collecting forecasting errors based on MRF forecasts and observed values of the target variable.

    Args:
        - oos_pos (numpy.array): Represents OOS period of interst for statistical evaluation. Passed in automatically when MRF.statistical_evaluation() is called.
        - actual (pd.Series): Actual (observed) values for the target variable over the OOS period. Passed in automatically when MRF.statistical_evaluation() is called.
        - forecasts (pd.Series): k-period forecasted values for the target variable.
        - k (int, default 1): Forecast horizon. 

    Returns:
        - errors (dict): Dictionary containing forecast errors corresponding to OOS period.
    '''

    errors = {}

    for t in oos_pos:

        actual_value = actual.loc[t+k]
        forecasted_value = forecasts.loc[t]

        errors[t] = actual_value - forecasted_value

    return errors


def get_MAE(error_dict, oos_pos):
    '''
    Calculating Mean Absolute Error (MAE) based on collected forecasting errors.

    Args:
        - error_dict (dict): List of forecasting errors obtained via collect_errors()
        - oos_pos (numpy.array): Time indices of OOS period

    Returns:
        - MAE (float)

    '''
    abs_errors = map(abs, list(error_dict.values()))
    sum_abs_errors = sum(abs_errors)
    return (1/len(oos_pos))*sum_abs_errors


def get_MSE(error_dict, oos_pos):
    '''
    Calculating Mean Squared Error (MSE) based on collected forecasting errors.

    Args:
        - error_dict (dict)
        - oos_pos (numpy.array)

    Returns:
        - MSE (float)

    '''
    errors_as_array = np.array(list(error_dict.values()))
    sum_squared_errors = sum(np.power(errors_as_array, 2))
    return (1/len(oos_pos))*sum_squared_errors


def get_sharpe_ratio(daily_profit):
    '''
    Calculating Sharpe Ratio financial return metric.

    Args:

        - daily_profit (pd.Series): Series corresponding to daily profit values obtained from financial_evaluation() and trading_strategy() functions

    Returns:
        - sharpe_ratio (float): Sharpe Ratio corresponding to OOS period

    '''
    mean = daily_profit.mean()
    std_dev = daily_profit.std()
    return 252**(0.5)*mean/std_dev


def get_max_dd_and_date(cumulative_profit):
    '''
    Calculating Maximum Drawdown financial return metric.

    Args:
        - cumulative_profit (pd.Series): Series corresponding to cumulative profit values obtained from financial_evaluation() and trading_strategy() functions

    Returns:
        - drawdown (float): Maximum Drawdown metric corresponding to OOS period
    '''

    rolling_max = (cumulative_profit+1).cummax()
    period_drawdown = (
        ((1+cumulative_profit)/rolling_max) - 1).astype(float)
    drawdown = round(period_drawdown.min(), 3)
    return drawdown


def get_annualised_return(cumulative_profit, T_profit):
    '''
    Calculating Annualised Return financial return metric.

    Args:

        - cumulative_profit (pandas.Series): Series corresponding to cumulative profit values obtained from financial_evaluation() and trading_strategy() functions.
        - T_profit: Time indices corresponding to profit-generating period. Note this starts k days after OOS start, since we need a previous signal to generate profit!

    Returns:
        - annualised_return (float): Yearly profit earned over OOS period
    '''
    return cumulative_profit.iloc[-1]*(252/len(T_profit))


def trading_strategy(model_forecasts, stock_price, t, k=1):
    '''

    Strategy for generating binary (long/short) trading signals based on MRF predictions. This strategy is agnostic to the forecast-horizon used. 

    Args:
        - forecasts (pd.Series): k-period forecasted values for the target variable.
        - stock_price (pd.Series): Series of stock prices corresponding to target variable (returns) during same OOS period.
        - k (int, default 1): Forecast horizon.

    Returns:
        - PL_t (pd.Series): Backtested daily profit corresponding to implementing MRF trading signals.

    '''
    PL_t = 0
    signal_t_minus_1 = 0

    # Long if return prediction > 0 ; otherwise short.

    for i in range(1, k+1):
        if model_forecasts.loc[t-i].values > 0:
            signal_t_minus_1 += 1
        elif model_forecasts.loc[t-i].values < 0:
            signal_t_minus_1 -= 1

    PL_t += (1/k)*signal_t_minus_1 * \
        ((stock_price.loc[t] - stock_price.loc[t-1])/stock_price.loc[t-1])

    return PL_t
