import numpy as np


def collect_errors(oos_pos, actual, forecasts, k=1):
    '''
    Collecting forecasting errors based on MRF forecasts and observed values of the target variable.
    '''

    errors = {}

    for t in oos_pos:

        actual_value = actual.loc[t+k]
        forecasted_value = forecasts.loc[t]

        errors[t] = actual_value - forecasted_value

    return errors


def get_MAE(error_dict, T_model):
    '''
    Calculating Mean Absolute Error (MAE) based on collected forecasting errors.
    '''
    abs_errors = map(abs, list(error_dict.values()))
    sum_abs_errors = sum(abs_errors)
    return (1/len(T_model))*sum_abs_errors


def get_MSE(error_dict, T_model):
    '''
    Calculating Mean Squared Error (MAE) based on collected forecasting errors.
    '''
    errors_as_array = np.array(list(error_dict.values()))
    sum_squared_errors = sum(np.power(errors_as_array, 2))
    return (1/len(T_model))*sum_squared_errors


def get_sharpe_ratio(daily_profit):
    '''
    Calculating Sharpe Ratio financial return metric.
    '''
    mean = daily_profit.mean()
    std_dev = daily_profit.std()
    return 252**(0.5)*mean/std_dev


def get_max_dd_and_date(cumulative_profit):
    '''
    Calculating Maximum Drawdown financial return metric.
    '''
    rolling_max = (cumulative_profit+1).cummax()
    period_drawdown = (
        ((1+cumulative_profit)/rolling_max) - 1).astype(float)
    drawdown = round(period_drawdown.min(), 3)
    return drawdown


def get_annualised_return(cumulative_profit, T_profit):
    '''
    Calculating Annualised Return financial return metric.
    '''
    return cumulative_profit.iloc[-1]*(252/len(T_profit))


def trading_strategy(model_forecasts, stock_price, t, k=1):
    '''
    Strategy for generating binary (long/short) trading signals based on MRF predictions. This strategy is agnostic to the forecast-horizon used. 
    '''
    PL_t = 0
    signal_t_minus_1 = 0

    # Long if return prediction > 0 ; otherwise short.
    for i in range(1, k+1):
        if model_forecasts.loc[t-i, "Ensembled_Prediction"] > 0:
            signal_t_minus_1 += 1
        elif model_forecasts.loc[t-i, "Ensembled_Prediction"] < 0:
            signal_t_minus_1 -= 1
            
    PL_t += (1/k)*signal_t_minus_1 * \
        ((stock_price[t] - stock_price[t-1])/stock_price[t-1])

    return PL_t
