import numpy as np


def DV_fun(sse, DV_pref=0.25):
    '''
    Implementing a middle of the range preference for middle of the range splits.

    Args:
        sse (np.array): Sum of Squared Errors obtained from split-sample OLS.
        DV_pref (float): Parameter controlling the rate of down-voting. 
    '''

    seq = np.arange(1, len(sse)+1)
    down_voting = 0.5*seq**2 - seq
    down_voting = down_voting/np.mean(down_voting)
    down_voting = down_voting - min(down_voting) + 1
    down_voting = down_voting**DV_pref

    return sse*down_voting


def standard(Y):
    '''
    Function to standardise the data. Remember we are doing ridge.

    Args: 
        - Y (np.matrix): Matrix of variables to standardise.

    Returns:
        - Standardised Data (dict): Including standardised matrix ("Y"), mean ("mean") and standard deviation "std"
    '''

    Y = np.matrix(Y)
    size = Y.shape
    mean_y = Y.mean(axis=0)
    sd_y = Y.std(axis=0, ddof=1)
    Y0 = (Y - np.repeat(mean_y,
                        repeats=size[0], axis=0)) / np.repeat(sd_y, repeats=size[0], axis=0)

    return {"Y": Y0, "mean": mean_y, "std": sd_y}
