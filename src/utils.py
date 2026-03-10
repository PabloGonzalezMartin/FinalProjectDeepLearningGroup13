import pandas as pd

def train_test_split_ts(df, train_size=None, split_date=None):
    """
    Split a time series DataFrame into train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex or PeriodIndex
    train_size : float, optional
        Percentage for training (e.g., 0.8 for 80%). Used if split_date is None.
    split_date : str, optional
        Fixed date to split on (e.g., '2020-01', '2020-06-15').
        Everything before this date is train, from this date onward is test.
        Takes priority over train_size if both are provided.
    
    Returns:
    --------
    train : pd.DataFrame
    test : pd.DataFrame
    """

    if split_date is not None:
        split = pd.Period(split_date, freq=df.index.freq) if isinstance(df.index, pd.PeriodIndex) else pd.to_datetime(split_date)
        train = df[df.index < split]
        test = df[df.index >= split]
    elif train_size is not None:
        n = len(df)
        split_idx = int(n * train_size)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
    else:
        raise ValueError("Provide either 'train_size' (e.g., 0.8) or 'split_date' (e.g., '2020-01').")

    print(f"Train: {train.index[0]} → {train.index[-1]}  ({len(train)} rows, {len(train)/len(df)*100:.1f}%)")
    print(f"Test:  {test.index[0]} → {test.index[-1]}  ({len(test)} rows, {len(test)/len(df)*100:.1f}%)")

    return train, test

def create_lagged_df(df, column, lags):
    """
    Create a temporary DataFrame with lagged columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original DataFrame
    column : str
        Column name to create lags for
    lags : list
        List of lag values (e.g., [1, 3, 7, 12])
    
    Returns:
    --------
    pd.DataFrame
        New DataFrame with original column and lagged columns
    """
    import pandas as pd
    
    temp_df = df.copy()
    cols = []
    for lag in lags:
        col_name = f'{column}_lag_{lag}'
        temp_df[col_name ] = df[column].shift(lag)
        cols.append(col_name)
    return temp_df.dropna(), cols  # Remove rows with NaN from shifting
