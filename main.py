# Interview project for Cubist
# Franklin Wang

# Objective: predict daily stock volume
# Target: 20d z-score of Log Volume (LogVol - mean(LogVol, 20)) / stdev(LogVol, 20)
#   Volume is "spiky" like volatility. Natural log transformation makes it better-behaved for linear modeling.
#   Volume is highly persistent. AR plot shows a long tail (60d ~= 0.5, 260d ~= 0.3). Log volume exhibits long-term
#       trends / drifts that presents stationarity problems. This is avoided by modeling its z-score instead.
#   Why 20d? No hard and fast rule here. I picked 20 to be broadly in line with the typical half life of volatility in
#       asset prices.
# Features
#   Past volume: non-overlapping rolling windows to reduce correlation
#   Past volatility (OHLC, RV): volatility tends to be correlated with volume (both are indicative of new information)
#   Past return: this comes from leverage effect for volatility (higher return -> lower volatility) maybe also works
#       volume?
#   Seasonality: day of week, day of month, day before/after holidays
#   Cross terms: crossing existing features with stock characteristics (sp_weight) would be interesting,
#       i.e. do larger-cap stocks react differently to past volume/vol/return/seasonality?
# Fitting
#   CV design: rolling train/validation periods to account for time series. Sample weight is exponentially decayed
#      To illustrate: Train 2000-2001, validate 2002; Train 2000-2002, validate 2003, ... ;
#           Train 2010-2016, validate 2017. Hold out 2018-2019 for test.
#   Fitting method: Lasso (ridge yields very similar results)
#   Hyper-param: exponential decay speed, Lasso alpha


# code starts here

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS

DATA_PATH = 'C:\\Users\\LucyYu\\Downloads\\sp500.h5'
RANDOM_STATE = 0

TEST_SIZE = 0.2
CV = 4
EPSILON = 1e-12


def normalize(df, halflife=60):
    assert df.head().index.names == ['date']
    df_mean = df.shift(1).ewm(halflife=halflife, min_periods=halflife).mean()
    df_std = df.shift(1).ewm(halflife=halflife, min_periods=halflife).std()
    z = (df - df_mean) / (df_std + EPSILON)  # if inputs are all zeros return zero (happens for dummy variables)
    return z


def inv_normalize(z, df_actual, halflife=60):
    assert z.head().index.names == ['date']
    df_mean = df_actual.shift(1).ewm(halflife=halflife, min_periods=halflife).mean()
    df_std = df_actual.shift(1).ewm(halflife=halflife, min_periods=halflife).std()
    df = z * (df_std + EPSILON) + df_mean  # if inputs are all zeros return zero (happens for dummy variables)
    return df


def load_data():
    print('{}: load raw data...'.format(pd.Timestamp.now()))
    data = pd.read_hdf(DATA_PATH)

    # add helper fields
    data['log_volume'] = np.log1p(data['volume'])
    data['price_prev_close'] = data['price_close'] / (data['ret_raw'] + 1)
    # Yang and Zhang 2000 OHLC vol estimator
    data['ohlc_1d'] = np.log(data['price_open'] / data['price_prev_close']) ** 2 + \
                      0.5 * np.log(data['price_high'] / data['price_low']) ** 2 + \
                      (1 - 2 * np.log(2)) * np.log(data['price_close'] / data['price_open']) ** 2
    # simple close-to-close RV estimator
    data['rv_1d'] = np.log(data['price_close'] / data['price_prev_close']) ** 2
    dates = data.index.get_level_values(0)
    data['is_monday'] = (dates.dayofweek == 0).astype(int)
    data['is_tuesday'] = (dates.dayofweek == 1).astype(int)
    data['is_thursday'] = (dates.dayofweek == 3).astype(int)
    data['is_friday'] = (dates.dayofweek == 4).astype(int)
    data['is_month_end'] = dates.is_month_end.astype(int)
    data['is_month_start'] = dates.is_month_start.astype(int)
    holidays = set(pd.bdate_range(start=dates[0], end=dates[-1])) - set(dates)
    holidays = pd.DatetimeIndex(holidays).sort_values()
    data['day_before_holiday'] = dates.isin(holidays - pd.offsets.Day(1)).astype(int)
    data['day_after_holiday'] = dates.isin(holidays + pd.offsets.Day(1)).astype(int)
    return data


def generate_features(df):
    print('{}: generating features...'.format(pd.Timestamp.now()))
    df_unstacked = df.unstack('uspn').sort_index()
    assert df_unstacked.head().index.names == ['date']
    log_volume = df_unstacked['log_volume']
    ohlc = df_unstacked['ohlc_1d']
    rv = df_unstacked['rv_1d']
    ret = np.log1p(df_unstacked['ret_raw'])
    mktcap = df_unstacked['sp_weight'].shift(1)
    features = dict()
    volume = df_unstacked['volume']
    vlm_pred_naive = volume.ewm(halflife=60, min_periods=60).mean()
    target = volume / vlm_pred_naive - 1
    for label in ['is_monday', 'is_tuesday', 'is_thursday', 'is_friday', 'is_month_end', 'is_month_start',
                  'day_before_holiday', 'day_after_holiday']:
        features[label] = df_unstacked[label]
    for start_, end_ in [(0, 1), (1, 2), (2, 5), (5, 20), (20, 60)]:
        features['log_volume_' + str(start_) + '_' + str(end_)] = log_volume.rolling(end_ - start_).mean().shift(
            start_ + 1)
        features['ret_' + str(start_) + '_' + str(end_)] = ret.rolling(end_ - start_).mean().shift(start_ + 1)
        features['ohlc_' + str(start_) + '_' + str(end_)] = ohlc.rolling(end_ - start_).mean().shift(start_ + 1) ** 0.5
        features['rv_' + str(start_) + '_' + str(end_)] = rv.rolling(end_ - start_).mean().shift(start_ + 1) ** 0.5
    features_list = list(features.keys())
    for f in features_list:
        if f == 'target':
            continue
        features['mktcap_x_' + f] = mktcap * features[f]
    features = pd.concat(features, axis=1)
    features = normalize(features)
    features = features.stack('uspn').sort_index().dropna()
    target = target.stack('uspn').reindex(index=features.index)
    vlm_pred_naive = vlm_pred_naive.stack('uspn').reindex(index=features.index)
    return target, features, vlm_pred_naive


def score_model(params, features, target, raw_target, cv_slices):
    param_grid_size = len(params['alpha']) * len(params['hl_days'])
    raw_target_unstacked = raw_target.unstack('uspn').sort_index()
    dates = features.index.get_level_values(0)
    scores = {}
    counter = 0
    for alpha in params['alpha']:
        for hl_days in params['hl_days']:
            counter += 1
            print('{}: fitting param grid id={}/{}'.format(pd.Timestamp.now(), counter, param_grid_size))
            predicted = []
            for i, (train_slicer, validate_slicer) in enumerate(cv_slices):  # last 2 slices are for test
                last_train_date = features.index[train_slicer][-1][0]
                sample_weight = np.exp((dates[train_slicer] - last_train_date) / pd.Timedelta(days=hl_days) * np.log(2))
                lasso_model = make_pipeline(StandardScaler(),
                                            linear_model.Lasso(alpha=alpha, random_state=RANDOM_STATE))
                lasso_model.fit(features[train_slicer], target[train_slicer],
                                lasso__sample_weight=sample_weight)
                predicted_ = lasso_model.predict(features[validate_slicer])
                predicted.append(pd.Series(predicted_, index=target[validate_slicer].index))
            predicted = pd.concat(predicted, axis=0)
            # inverse transform target -> raw target
            predicted = np.expm1(predicted)
            # run regression for evaluation
            score_data = pd.DataFrame({'vlm_pred/vlm_pred_naive-1': predicted, 'vlm/vlm_pred_naive-1': raw_target}).dropna()
            model = OLS(endog=score_data['vlm/vlm_pred_naive-1'], exog=score_data['vlm_pred/vlm_pred_naive-1']).fit()
            scores[(alpha, hl_days)] = model.rsquared, model.params.iloc[0], model.tvalues.iloc[0]
    scores = pd.Series(scores)
    scores.index.set_names(['alpha', 'sample weight halflife'], inplace=True)
    scores = scores.unstack(0).sort_index()
    print(scores.applymap(lambda x: [np.round(a, 4) for a in x]))


def run_master():
    df = load_data()
    raw_target, features, vlm_pred_naive = generate_features(df)
    # raw_target_unstacked = raw_target.unstack('uspn').sort_index()
    # target_unstacked = normalize(np.log1p(raw_target_unstacked.clip(lower=-0.9)))  # transform Y for fitting (this will be reversed later)
    # target = target_unstacked.stack('uspn').sort_index()
    target = np.log1p(raw_target.clip(lower=-0.9))
    features = features.clip(lower=-3, upper=3)  # simple outlier removal

    # Time series CV sampling
    # the last 2 slcies are reserved for testing (2018-2019); hyper-parameter tuning uses in-sample data (16 slices)
    dates = features.index.get_level_values(0)
    cv_slices = []
    for year in range(2002, 2020):
        validate = str(year)
        train_start = '2000'
        train_end = str(year-1)
        train_slicer = dates.slice_indexer(start=train_start, end=train_end)
        validate_slicer = dates.slice_indexer(start=validate, end=validate)
        cv_slices.append((train_slicer, validate_slicer))

    # parameter grid
    params = {
        'hl_days': [30, 180, 365, 365*3],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001]
    }
    params = {
        'hl_days': [180], # [30, 180, 365, 365*3],
        'alpha': [0.001], # [1, 0.1, 0.01, 0.001, 0.0001]
    }
    # hyperparam tuning
    score_model(params, features, target, raw_target, cv_slices[:-2])

    # test
    params = {
        'hl_days': [180], # [30, 180, 365, 365*3],
        'alpha': [0.001], # [1, 0.1, 0.01, 0.001, 0.0001]
    }
    score_model(params, features, target, raw_target, cv_slices[-2:])


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     run_master()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
