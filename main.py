# Interview project for Cubist
# Franklin Wang


# code starts here

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS

DATA_PATH = 'C:\\Users\\LucyYu\\Downloads\\sp500.h5'
RANDOM_STATE = 0
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
    volume = df_unstacked['volume']
    log_volume = df_unstacked['log_volume']
    ohlc = df_unstacked['ohlc_1d']
    rv = df_unstacked['rv_1d']
    ret = np.log1p(df_unstacked['ret_raw'])
    mktcap_lagged = df_unstacked['sp_weight'].shift(1).stack('uspn').sort_index()
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
    features = pd.concat(features, axis=1)
    features = normalize(features)
    features = features.stack('uspn').sort_index().dropna()
    for f in features.columns:
        if f == 'target':
            continue
        features['mktcap_x_' + f] = mktcap_lagged * features[f]
    target = target.stack('uspn').reindex(index=features.index)
    vlm_pred_naive = vlm_pred_naive.stack('uspn').reindex(index=features.index)
    return target, features, vlm_pred_naive


def predict(features, target, raw_target, cv_slices, **params):
    dates = features.index.get_level_values(0)
    alpha = params['alpha']
    hl_days = params['hl_days']
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
    return score_data


def score(score_data):
    dates = score_data.index.get_level_values(0)
    ranks = (dates[1:] != dates[:-1]).cumsum()
    ranks = np.append([0], ranks)
    # std err is robust to auto-correlation as well as clustering by time
    # max lag at 20 days is very conservative
    model = OLS(endog=score_data['vlm/vlm_pred_naive-1'], exog=score_data['vlm_pred/vlm_pred_naive-1']). \
        fit(cov_type='hac-groupsum', cov_kwds={'time': ranks, 'maxlags': 20})
    return np.round(model.rsquared, 4), np.round(model.params.iloc[0], 2), np.round(model.tvalues.iloc[0], 2)


def run_master():
    df = load_data()

    # raw_target = vlm/vlm_pred_naive - 1
    # target = log(max(raw_target, -0.9))
    raw_target, features, vlm_pred_naive = generate_features(df)
    target = np.log1p(raw_target.clip(lower=-0.9))
    features = features.clip(lower=-3, upper=3)  # simple outlier removal

    # Time series CV sampling
    # the last 2 slcies are reserved for testing (2018-2019); hyper-parameter tuning uses in-sample data (16 slices)
    dates = features.index.get_level_values(0)
    cv_slices = []
    for year in range(2002, 2020):
        validate = str(year)
        train_start = str(max(2000, year - 10))  # max out at 10y, mainly to make the fitting code run faster
        train_end = str(year - 1)
        train_slicer = dates.slice_indexer(start=train_start, end=train_end)
        validate_slicer = dates.slice_indexer(start=validate, end=validate)
        cv_slices.append((train_slicer, validate_slicer))

    # parameter grid
    params = {
        'hl_days': [30, 180, 365, 365 * 3, 365 * 5],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001]
    }
    scores = dict()
    counter = 0
    total = int(np.product([len(v) for v in params.values()]))
    for hl_days in params['hl_days']:
        for alpha in params['alpha']:
            counter += 1
            print('{}: testing param set {}/{}...'.format(pd.Timestamp.now(), counter, total))
            prediction = predict(features, target, raw_target, cv_slices[:-2],
                                 hl_days=hl_days, alpha=alpha)
            scores[hl_days, alpha] = score(prediction)
    print('===hyperparam scores===')
    print('- Format: (R2, beta, t) -')
    print('- based on in-sample data only: 2000-2017')
    print(pd.Series(scores))

    # test
    print('===detailed fitting stats for chosen params====')
    hl_days = 365 * 3
    alpha = 0.001
    prediction = predict(features, target, raw_target, cv_slices,
                         hl_days=hl_days, alpha=alpha)
    print('- in-sample fit 2000-2017 -')
    print(score(prediction.loc[pd.IndexSlice[:'2017', :], :]))
    print('- out-of-sample fit 2018-2019 -')
    print(score(prediction.loc[pd.IndexSlice['2018':, :], :]))
    print('- fit by year -')
    for year in range(2002, 2020):
        print('{}: {}'.format('Y' + str(year), score(prediction.loc[pd.IndexSlice[str(year), :], :])))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_master()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
