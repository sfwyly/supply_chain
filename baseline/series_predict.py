

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# error计算误差
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# data split分割数据，训练集、测试集、预测集
def timeseries_split(x, y, test_size, pred_size):
    index_test = int(len(x) * (1 - test_size))
    x_train = x.iloc[:index_test]
    y_train = y.iloc[:index_test]
    x_test = x.iloc[index_test:len(x) - pred_size]
    y_test = y.iloc[index_test:len(x) - pred_size]
    x_pred = x.iloc[-pred_size:]
    y_pred = y.iloc[-pred_size:]
    return x_train, y_train, x_test, y_test, x_pred, y_pred


# calculate mean计算均值特征
def cal_mean(data, x_feature, y_feature):
    return dict(data.groupby(x_feature)[y_feature].mean())#利用分组，计算均值特征


# make feature计算平移特征
def build_feature(data, lag_start, lag_end, test_size, target_encoding=False, num_day_pred=1):#target_encoding是否要开启均值特征，num_day_pred预测多少天

    start_dt = pd.to_datetime("20210302")
    end_dt = pd.to_datetime("20210607")
    date_list = pd.date_range(start=start_dt, end=end_dt)

    # build future data with 0
    last_date = data["time"].max()
    # 预测点个数，由数据粒度决定
    pred_points = int(num_day_pred * 24)  # 1h粒度，1day = 24个点
    pred_date = pd.date_range(start=last_date, periods=pred_points + 1, freq="1h")
    pred_date = pred_date[pred_date > last_date]  # 排除掉last_date(非预测)， preiods = pred_points +1,也是因为last_date为非预测point，所以后延1个point
    future_data = pd.DataFrame({"time": pred_date, "y": np.zeros(len(pred_date))})#先将预测时间段的value设置为0，然后在利用shift和均值等构件特征，做预测
    # concat future data and last data
    df = pd.concat([data, future_data])
    df.set_index("time", drop=True, inplace=True)
    #print(df)
    # make feature
    # shift feature平移特征，lag_start,lag_end分别为shift平移多少，如从80-120,80,81,82，，，119,120.
    for i in range(lag_start, lag_end):
        df["lag_{}".format(i)] = df.y.shift(i)
    #diff feature#差分特征，对平移后的lag做差分操作，此特征作用不大
    df["diff_lag_{}".format(lag_start)] = df["lag_{}".format(lag_start)].diff(1)
    # time feature时间特征
    df["hour"] = df.index.hour
    # df["day"] = df.index.day
    # df["month"] = df.index.month
    df["minute"] = df.index.minute
    df["weekday"] = df.index.weekday
    df["weekend"] = df.weekday.isin([5, 6]) * 1

    df["holiday"] = 0
    df.loc["2018-10-01 00:00:00":"2018-10-07 23:00:00","holiday"] = 1
    #print(df)
    # df["holiday"]
    # average feature
    if target_encoding:  # 用test
        # 计算到已有数据截止，然后在映射到预测的数据中，这样就训练、测试、预测都有此特征
        df["weekday_avg"] = list(map(cal_mean(df[:last_date], "weekday", "y").get, df.weekday))#时间均值特征
        df["hour_avg"] = list(map(cal_mean(df[:last_date], "hour", "y").get, df.hour))
        df["weekend_avg"] = list(map(cal_mean(df[:last_date], "weekend", "y").get, df.weekend))
        df["minute_avg"] = list(map(cal_mean(df[:last_date], "minute", "y").get, df.minute))
        df = df.drop(["hour","minute","weekday", "weekend"], axis = 1)
    #one-hot没有作用
    #df = pd.get_dummies(df, columns = ["hour", "minute", "weekday", "weekend"])
    # data split
    y = df.dropna().y
    x = df.dropna().drop("y", axis=1)
    x_train, y_train, x_test, y_test, x_pred, y_pred = \
        timeseries_split(x, y, test_size=test_size, pred_size=pred_points)
    return x_train, y_train, x_test, y_test, x_pred, y_pred

# predict
def predict_future(model, scaler, x_pred, y_pred, lag_start, lag_end):#model拟合的模型，scaler归一化，lag平移特征，x_pred/y_pred预测x和y
    y_pred[0:lag_start] = model.predict(scaler.transform(x_pred[0:lag_start]))  # 预测到lag_start上一行
    for i in range(lag_start, len(x_pred)):
        last_line = x_pred.iloc[i-1]  # 已经预测数据的最后一行,还没预测数据的上一行，即shift,上一行填充到斜角下一行
        index = x_pred.index[i]
        x_pred.at[index, "lag_{}".format(lag_start)] = y_pred[i-1]
        x_pred.at[index, "diff_lag_{}".format(lag_start)] = y_pred[i-1] -  x_pred.at[x_pred.index[i-1], "lag_{}".format(lag_start)]
        for j in range(lag_start + 1, lag_end):  # 根据平移变换shift，前一个lag_{}列的值，shift后为下一个列的值
            x_pred.at[index, "lag_{}".format(j)] = last_line["lag_{}".format(j-1)]
        # 已经预测的最后一个值，赋值给lag_start对应的"lag_{}.format(lag_start)列
        # x_pred.at[index, "lag_{}".format(lag_start)] = y_pred[i - 1]
        y_pred[i] = model.predict(scaler.transform([x_pred.iloc[i]]))[0]
    return y_pred

# plot显示结果
def plot_result(y, y_fit, y_future):#y真实，y_fit拟合，y_future预测
    assert len(y) == len(y_fit)
    plt.figure(figsize=(16, 8))
    # plt.plot(y.index, y, "k.", label="y_orig")
    plt.plot(y.index, y, label="y_orig")
    plt.plot(y.index, y_fit, label="y_fit")
    plt.plot(y_future.index, y_future, "y", label="y_predict")
    error = mean_absolute_error(y, y_fit)
    plt.title("mean_absolute_error{0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# coefs显示重要性
def plot_importance(model, x_train):
    coefs = pd.DataFrame(model.coef_, x_train.columns)
    #coefs = pd.DataFrame(model.feature_importances_, x_train.columns)
    coefs.columns = ["coefs"]
    coefs["coefs_abs"] = coefs.coefs.apply(np.abs)
    coefs = coefs.sort_values(by="coefs_abs", ascending=False).drop(["coefs_abs"], axis=1)
    plt.figure(figsize=(16, 6))
    coefs.coefs.plot(kind="bar")
    plt.grid(True, axis="y")
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles="dashed")
    plt.show()

# read data
if __name__ == "__main__":
    dataf = pd.read_csv("data.csv")
    dataf["time"] = pd.to_datetime(dataf["time"])
    dataf = dataf.sort_values("time")
    dataf.rename(columns={"sump": "y"}, inplace=True)
    lag_start = 80#要根据数据周期，调试
    lag_end = 120#平移特征
    x_train, y_train, x_test, y_test, x_pred, y_pred = build_feature \
        (dataf, lag_start=lag_start, lag_end=lag_end, test_size=0.3, target_encoding=True, num_day_pred=1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    tscv = TimeSeriesSplit(n_splits=5)
    #lr = LassoCV(cv=tscv)
    lr = LinearRegression(normalize= "l1")
    #可以尝试随机森林的效果，也不错。也可以做多模型结果融合，请自己尝试。
    #lr = RandomForestRegressor(n_estimators=100, max_depth=10) #lag_start = 288, lag_end = 320
    # lr = RidgeCV(cv = tscv)
    lr.fit(x_train_scaled, y_train)
    #train_score = lr.score(x_train_scaled, y_train)
    #test_score = lr.score(x_test_scaled, y_test)
    #print("num_tree", each, "score", train_score, test_score)
    # future预测
    y_future = predict_future(lr, scaler, x_pred, y_pred, lag_start, lag_end)
    #print(x_pred)
    # now 拟合
    y_fit = lr.predict(np.concatenate((x_train_scaled, x_test_scaled)))
    y = pd.concat([y_train, y_test])
    # 显示结果
    plt.figure(figsize=(16, 8))
    plt.plot(data["time"], data["sump"])
    plot_result(y, y_fit, y_future)
    y_future.to_csv("y_future_lr_test.csv")
    plot_importance(lr, x_train)
