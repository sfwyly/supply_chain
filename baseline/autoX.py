

data_name = 'tianchi_clp'
path = f'./{data_name}'

# 赛题数据demand_test_A中给了标签，我们需要将它删掉。同时我们顺便删掉无用的'Unnamed: 0'列

demand_train_A = pd.read_csv(f'{path}/demand_train_A.csv')
demand_test_A = pd.read_csv(f'{path}/demand_test_A.csv')

demand_train_A.drop('Unnamed: 0', axis=1, inplace=True)
demand_test_A.drop(['Unnamed: 0', 'qty'], axis=1, inplace=True)

# 将 demand_train_A, demand_test_A 保存为train.csv, test.csv
demand_train_A.to_csv(path + '/train.csv', index = False)
demand_test_A.to_csv(path + '/test.csv', index = False)

from autox import AutoX


# 数据集是多表数据集，需要配置表关系
relations = [
    {
            "related_to_main_table": "true", # 是否为和主表的关系
            "left_entity": "train.csv",  # 左表名字
            "left_on": ["product"],  # 左表拼表键
            "right_entity": "product_topo.csv",  # 右表名字
            "right_on": ["product_level_2"], # 右表拼表键
            "type": "1-1" # 左表与右表的连接关系
        },  # train.csv和product_topo.csv两张表是1对1的关系，拼接键为train.csv中的product列 和 product_topo.csv中的product_level_2列
    {
            "related_to_main_table": "true", # 是否为和主表的关系
            "left_entity": "test.csv",  # 左表名字
            "left_on": ["product"],  # 左表拼表键
            "right_entity": "product_topo.csv",  # 右表名字
            "right_on": ["product_level_2"], # 右表拼表键
            "type": "1-1" # 左表与右表的连接关系
        },  # test.csv和product_topo.csv两张表是1对1的关系，拼接键为test.csv中的product列 和 product_topo.csv中的product_level_2列
    {
            "related_to_main_table": "true", # 是否为和主表的关系
            "left_entity": "train.csv",  # 左表名字
            "left_on": ["geography"],  # 左表拼表键
            "right_entity": "geo_topo.csv",  # 右表名字
            "right_on": ["geography_level_3"], # 右表拼表键
            "type": "1-1" # 左表与右表的连接关系
        },  # train.csv和geo_topo.csv两张表是1对1的关系，拼接键为train.csv中的geography列 和 geo_topo.csv中的geography_level_3列
    {
            "related_to_main_table": "true", # 是否为和主表的关系
            "left_entity": "test.csv",  # 左表名字
            "left_on": ["geography"],  # 左表拼表键
            "right_entity": "geo_topo.csv",  # 右表名字
            "right_on": ["geography_level_3"], # 右表拼表键
            "type": "1-1" # 左表与右表的连接关系
        } # test.csv和geo_topo.csv两张表是1对1的关系，拼接键为test.csv中的geography列 和 geo_topo.csv中的geography_level_3列
]

autox = AutoX(target = 'qty', train_name = 'train.csv', test_name = 'test.csv',
               id = ['unit'], path = path, time_series=True, ts_unit='D',time_col = 'ts',
               relations = relations
              )  #feature_type = feature_type,
sub = autox.get_submit_ts()

# 查看预测结果
sub.head()

# 检查预测结果和真实结果的差距
sub.rename({'qty': 'qty_pre'}, axis=1, inplace=True)
demand_test_A = pd.read_csv(f'{path}/demand_test_A.csv', usecols = ['unit','ts','qty'])

analyze = demand_test_A.merge(sub, on = ['unit', 'ts'], how = 'left')

# 查看mae
from sklearn.metrics import mean_absolute_error
y_true = analyze['qty']
y_pred = analyze['qty_pre']
mean_absolute_error(y_true, y_pred)

