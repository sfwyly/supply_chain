
import os
import pandas as pd

# 1. 读取数据
demand_train_A = 'E:/competion/gongyinglian/test/demand_train_A.csv'
geo_topo = 'E:/competion/gongyinglian/geo_topo.csv'
inventory_info_A = 'E:/competion/gongyinglian/test/inventory_info_A.csv'
product_topo = 'E:/competion/gongyinglian/product_topo.csv'
weight_A = 'E:/competion/gongyinglian/test/weight_A.csv'

demand_train_A = pd.read_csv(demand_train_A)
geo_topo = pd.read_csv(geo_topo)
inventory_info_A = pd.read_csv(inventory_info_A)
product_topo = pd.read_csv(product_topo)
weight_A = pd.read_csv(weight_A)

demand_test_A = 'data/demand_test_A.csv'
demand_test_A = pd.read_csv(demand_test_A)

dfs = [demand_train_A,geo_topo,inventory_info_A,product_topo,weight_A,demand_test_A]

for df in dfs:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0',inplace=True)
    if 'ts' in df.columns:
        df = df.sort_values(by='ts')

# 2. 数据合并
all_data =pd.concat([demand_train_A,demand_test_A])
all_data = all_data.sort_values(by='ts')
all_data = all_data.reset_index().drop(columns='index')

# 3. 预测未来需求量
#真实值串14天
submission = demand_test_A
submission['yesterday_qty'] = submission.groupby('unit')['qty'].shift(1).fillna(method='ffill').reset_index().sort_index().set_index('index')

submission['diff_1'] = submission['qty'] - submission['yesterday_qty']
submission['qty'] = submission['diff_1']

submission['shift_14']=submission.groupby('unit')['qty'].shift(-14).fillna(0).reset_index().sort_index().set_index('index')
submission = submission[['unit','ts','shift_14']].rename(columns={'shift_14':'qty'})

#按照7天聚合
submission['dt'] = pd.to_datetime(submission['ts'])
submission['weekofyear'] = submission['dt'].dt.weekofyear
submission['year'] = submission['dt'].dt.year
submission_week = submission.copy()
submission_week = submission_week.groupby(['weekofyear','year','unit'],as_index=False).sum()
submission_week['sum_qty'] = submission_week['qty']
submission = pd.merge(submission_week,submission,on = ['weekofyear','year','unit'])
submission['dayofweek'] = submission['dt'].dt.dayofweek
submission = submission[submission['dayofweek']==0]
submission = submission[['unit','ts','sum_qty']].rename(columns={'sum_qty':'qty'})

# 4. 根据未来需求量消耗掉库存
init_inventory = inventory_info_A.set_index(['unit'])['qty'].to_dict()

def consume_init_inventory(arr, init_val):
    remain = init_val
    i = 0
    while remain > 0 and i < len(arr):
        arr[i] = max(0, arr[i] - remain)
        remain -= arr[i]
        i += 1
    return arr


r = []
for i, group in submission.groupby('unit'):
    unit = group['unit'].values[0]
    init_val = init_inventory[unit]

    group = group.sort_values(by='ts')
    qty_list = group['qty'].values
    qty_list = consume_init_inventory(qty_list, init_val)
    group['qty'] = qty_list
    r.append(group)

submission = pd.concat(r)