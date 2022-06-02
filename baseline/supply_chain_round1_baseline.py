import pandas as pd
import numpy as np
import time


class ReplenishUnit:
    def __init__(self,
                 unit,
                 demand_hist,
                 intransit,
                 qty_replenish,
                 qty_inventory_today,
                 qty_using_today,
                 arrival_sum,
                 lead_time
                 ):
        '''
        记录各补货单元状态 一定要注意都是当前单元的3.1号库存与需求量进行预测 3，2号-6.7号的库存数据
        :param unit:
        :param demand_hist: 净需求历史 对与指定单元unit 小于等于3.1 号的数据
        :param intransit: 补货在途   从 3.2 到 6.7 号的数据  初始化0
        :param qty_replenish: 补货记录  从 3.2 到 6.7 号的数据  初始化0
        :param qty_inventory_today: 当前可用库存 初始为 3.1 号 inventory表数据
        :param qty_using_today: 当前已用库存（使用量） 初始为 训练表内的 3.1 号 qty数据
        :param arrival_sum: 补货累计到达  初始为0
        :param lead_time: 补货时长，交货时间  i 天补货 replenish  i + lead_time 天后到达
        '''
        self.unit = unit
        self.demand_hist = demand_hist
        self.demand = demand_hist
        self.intransit = intransit
        self.qty_replenish = qty_replenish
        self.qty_inventory_today = qty_inventory_today
        self.qty_using_today = qty_using_today
        self.arrival_sum = arrival_sum
        self.init_val = qty_inventory_today
        self.lead_time = lead_time

    def update(self,
               date,
               arrival_today,
               demand_today):
        '''
        每日根据当天补货到达与当日净需求更新状态
        :param date:
        :param arrival_today: 当天补货到达
        :param demand_today: 当天净需求
        :return:
        '''
        self.init_val += arrival_today
        self.qty_inventory_today += arrival_today  # 可用库存 + 当天到达的供应
        self.arrival_sum += arrival_today  # 对应于某一单元 这些天到达的 供应总和
        inv_today = self.qty_inventory_today  #
        if demand_today < 0:  # 当天库存释放了
            self.qty_inventory_today = self.qty_inventory_today + min(-demand_today, self.qty_using_today)  # 只能从已经使用了的中进行释放 所以最小值
        else:
            self.qty_inventory_today = max(self.qty_inventory_today - demand_today, 0.0)
        self.qty_using_today = max(self.qty_using_today + min(demand_today, inv_today), 0.0)
        # self.demand_hist = self.demand_hist.append({"ts": date, "unit": self.unit, "qty": demand_today}, ignore_index = True)  # 是对于每个3.2 -> 6.7都是单独预测的， 没有联系之前的数据
        self.demand_hist = self.demand_hist[self.demand_hist["ts"] <= date]  # 当前天数之前的

    def forecast_function(self,
                          demand_hist):
        demand_average = np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:])  # 根据 需求给策略 确保每天补充的能够尽可能的最大化满足当前需求 -3 * lead_time 效果更好
        return [demand_average] * 90

    def getFutureResult(self, date): # 这里不一定满lead_time的补充和，需要进行动态根据扩充
        demand = self.demand[self.demand["ts"] >= date]
        demand = demand[demand["ts"] < date + date.freq * self.lead_time]["qty"]
        res = 0.
        if len(demand) < self.lead_time:
            res = sum(demand) + np.mean(demand) * (self.lead_time - len(demand))
        else:
            res = sum(demand)
        return res

    def getTodaySupply(self, date, demand, intransit, safety_stock):
        """
        根据补货到达的数目， 动态计算需要补货的数量
        :param date: 当天日期
        :param demand: lead_time 共lead_time的需求
        :param intransit: 补货的时机
        :return: 当天补货量
        """
        # dl = len(pd.date_range(pd.to_datetime("20210302"), date))
        # qty_intransit = 0.
        l = 0
        reorder_point = 0.
        for td, ti in zip(demand, intransit): # 当天update执行已经更新了补货到途
            # 当天需求 td 当天补货 ti
            if ti != 0 and l!=0:
                break
            reorder_point += td
            l += 1

        return reorder_point + safety_stock * l

    def replenish_function(self,
                           date, demand_today):
        '''
        根据当前状态判断需要多少的补货量
        补货的策略由选手决定，这里只给一个思路
        :param date:
        :return:
        '''
        replenish = 0.0
        if date.dayofweek != 0:
            # 周一为补货决策日，非周一不做决策
            pass
        else:
            # 预测未来需求量
            qty_demand_forecast = self.forecast_function(demand_hist = self.demand_hist)

            # 计算在途的补货量
            qty_intransit = sum(self.intransit[:date+self.lead_time * date.freq]) - self.arrival_sum

            # 安全库存 用来抵御需求的波动性 选手可以换成自己的策略
            safety_stock = (max(self.demand_hist["qty"].values[-3 * self.lead_time:]) - (np.mean(self.demand_hist["qty"].values[-3 * self.lead_time:]))) * self.lead_time

            # 再补货点，用来判断是否需要补货 选手可以换成自己的策略
            reorder_point = sum(qty_demand_forecast[:self.lead_time]) + safety_stock

            # reorder_point = self.getFutureResult(date)
            # reorder_point = self.getTodaySupply(date, qty_demand_forecast[:self.lead_time], self.intransit[date:date+self.lead_time * date.freq], safety_stock)

            # replenish = max(0, demand_today - self.init_val)
            # self.init_val = max(0, self.init_val - demand_today)

            # 判断是否需要补货并计算补货量，选手可以换成自己的策略，可以参考赛题给的相关链接
            if self.qty_inventory_today + qty_intransit < reorder_point:  # qty_inventory_today + qty_intransit
                replenish = reorder_point - (self.qty_inventory_today + qty_intransit)

            self.qty_replenish.at[date] = replenish
            self.intransit.at[date + self.lead_time * date.freq] = replenish


def consume_init_inventory(arr, init_val):
    remain = init_val
    i = 0
    while remain > 0 and i < len(arr):
        arr[i] = max(0, arr[i] - remain)
        remain -= arr[i]
        i += 1
    return arr


class SupplyChainRound1Baseline:
    def __init__(self, prefix="E:/competition/gongyinglian/"):
        """
            unit: 单元 加密字符
            ts: 日期
            qty: 资源使用量
            geography: 地理位置
            geography_level: 地理聚合维度
            product: 产品信息
            product_level: 产品聚合维度
        """
        self.using_hist = pd.read_csv(prefix + "demand_train_A.csv")
        self.using_future = pd.read_csv(prefix + "demand_test_A.csv")
        self.inventory = pd.read_csv(prefix + "inventory_info_A.csv")
        self.last_dt = pd.to_datetime("20210301")
        self.start_dt = pd.to_datetime("20210302")
        self.end_dt = pd.to_datetime("20210607")
        self.lead_time = 14
        self.prefix = prefix

    def run(self):
        self.using_hist["ts"] = self.using_hist["ts"].apply(lambda x:pd.to_datetime(x))
        self.using_future["ts"] = self.using_future["ts"].apply(lambda x:pd.to_datetime(x))
        qty_using = pd.concat([self.using_hist, self.using_future])
        date_list = pd.date_range(start = self.start_dt, end = self.end_dt)
        unit_list = self.using_future["unit"].unique()
        res = pd.DataFrame(columns = ["unit", "ts", "qty"])

        replenishUnit_dict = {}
        demand_dict = {}

        #初始化，记录各补货单元在评估开始前的状态
        for chunk in qty_using.groupby("unit"):
            unit = chunk[0]  # 产品
            demand = chunk[1]  # 针对这个产品的信息
            # print(demand.head(10))
            demand.sort_values("ts", inplace = True, ascending = True)

            #计算净需求量
            demand["diff"] = demand["qty"].diff().values
            demand["qty"] = demand["diff"]  # 资源使用量相对第一天的偏移， 并且剔除了第一天

            del demand["diff"]
            demand = demand[1:]
            replenishUnit_dict[unit] = ReplenishUnit(unit = unit,
                                                     demand_hist = demand,  # < 3.2 demand[demand["ts"] < self.start_dt]
                                                     intransit = pd.Series(index = date_list.tolist(), data = [0.0] * (len(date_list))),  # 具体lead_time天后进行补货
                                                     qty_replenish = pd.Series(index = date_list.tolist(), data = [0.0] * (len(date_list))),  # 决策当天补货
                                                     qty_inventory_today = self.inventory[self.inventory["unit"] == unit]["qty"].values[0],  # 3.1 号库存数据  Ventory文件只保存3.1号的数据，相当于预测3.2-6.7号数据的初始库存
                                                     qty_using_today = self.using_hist[(self.using_hist["ts"] == self.last_dt) & (self.using_hist["unit"] == unit)]["qty"].values[0],  # 3.1 号的资源使用量
                                                     arrival_sum = 0.0,
                                                     lead_time = self.lead_time)  # 补货周期
            #记录评估周期内的净需求量  当前单元的 在3.1号及之后的数据 也就是训练集的3.1号数据 与测试集的数据
            # 这里他是将训练集与测试集联合起来进行同样的处理 通过时间的大小决定取得那些数据
            demand_dict[unit] = demand[(demand["unit"] == unit) & (demand["ts"] >= self.start_dt)]  # 3.2

        for date in date_list:
            #按每日净需求与每日补货到达更新状态，并判断补货量
            for unit in unit_list:
                demand = demand_dict[unit]  # 当前单元的 3.2 -> 6.7 号的数据
                demand_today = demand[demand["ts"] == date]["qty"].values[0]  # 当天指定单元需求量
                arrival = replenishUnit_dict[unit].intransit.get(date, default = 0.0)  # 当天补货到达的量 初始化为全0
                replenishUnit_dict[unit].update(date = date,
                                                arrival_today = arrival,
                                                demand_today = demand_today)
                replenishUnit_dict[unit].replenish_function(date, demand_today)

        for unit in unit_list:
            res_unit = replenishUnit_dict[unit].qty_replenish
            res_unit = pd.DataFrame({"unit": unit,
                                     "ts": res_unit.index,
                                     "qty": res_unit.values})
            res_unit = res_unit[res_unit["ts"].apply(lambda x:x.dayofweek == 0)]
            res = pd.concat([res, res_unit])
        #输出结果
        res.to_csv(self.prefix+"result/baseline"+str(time.time())+".csv")


if __name__ == '__main__':
    supplyChainRound1Baseline = SupplyChainRound1Baseline()
    supplyChainRound1Baseline.run()
