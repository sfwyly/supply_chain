

# 获取最大子序列和
def getMSS(nums):
    length = len(nums)

    dp = [[0] * length for _ in range(length)]

    for i in range(length):

        for j in range(length):



    return 0


# 获取inv_rate 与 sla
def getNetrics(qty_inventory_list, qty_using_list):

    # 1. 计算sla

    inv_rate = 0.
    T = len(qty_inventory_list)
    for qty_inventory, qty_using in zip(qty_inventory_list, qty_using_list):
        inv_rate += qty_inventory / ((qty_inventory + qty_using) * T)

