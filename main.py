
import numpy as np
import pandas as pd

# date_list = pd.date_range(start = pd.to_datetime("20210302"), end = pd.to_datetime("20210305"))
#
# print(pd.to_datetime("20210302") - pd.to_datetime("20210301"))
length = 4
dic = {"*":2, "+":1,"-":1}
def judge(num, osp):
    a = [int(num[0])]
    b = []
    i = 1
    j = 0
    res = 0
    while i < length:

        while b and dic[b[-1]] >= dic[osp[j]]:
            oper = b.pop(-1)
            aai = a.pop(-1)
            ai = a.pop(-1)
            if oper == "+":
                aai = aai + ai
            elif oper == "-":
                aai = ai - aai
            elif oper == "*":
                aai = aai * ai
            a.append(aai)
        a.append(int(num[i]))
        b.append(osp[j])

        i += 1
        j += 1
    while b:
        oper = b.pop(-1)
        aai = a.pop(-1)
        ai = a.pop(-1)
        if oper == "+":
            aai = aai + ai
        elif oper == "-":
            aai = ai - aai
        elif oper == "*":
            aai = aai * ai
        a.append(aai)
    return a[0]

print(judge("1234", "+*-"))

