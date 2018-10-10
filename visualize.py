import pandas as pd
import matplotlib
import time
from pandas import Series
import matplotlib
from matplotlib import pyplot as plt
import csv
import os
import random
import numpy as np


def get_date(epoch):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))


def get_date_list(epoch_list):
    li=list()
    for epoch in epoch_list:
        li.append(get_date(int(epoch)))
    return li


def get_hour(epoch_list):
    li=list()
    for epoch in epoch_list:
        hour = time.strftime('%H', time.localtime(epoch))
        hour = int(hour)
        li.append(hour)
    return li


def get_hour_list(epoch_list):
    li=list()
    for ts_list in epoch_list:
        for epoch in ts_list:
            hour=time.strftime('%H', time.localtime(epoch))
            hour=int(hour)
            li.append(hour)
    return li


def read_hist(file):
    filepath = os.getcwd() + '/datasets/' + file
    with open(filepath, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        data = random.choice(list(data))
        print(data)
        cong = list(data[3].replace('[', '').replace(']', '').split(','))
        print(cong)
        # cong = [int(i) for i in cong]
        cong = list(map(int, cong))
        t = list(data[2].replace('[', '').replace(']', '').split(','))
        extended_t=list()
        for i in range(len(t)):
            congestion = cong[i]
            ts = t[i]
            cur = int(ts)
            while(cur <= int(ts) + int(congestion)):
                extended_t.append(cur)
                cur = cur + 3600
        return extended_t


def read_hist_list(data):
    #filepath = os.getcwd() + '/datasets/' + file
    #with open(filepath, 'r') as f:
    cong = list(data[3].replace('[', '').replace(']', '').split(','))
    print(cong)
    # cong = [int(i) for i in cong]
    cong = list(map(int, cong))
    t = list(data[2].replace('[', '').replace(']', '').split(','))
    extended_t=list()
    for i in range(len(t)):
        congestion = cong[i]
        ts = t[i]
        cur = int(ts)
        while(cur <= int(ts) + int(congestion)):
            extended_t.append(cur)
            cur = cur + 3600
    return extended_t


def read(file):
    filepath= os.getcwd() + '/datasets/' + file
    with open(filepath, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        data = random.choice(list(data))
        print(data)
        cong = list(data[3].replace('[', '').replace(']', '').split(','))
        print(cong)
        #cong = [int(i) for i in cong]
        cong = list(map(int,cong))
        t = list(data[2].replace('[', '').replace(']', '').split(','))
        series = pd.DataFrame(
            {
                'Date': t,
                'Congestion': cong
            }
        )
        return series


def read_hist_all(file):
    filepath = os.getcwd() + '/datasets/' + file
    extended_t=list()
    with open(filepath, 'r') as f:
        data = csv.reader(f, delimiter='\t')
        for row in data:
            extended_list = read_hist_list(list(row))
            extended_t.append(extended_list)
    return extended_t


if __name__ == '__main__':

    #series = read('chn-congestions.csv')
    #series.set_index('Date',inplace=True)
    # print(series.shape)
    # print(series)
    # ts = list()
    # cg = list()
    # for i in range(len(series)):
    #     end_ts = int(series['Date'][i]) + int(series['Congestion'][i])
    #     ts.append(int(series['Date'][i]))
    #     ts.append(int(series['Date'][i]))
    #     cg.append(0)
    #     cg.append(1)
    #     ts.append(int(end_ts))
    #     ts.append(int(end_ts))
    #     cg.append(1)
    #     cg.append(0)
    #
    # ts = ts[:160]
    # cg = cg[:160]
    # # num_steps = 10
    # # step = int((max(ts)-min(ts))/num_steps)
    # step = 24*60*60
    # xt = list(range(min(ts), max(ts) + 1, step))
    # xt_label = get_date_list(xt)
    # print(xt_label)
    # # Change font size
    # matplotlib.rcParams.update({'font.size': 5})
    #
    # plt.xticks(xt, xt_label, rotation=90)
    # plt.plot(ts, cg, linewidth= 0.5)
    # #plt.show(bbox_inches='tight')
    # filepath=os.getcwd()+'/experiments/images/groundTruth10.pdf'
    # plt.savefig(filepath,  bbox_inches='tight')

    hist_series = read_hist_all('pune-congestions.csv')
    hist_series = get_hour_list(hist_series)
    print(hist_series[:30])
    plt.xticks([i for i in range(24)])
    plt.hist(hist_series, bins=24)
    plt.show()
