import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
import cv2 as cv
import matplotlib.cm as cm
from ic_group import plot_df
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

def plot_history_corr(data,factors,targets,time_=22*350,figsize=(15,9),locator='Month'):
    plt.figure(figsize=figsize)
    x = [datetime.strptime(h, '%Y-%m-%d').date() for h in data['day']]
    if locator=='Month':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    elif locator=='Year':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    elif locator=='Day':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%D'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    for factor in factors:
        for target in targets:
            y = data[factor].rolling(time_).corr(data[target])
            plt.plot(x,y,label=target)
            plt.legend()
    plt.title('%s corr' %(factor))
    plt.show()

def plot_gaus(x, y, label='', color='b',ax=None):
    use_index = np.array([len(y[i]) > 1 for i in range(len(y))])
    x = x[:-1]
    x = x[use_index]
    mean = np.array([np.mean(y[i]) for i in np.arange(use_index.shape[0])[use_index]])
    std = np.array([np.std(y[i]) for i in np.arange(use_index.shape[0])[use_index]])/5

    if type(ax)==type(None):
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.3)
        plt.plot(x, mean, label=label)
    else:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.3)
        ax.plot(x, mean, label=label)


        
def look(data, x, y, thresh=0.001, div_num=20,charge=1e-3,x_rank=False):
    '''
    单因子有效性分析
        data: DataFrame格式数据
        x：因子名称
        y:return的名称
        data[y] 就是因子
        data[x] 就是return
    '''

    X = data[x]
    if x_rank:
        X = X.rank()
    Y = data[y]
    X_min, X_max = X.quantile(thresh), X.quantile(1 - thresh)
    # 图1：元素分布直方图
    X_copy = np.array(X)
    X_copy[X_copy < X_min] = X_min
    X_copy[X_copy > X_max] = X_max

    plt.figure(figsize=(15, 12))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes[0, 0].hist(X_copy, bins=50, density=True)
    axes[0, 0].set_xlabel(x)
    axes[0, 0].set_title('%s 分布直方图' % x)

    ranks = np.arange(X_min, X_max, (X_max - X_min) / div_num)

    # 图2：相关性可视化
    Ys = []
    for i in range(1, ranks.shape[0]):
        indice = ((X >= ranks[i - 1]) * (X < ranks[i])).astype(bool)
        Ys.append(Y[indice])

    plot_gaus(ranks, Ys, ax=axes[1, 0])
    axes[1, 0].axhline(charge,color='r', linestyle='--')
    axes[1, 0].axhline(-charge,color='r', linestyle='--')
    
    axes[1, 0].set_xlabel(x)
    axes[1, 0].set_title('%s-%s 分布' % (y, x))

    # 图3：相关性积分
    df = pd.DataFrame({'X': X_copy, 'Y': Y})
    df = df.sort_values('X', ascending=True)
    df['Y_sum'] = df['Y'].cumsum()
    axes[0, 1].plot(df['X'], df['Y_sum'])
    axes[0, 1].set_xlabel(x)
    axes[0, 1].set_title('y-x积分曲线')

    # 图4：IR
    axes[1, 1].plot(ranks[:-1], [np.mean(y) / np.std(y) for y in Ys])
    axes[1, 1].set_xlabel(x)
    axes[1, 1].set_title('%s因子IR曲线' % x)
    plt.show()

def plot_contour(data, factors, target, line_num=5, thresh=1e-3, div_num=20, plot_hist=True, plot_2d=False,
                 blur=3, target_range=(None, None)):
    '''
    复合因子有效性分析
    data: DataFrame格式数据
    factors：需要分析的二个元素名称列表
    target:目标元素名称
    line_num:等高线数目
    thresh:剔除部分边缘数据，避免影响数据切分
    div_num：切分数目
    plot_hist:画出2个因子的分布直方图
    target_range=(max,min),对每个方块的return大小进行限制，避免噪声影响画图

    例子：
        keys= ['close','premium_last']
        target = 'profit_4d_am_am'
        target_range=(1e-2,-1e-2)
        plot_contour(dailys,keys,target,thresh=1e-2,div_num=20,blur=5,line_num=15,plot_2d=True,plot_hist=True,target_range=target_range)
    '''

    xy = []
    for factor in factors:
        X_min, X_max = data[factor].quantile(thresh), data[factor].quantile(1 - thresh)

        X_copy = np.array(data[factor])
        X_copy[X_copy < X_min] = X_min
        X_copy[X_copy > X_max] = X_max
        # 画因子分布直方图
        if plot_hist:
            plt.hist(X_copy, bins=50, density=True)
            plt.title('%s histogram' % factor)
            plt.show()
        xy.append(np.arange(X_min, X_max, (X_max - X_min) / (div_num + 1)))

    X, Y = xy
    X = X[:div_num]
    Y = Y[:div_num]
    Z = np.zeros([div_num - 1, div_num - 1])
    Z_num = np.zeros([div_num - 1, div_num - 1])
    factor_x, factor_y = factors
    for i in range(1, X.shape[0]):
        for j in range(1, Y.shape[0]):
            indice = ((data[factor_x] >= X[i - 1]).astype(float) * (data[factor_x] < X[i]).astype(float) * (
                    data[factor_y] >= Y[j - 1]).astype(float) * (data[factor_y] < Y[j]).astype(float)).astype(bool)
            if np.sum(indice) > 0:
                Z[i - 1, j - 1] = np.mean(data[target][indice])
                Z_num[i - 1, j - 1] = np.sum(indice)

    # 取最大最小值
    z_max, z_min = target_range
    if type(z_max) != type(None):
        Z[Z > z_max] = z_max
    if type(z_min) != type(None):
        Z[Z < z_min] = z_min

    # 模糊
    if blur > 0:
        Z = cv.blur(Z, (blur, blur))

    X, Y = np.meshgrid(X[1:], Y[1:])
    Z = Z.transpose()  # 画图时X，Y坐标会不一样
    # 画因子return热力图
    extent = (np.amin(X), np.amax(X), np.amin(Y), np.amax(Y))
    plt.imshow(np.flip(Z, 0), cmap=cm.hot)
    plt.colorbar()
    plt.show()

    C = plt.contour(X, Y, Z, line_num, colors='black', linewidth=.5)
    # 填充等高线
    plt.contourf(X, Y, Z, line_num, alpha=1, cmap=plt.cm.hot)
    plt.clabel(C, inline=True, fontsize=20)
    plt.xlabel(factor_x)
    plt.ylabel(factor_y)
    plt.show()
    # 双因子分布热力图+等高线
    if plot_2d:
        Z_num = np.log(Z_num)
        Z_num = Z_num.transpose()  # 画图时X，Y坐标会不一样
        C = plt.contour(X, Y, Z_num, line_num, colors='black', linewidth=.5)
        # 填充等高线
        plt.contourf(X, Y, Z_num, line_num, alpha=1, cmap=plt.cm.hot)
        plt.clabel(C, inline=True, fontsize=20)
        plt.xlabel(factor_x)
        plt.ylabel(factor_y)
        plt.show()


def plot_npl(date, y, freq=1, name='净值图', locator='Month', figsize=(14, 8), area=False, ymin=0.8):
    plt.figure(figsize=figsize)
    x = [datetime.strptime(h, '%Y-%m-%d').date() for h in date]
    if locator == 'Month':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        if freq != 1:
            X_ticks = pd.date_range(x[0], x[-1], freq='%dM' % freq)
            plt.xticks(X_ticks)
    elif locator == 'Year':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    elif locator == 'Day':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        X_ticks = pd.date_range(x[0], x[-1], freq='%dD' % freq)
        plt.xticks(X_ticks)
    if type(y) == dict:
        for k in y.keys():
            t = np.array(y[k])
            if t[0] != 0:
                t /= t[0]
            if area:
                plt.fill_between(x, t, alpha=0.2)
                plt.plot(x, t, alpha=0.6, label=k)
            else:
                plt.plot(x, t, label=k, linewidth=1)

    if area:
        plt.axis(ymin=ymin)
    plt.legend()
    title = '%s ' % (name)
    plt.title(title)
    plt.grid()
    plt.show()



'''
分层回测
'''
def analyse_group_new(data,factor,y,is_alpha=False,num_group=5,figsize=(16,5)): 
    name = 'abosolute group test' if is_alpha == False else 'relative group test'
    # 数据准备
    data = data[['date',factor,y]].dropna()    
    labels=range(1,num_group+1)            
    def cut(x):
        ### 分组函数
        x[factor],cut_bin = pd.qcut(x[factor].rank(method='first'),q=num_group,labels=labels,retbins = True,duplicates='drop')   
        if is_alpha == True:
            x[y] = x[y] - x[y].mean() 
        return x
    # 分桶
    data  = data.groupby(['date']).apply(lambda x:cut(x))
    # 日收益
    dates = data['date'].sort_values().unique()
    result = data.groupby(['date',factor])[y].mean().reset_index().set_index(['date'])   
    df_r = pd.DataFrame(index=dates)
    for label in labels:
        df_r[label] = result[result[factor]==label][y].fillna(0)  
    df_pnl = df_r.cumsum()   
    # 画图
    plot_df(df_pnl,freq=2,name=name,locator='Month',figsize=figsize,area=False,ymin=0.8)   