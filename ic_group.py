import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.api import OLS, add_constant
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
def plot_df(df, freq=1,name='unknown',locator='Month',figsize=(16,6),area=False,ymin=0.8):
    # 输入为dataframe
    plt.figure(figsize=figsize)
    x = [datetime.strptime(h, '%Y-%m-%d').date() for h in df.index]
    if locator=='Month':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        X_ticks = pd.date_range(x[0],x[-1],freq ='%dM'%freq)
        plt.xticks(X_ticks)
    elif locator=='Year':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        X_ticks = pd.date_range(x[0],x[-1],freq ='%dY'%freq)
        plt.xticks(X_ticks)
    elif locator=='Day':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        X_ticks = pd.date_range(x[0],x[-1],freq ='%dD'%freq)
        plt.xticks(X_ticks)    
    for k in df.columns:
        t = np.array(df[k])        
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
def plot_twinx_ma(dates,x,x_ma, y, freq=1,name='unknown',locator='Month',figsize=(16,6),area=False,ymin=0.8):
    ### 双轴三图
    plt.figure(figsize=figsize)
    x_name = x.name
    y_name = y.name
    d = [datetime.strptime(h, '%Y-%m-%d').date() for h in dates]
    if locator=='Month':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        X_ticks = pd.date_range(d[0],d[-1],freq ='%dM'%freq)
        plt.xticks(X_ticks)
    elif locator=='Year':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        X_ticks = pd.date_range(d[0],d[-1],freq ='%dY'%freq)
        plt.xticks(X_ticks)
    elif locator=='Day':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        X_ticks = pd.date_range(d[0],d[-1],freq ='%dD'%freq)
        plt.xticks(X_ticks)
    if type(y) == pd.Series:         
        x = np.array(x)  
        y = np.array(y)
        plt.plot(d,x_ma,label='ma',color='dodgerblue',linewidth=1.5)
        plt.bar(d, x, label=x_name,color='dimgray',width=0.5)
        plt.twinx()        
        plt.plot(d, y, label=y_name, color='magenta',linewidth=1.5)
    plt.legend()
    title = '%s ' % (name)
    plt.title(title)
    plt.grid()
    plt.show()
def get_ic(data,x,y): 
    data = data[[x,y]].dropna()
    if len(data)==0:
        return None    
    data['x_rank'] = data[x].rank()
    data['y_rank'] = data[y].rank()    
    ic = data['x_rank'].corr(data[y])
    reg_fit = OLS(data['y_rank'],data['x_rank']).fit()  
    res = {}
    res['ic'] = ic
    res['t'] = reg_fit.tvalues[-1]
    return res
def analyse_ic(data,x,y,w=20):
    data = data.reset_index()[['date',x,y]]
    dates = data['date'].sort_values().unique()
    res_dict = {}
    for date in dates:
        d = data[data.date==date].dropna()
        res = get_ic(d,x,y)
        res_dict[date]=res
    df = pd.DataFrame(res_dict).T
    df = df.dropna()
    if len(df)==0:
        return None
    df['ic_cum'] = df['ic'].cumsum()
    df['ic_ma'] = df['ic'].rolling(min_periods=1, window=w, center=False).mean()
    df['ic_std'] = df['ic'].rolling(min_periods=1, window=w, center=False).std()
    df['ir_ma'] = df['ic_ma']/df['ic_std']
    ic_mean = df['ic'].mean()
    ic_std = df['ic'].std()
    ic_abs_mean = df['ic'].abs().mean()
    ic_positive_rate = len(df[df.ic > 0]) / len(df['ic'])
    ir = ic_mean / ic_std
    t_abs_mean = df['t'].abs().mean()  
    result = {}
    result['factor'] = x
    result['ic_mean'] = ic_mean
    result['ic_std'] = ic_std
    result['ic_abs_mean'] = ic_abs_mean
    result['ic_positive_rate'] = ic_positive_rate
    result['ir'] = ir
    result['t_abs_mean'] = t_abs_mean   
    dates = df.index
    x = df['ic']
    x_ma = df['ic_ma']
    y = df['ic_cum']
    plot_twinx_ma(dates , x, x_ma,y, freq=2,name='ic',locator='Month',figsize=(16,6),area=False,ymin=0.8)
    return result
#分组函数
def get_group (data,num_group=5,factor='momentum'):  
    ranks=data[factor].rank(ascending=True)  
    label=['g'+str(i) for i in range(1,num_group+1) ]  
    category=pd.cut(ranks,bins=num_group,labels=label,duplicates='drop') 
    category.name='group'
    new_data=data.join(category) 
    return new_data  
# 分层结果分析
def analyse_group(data,x,y,num_group=5): 
    #创建组号    
    labels=['g'+str(i) for i in range(1,num_group+1) ]  
    data = data.reset_index()[['date',x,y]].dropna()
    # 日期
    dates = data['date'].sort_values().unique()
    # 分组
    new_data = data.groupby(['date']).apply(lambda df:get_group(df,num_group=num_group,factor=x))
    # 日收益
    result = new_data.groupby(['date','group'])[y].mean().reset_index().set_index(['date'])
    df_r = pd.DataFrame(index=dates)
    for label in labels:
        df_r[label] = result[result.group==label][y].fillna(0)  
    df_pnl = df_r.cumsum()   
    # 画图
    plot_df(df_pnl,freq=2,name='pnl',locator='Month',figsize=(16,6),area=False,ymin=0.8)