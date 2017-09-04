# -*- coding:utf-8 -*-
# Author:jqzhang
# Editdate:2015-12-21
from WindPy import w
import pymysql
from datetime import datetime
import numpy as np
import pandas as pd


dt = datetime.now()
w.start()


if dt.hour < 16:
    date_before = w.tdaysoffset(-1, dt).Data[0][0]
    date_before =  ''.join(str(date_before).split(' ')[0].split('-'))
    dt = date_before

# 命令如何写可以用命令生成器来辅助完成
# 定义打印输出函数，用来展示数据使用
# 连接数据库
conn = pymysql.connect(host='127.0.0.1',
                       port=3306,
                       user='root',
                       password='lyz940513',
                       db='mysql',
                       charset='utf8mb4',
                       cursorclass=pymysql.cursors.DictCursor)
cursor = conn.cursor()

sql = "INSERT INTO stockprice VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"



cursor.execute('select max(tradedate) from stockprice;')
last_date = cursor.fetchall()[0].values()[0]

# 通过wset来取数据集数据


tradedate = w.tdays(last_date, dt).Data[0]
tradedate =  map(lambda x:''.join(str(x).split(' ')[0].split('-')), tradedate)

for date in tradedate[1:]:
    try:
        print "update the data of %s" % date
        wsetdata = w.wset(
            'SectorConstituent',
            'date=%s;sectorId=a001010100000000;field=wind_code' % date).Data[0]

        wssdata = w.wss(wsetdata, "open,high,low,close,vwap,volume,amt,free_float_shares,total_shares","tradeDate=%s;priceAdj=F;cycle=D;unit=1" % date)


        data = np.array(wssdata.Data).T
        print wssdata

        for i in range(data.shape[0]):
            sqllist = []
            sqllist.append(str(wsetdata[i]))
            sqllist.append(str(date))
            for d in data[i]:
                sqllist.append(float(d))

            sqllist = pd.Series(sqllist).fillna(method='pad').tolist()
            sqltuple = tuple(sqllist)
            cursor.execute(sql, sqltuple)
    except Exception as e:
        print e
    conn.commit()
conn.close()
