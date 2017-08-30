# -*- coding: utf-8 -*-
TS_PATH = "C://program files//Tinysoft//Analyse.NET//"
import numpy as np
import pandas as pd
import sys
sys.path.append(TS_PATH)
import TSLPy2
from dateutil.parser import parse
import talib
import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf8')

def tostry(data):
	ret = ""

	if isinstance(data,(int,float)):
		ret = "{0}".format(data)
	elif isinstance(data,(bytes)):
		ret = "{0}".format(data.decode("gbk"))
	elif isinstance(data,(str)):
		ret = "\"{0}\"".format(data)
	elif isinstance(data,(list)):
		lendata = len(data)
		ret+="["
		for i in range(lendata):
			ret+=tostry(data[i])
			if i<(lendata-1):
				ret+=","
		ret+="]"
	elif isinstance(data,(tuple)):
		lendata=len(data)
		ret+="("
		for i in range(lendata):
			ret+=tostry(data[i])
			if i<(lendata-1):
				ret+=","
		ret+=")"
	elif isinstance(data,(dict)):
		it = 0
		lendata = len(data)
		ret+="{"
		for i in data:
			ret+=tostry(i)+":"+tostry(data[i])
			it+=1
			if it<lendata:
				ret+=","
		ret+="}"
	else:
		ret = "{0}".format(data)
	return ret

def TStoPY(data):
	#import pdb;pdb.set_trace()
	ret = None
	if isinstance(data,(int,float)):
		ret = data
	elif isinstance(data,(bytes)):
		ret = "{0}".format(data.decode("gbk"))
	elif isinstance(data,(str)):
		ret = "\"{0}\"".format(data)
	elif isinstance(data,(list)):
		lendata = len(data)
		ret = ['']*lendata
		for i, item in enumerate(data):
			ret[i] = TStoPY(item)
	elif isinstance(data,(tuple)):
		lendata=len(data)
		ret = ('',)*lendata
		for i, item in enumerate(data):
			ret[i] = TStoPY(item)
	elif isinstance(data,(dict)):
		ret = {}
		for i, item in enumerate(data):
			#import pdb;pdb.set_trace()
			ret[TStoPY(item)] = TStoPY(data[item])
	else:
		ret = "{0}".format(data)
	return ret

def TSResult(data):
	'''
	data: data returned from TS
	'''
	if data[0] == 0:
		print('TS results achieved')
		return TStoPY(data[1])
	else:
		print("TS failed because {0}".format(data[2].decode('gbk')))
		return None

def get_ts_td(BegT, EndT, freq):
	'''
	get tradingday from tinysoft
	BegT: int 20000101
	EndT: int 20151231
	freq: cycle "月线",1分钟线，1秒线
	'''
	data = TSLPy2.RemoteCallFunc("get_td", [BegT, EndT, freq],{})
	return TSResult(data)

y1 = get_ts_td(20081220,20170520,"月线")
td = [parse(str(x)).strftime('%Y-%m-%d') for x in y1]

#import pdb;pdb.set_trace()

def get_ts_stks(index,date):
	'''
	runstr = 	"""
				return get_stks({0}, {1});
				""".format(index,date)
	data = TSLPy3.RemoteExecute(runstr,{})
	'''
	data = TSLPy2.RemoteCallFunc("get_stks", [index,date],{})
	return TSResult(data)
'''
a1 = get_ts_stks('深证A股;上证A股;中小企业板;创业板',20070110)
a2 = get_ts_stks('SH000300',20070110)

import pdb;pdb.set_trace()

_index = "深证A股;上证A股;中小企业板;创业板"
#y2 = get_ts_stks(_index,20161230)
'''
def get_ts_industry(index,date):
	'''
	runstr = 	"""
				return get_stks({0}, {1});
				""".format(index,date)
	data = TSLPy3.RemoteExecute(runstr,{})
	'''
	data = TSLPy2.RemoteCallFunc("get_industry", [index,date],{})
	#import pdb;pdb.set_trace()
	return TSResult(data)

#y3 = get_ts_industry(_index,20161230)

def get_ts_close(stk_code,begt,endt,fq):
	data = TSLPy2.RemoteCallFunc("get_close", [stk_code,begt,endt,fq],{})
	#import pdb;pdb.set_trace()
	c = pd.DataFrame(TSResult(data))
	#print(c)
	return c
'''
y4 = get_ts_close('SH000906',20100101,20170630,'日线')
y5 = get_ts_close('SH000300',20100101,20170630,'日线')
y4 = y4.set_index('date')
y5 = y5.set_index('date')
diff = y4['SH000906'] - y5['SH000300']
import pdb;pdb.set_trace()
'''
def get_close_all(index,begt,endt,cycle):
	#tradingday = get_ts_td(begt, endt, cycle)
	#import pdb;pdb.set_trace()
	#beg = tradingday[0]
	#end = tradingday[-1]
	data = TSLPy2.RemoteCallFunc("get_prices",
								[index,begt,endt,cycle],
								{})
	prices = pd.DataFrame(TSResult(data))
	#import pdb;pdb.set_trace()
	#prices = prices.fillna(0)
	return prices




def get_high_all(index,begt,endt,cycle):
	#tradingday = get_ts_td(begt, endt, cycle)
	#import pdb;pdb.set_trace()
	#beg = tradingday[0]
	#end = tradingday[-1]
	data = TSLPy2.RemoteCallFunc("get_highs",
								[index,begt,endt,cycle],
								{})
	prices = pd.DataFrame(TSResult(data))
	#import pdb;pdb.set_trace()
	prices = prices.fillna(0)
	return prices

def get_low_all(index,begt,endt,cycle):
	#tradingday = get_ts_td(begt, endt, cycle)
	#import pdb;pdb.set_trace()
	#beg = tradingday[0]
	#end = tradingday[-1]
	data = TSLPy2.RemoteCallFunc("get_lows",
								[index,begt,endt,cycle],
								{})
	prices = pd.DataFrame(TSResult(data))
	#import pdb;pdb.set_trace()
	prices = prices.fillna(0)
	return prices

#y5 = get_close_all('SH000906',20050101,20170101,'月线')
#import pdb;pdb.set_trace()

def get_financials(index,endt):
	data = TSLPy3.RemoteCallFunc("get_financials",
								[index,endt],
								{})
	fin = pd.DataFrame(TSResult(data))
	return fin

#y6 = get_financials('SH000300',20070110)
#import pdb;pdb.set_trace()


def get_indexweight(index,endt):
	data = TSLPy2.RemoteCallFunc("cxGetIndexWeight",
								[index,endt],
								{})
	#import pdb;pdb.set_trace()
	r = pd.DataFrame(TSResult(data))
	return r
#y7 = get_indexweight('SH000300','2010-12-01')
#print y7
#import pdb;pdb.set_trace()

def get_intraday_prices(index,begt,endt,cycle):
	data = TSLPy2.RemoteCallFunc("get_intraday_prices",
								[index,begt,endt,cycle],
								{})
	#import pdb;pdb.set_trace()
	r = pd.DataFrame(TSResult(data))
	return r

def get_barra_factor(start_date, end_date):
	data = TSLPy2.RemoteCallFunc("barra_factor",[start_date, end_date],{})
	r = pd.DataFrame(TSResult(data))
	return r



#y8 = get_intraday_prices('沪深300',20161101,20161130,'30分钟线')
#import pdb;pdb.set_trace()

"""
if __name__=='__main__':
	px =
	import pdb;pdb.set_trace()
"""
