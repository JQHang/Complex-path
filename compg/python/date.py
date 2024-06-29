import os
import json
import sys

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

"""
作用：
    根据给定的目标时间，将其切分为按月分隔的时间区间

输入：
    KG_time_range_list：给定的目标时间

返回值：
    按月分隔的时间区间
"""
def divid_range_list_to_monthly_list(KG_time_range_list):
    KG_time_monthly_list = []
    
    tmp_range_time = KG_time_range_list[0]
    tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
    tmp_range_time_add_month = datetime(tmp_range_time_add_month.year, tmp_range_time_add_month.month, 1)
    
    while tmp_range_time_add_month < KG_time_range_list[1]:

        KG_time_monthly_list.append([tmp_range_time.strftime("%Y-%m-%d"), tmp_range_time_add_month.strftime("%Y-%m-%d")])
        
        tmp_range_time = tmp_range_time_add_month
        tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
        
    KG_time_monthly_list.append([tmp_range_time.strftime("%Y-%m-%d"), KG_time_range_list[1].strftime("%Y-%m-%d")])
        
    return KG_time_monthly_list

"""
作用：
    根据给定的目标时间，将其切分为按月分隔的时间区间

输入：
    KG_time_range_list：给定的目标时间

返回值：
    按月分隔的时间区间
"""
def divid_range_list_to_monthly_first_day_list(KG_time_range_list):
    KG_time_monthly_list = []
    
    tmp_range_time = KG_time_range_list[0]
    
#     tmp_Label_Time_monthly_dt = datetime.strptime(tmp_Label_Time_monthly, "%Y-%m-%d")
    KG_time_monthly_list.append(datetime(tmp_range_time.year, tmp_range_time.month, 1).strftime("%Y-%m-%d"))
    
    tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
    tmp_range_time_add_month = datetime(tmp_range_time_add_month.year, tmp_range_time_add_month.month, 1)
    
    while tmp_range_time_add_month < KG_time_range_list[1]:

        KG_time_monthly_list.append(tmp_range_time_add_month.strftime("%Y-%m-%d"))
        
        tmp_range_time = tmp_range_time_add_month
        tmp_range_time_add_month = tmp_range_time + relativedelta(months = 1)
        
    return KG_time_monthly_list