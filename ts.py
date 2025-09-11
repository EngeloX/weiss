import numpy as np
import pandas as pd
from math import ceil


# Выделить из переменной даты - НАЗВАНИЕ СЕЗОНА(кат.)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    if month in [3, 4, 5]:
        return 'Spring'
    if month in [6, 7, 8]:
        return 'Summer'
    if month in [9, 10, 11]:
        return 'Fall'

# Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР НЕДЕЛИ В МЕСЯЦЕ
def get_week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom / 7.0))


# Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР ДЕКАДЫ МЕСЯЦА(ДЕСЯТИДНЕВКА)
def get_decades(series):
    lst = []
    for d in series:
        if 1 <= d.day <= 10:
            lst.append(1)
        elif 11 <= d.day <= 20:
            lst.append(2)
        else:
            lst.append(3)
    return lst


def date_features(data, date_col, indicators=True, copy=True):
    """
    Params:
    -------
    data: pd.DataFrame
    date_col: str
    indicators: bool
    copy: bool
    """
    if copy:
        data = data.copy()
    
    data[date_col] = pd.to_datetime(data[date_col])
    
    # Выделить из переменной даты - ГОД
    data['year'] = data[date_col].dt.year
    # Выделить из переменной даты - КВАРТАЛ
    data['quarter'] = data[date_col].dt.quarter
    # Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР МЕСЯЦА
    data['month'] = data[date_col].dt.month
    # Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР ДНЯ ГОДА
    data['dayofyear'] = data[date_col].dt.dayofyear
    # Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР ДНЯ МЕСЯЦА
    data['dayofmonth'] = data[date_col].dt.day

    # Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР ДНЯ НЕДЕЛИ (0 понедельник - 6 воскресеньие / Американская система)
    data['dayofweek_usa'] = data[date_col].dt.dayofweek
    # Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР ДНЯ НЕДЕЛИ (1 - 7 / Российская система)
    data['dayofweek_rus'] = data['dayofweek_usa'] + 1
    # Выделить из переменной даты - НАЗВАНИЕ ДНЯ НЕДЕЛИ(категориальное)
    data['weekday'] = data[date_col].dt.day_name()
    # Выделить из переменной даты - НАЗВАНИЕ МЕСЯЦА(кат.)
    data['month_name'] = data[date_col].dt.month_name()
    
    
    # Выделить из переменной даты - НАЗВАНИЕ СЕЗОНА(кат.)
    data['season'] = data[date_col].dt.month.apply(lambda x: get_season(x))
    # Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР НЕДЕЛИ В МЕСЯЦЕ
    data['week_of_month'] = data[date_col].apply(get_week_of_month)
    # Выделить из переменной даты - ПОРЯДКОВЫЙ НОМЕР ДЕКАДЫ МЕСЯЦА (1-10, 11-20, 21-31)
    data['decade'] = get_decades(data[date_col])
    
    
    # Переменные индикаторы
    if indicators:
        # Выделить из переменной даты - ИНДИКАТОР начало года
        data['year_start'] = data[date_col].dt.is_year_start
        # Выделить из переменной даты - ИНДИКАТОР конец года
        data['year_end'] = data[date_col].dt.is_year_end
        # Выделить из переменной даты - ИНДИКАТОР начало квартала
        data['quarter_start'] = data[date_col].dt.is_quarter_start
        # Выделить из переменной даты - ИНДИКАТОР конец квартала
        data['quarter_end'] = data[date_col].dt.is_quarter_end
        # Выделить из переменной даты - ИНДИКАТОР начало месяца
        data['month_start'] = data[date_col].dt.is_month_start
        # Выделить из переменной даты - ИНДИКАТОР конец месяца
        data['month_end'] = data[date_col].dt.is_month_end
        
    return data