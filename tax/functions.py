from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os
import logging

"""
This file define functions to calculate tax.
"""

status_list = ['Single', #0
               'Married filing jointly', #1
               'Married filing separately', #2
               'Head of a house hold' #3
              ]


def fed_exemption(year, x, status):
    """
    Given taxable income and tax status, return federal tax exemption for 2017
    input:  year    year to be calculated
            x       AGI (adjusted gross income)
            status  a string of one of 'Single', 'Married filing jointly',
                    'Married filing separately', 'Head of a house hold'
    Return a number
    """
    if year>=2018: return 0
    num = {'Single':1, 'Married filing jointly':2,
           'Married filing separately':1, 'Head of a house hold':2}
    dirs = os.path.dirname(os.path.abspath(__file__))
    line4 = pd.read_csv(dirs+'\\fed_exemption.csv', header=0, index_col=0).fillna(np.infty)
    line2 = line4.at[year,'Mult']*num[status]
    if x <= line4.at[year,status]: return line2
    else:
        line5 = x - line4.at[year,status]
        if status=='Married filing separately':
            if line5>61250: return 0
            line6 = np.ceil(line5/1250)
        else:
            if line5>122500: return 0
            line6 = np.ceil(line5/2500)
        line8 = line2 * line6*0.02
        return np.round(line2 - line8, 2)


def load_NJ_tax_table(file, year=None):
    if year is None:
        from datetime import date
        year = date.today().year-1
    dirs = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(file, sep=' ', header=None, index_col=None, thousands=',')
    data = pd.DataFrame(np.concatenate([data.iloc[:,:4].values, data.iloc[:,4:8].values,
                       data.iloc[:,8:12].values, data.iloc[:,12:16].values], axis=0),
                    columns=['At least','But less than','Single','Married filing jointly']).sort_values('At least')
    data["Married filing separately"] = data['Single']
    data["Head of a house hold"] = data['Married filing jointly']
    data.to_csv(dirs+'/{}NJ.csv'.format(year), index=None)


def load_NY_tax_table(file, year=None):
    if year is None:
        from datetime import date
        year = date.today().year-1
    dirs = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(file, sep=' ', header=None, index_col=None, thousands=',')
    data.columns = ['At least','But less than','Single',
                    'Married filing jointly','Head of a house hold']
    data["Married filing separately"] = data['Single']
    data.to_csv(dirs+'/{}NY.csv'.format(year), index=None)


def tax_from_taxable(year, x, status, state=""):
    """
    Given taxable income and tax status, return tax
    input:  year    year to be calculated
            x       taxable income
            status  a string of one of 'Single', 'Married filing jointly',
                    'Married filing separately', 'Head of a house hold'
            state   the state to calculate, default "" (federal)
    Return a number
    """
    dirs = os.path.dirname(os.path.abspath(__file__))
    try:
        data = pd.read_csv(dirs+'\\{}{}.csv'.format(year, state), header=0)
        i = data[(data.iloc[:,0]<=x)&(x<data.iloc[:,1])].index[0]
        return data.loc[i,status]
    except:
        data = pd.read_csv(dirs+'\\{}{}.txt'.format(year, state), sep=' ',
                           header=None, index_col=None).fillna(np.infty)
        data.columns = ['status','start','end','eval']
        i = data[(data.status==status.replace(' ','-'))&(data.start<=x)&(x<data.end)].index[0]
        return eval(data.at[i,'eval'].replace('x',"{}".format(x)))


def NY_2018_tax(x, y, status):
    """
    Given AGI, taxable income and tax status, return NY state tax
    input:  x       AGI (adjusted gross income)
            y       taxable income
            status  a string of one of 'Single', 'Married filing jointly',
                    'Married filing separately', 'Head of a house hold'
    Return a number
    """
    tax = tax_from_taxable(2018, y, status, state="NY")
    def calc(rate, diff, floor):
        return tax + diff + (rate*y-tax-diff) * min(1, np.round((x-floor)/50000, 4))
    if x<=107650:
        return tax
    else:
        if status=='Married filing jointly':
            if 107650<x and x<=2155350 and y<=161550:               return calc(0.0633, 0, 107650)
            if 161550<x and x<=2155350 and 161550<y and y<=323200:  return calc(0.0657, 629, 161550)
            if 323200<x and x<=2155350 and 323200<y:                return calc(0.0685, 1017, 323200)
            if 2155350<x and y<=161550:                             return calc(0.0882, 629, 2155350)
            if 2155350<x and 161550<y and y<=323200:                return calc(0.0882, 1017, 2155350)
            if 2155350<x and 323200<y:                              return calc(0.0882, 1922, 2155350)
        if status=='Single' or status=='Married filing separately':
            if 107650<x and x<=1077550 and y<=215400:               return calc(0.0657, 0, 107650)
            if 215400<x and x<=1077550 and 215400<y:                return calc(0.0685, 506, 215400)
            if 1077550<x and y<=215400:                             return calc(0.0882, 506, 1077550)
            if 1077550<x and 215400<y:                              return calc(0.0882, 1109, 1077550)
        if status=='Head of a house hold':
            if 107650<x and x<=1616450 and y<=369300:               return calc(0.0657, 0, 107650)
            if 269300<x and x<=1616450 and 369300<y:                return calc(0.0685, 729, 269300)
            if 1616450<x and y<=369300:                             return calc(0.0882, 729, 1616450)
            if 1616450<x and 369300<y:                              return calc(0.0882, 1483, 1616450)
    return -1


def taxable_from_agi(year, status, x, state):
    """
    Calculate default taxable income from adjusted gross income
    input:  year    year to be calculated
            status  a string of one of 'Single', 'Married filing jointly',
                    'Married filing separately', 'Head of a house hold'
            x       AGI (adjusted gross income)
            state   the state to calculate, default "" (federal)
    Return a number
    """
    dirs = os.path.dirname(os.path.abspath(__file__))
    deduction = pd.read_csv(dirs+'\\deduction.csv', header=0, index_col=None).fillna("")
    deduction = deduction[(deduction.Year==year)&(deduction.State==state)]
    deduction = deduction[status].iat[0]
    if year==2017 and state=="":
        deduction += fed_exemption(year, x, status)
    return max(0, x-deduction)


def calculate_tax(year, status, x=None, y=None, state=""):
    """
    Given taxable income and tax status, return tax
    input:  year    year to be calculated
            status  a string of one of 'Single', 'Married filing jointly',
                    'Married filing separately', 'Head of a house hold'
            x       AGI (adjusted gross income), default None
            y       taxable income, default None
            state   the state to calculate, default "" (federal)
    Either x or y should not be None
    Return a number
    """
    if (x is None) and (y is None):
            raise ValueError("Either x or y should not be None")
    if y is None:
        try:
            y = taxable_from_agi(year, status, x, state)
        except:
            raise ValueError("Cannot calculate default taxable income.")
    if state=='NY':
        return NY_2018_tax(x, y, status)
    return tax_from_taxable(year, y, status, state)


def tax_value(year, status, state=""):
#     dirs = os.path.dirname(os.path.abspath(__file__))
    dirs = r'C:\Users\Emma\Documents\code\PyLib\tax'
    x = np.array([])
    y = np.array([])
    bottom = 0
    try:
        data = pd.read_csv(dirs+'\\{}{}.csv'.format(year, state), header=0)
        x = data[["At least","But less than"]].astype(float).values
        x[:,1] = x[:,1] - 0.01
        y = data.loc[:, status]
        x = x.ravel()
        y = np.repeat(y,2)
        bottom = x[-1] + 0.01
    except:
        pass
    try:
        top = 5.1e5
        data = pd.read_csv(dirs+'\\{}{}.txt'.format(year, state), sep=' ',
                           header=None, index_col=None).fillna(top)
        data.columns = ['status','start','end','eval']
        data = data[(data.status==status.replace(' ','-'))]
        for i in data.index:
            if data.at[i,'start']>=bottom:
                x = np.append(x, data.at[i,'start'])
                y = np.append(y, eval(data.at[i,'eval'].replace('x',"{}".format(data.at[i,'start']))))
        x = np.append(x, top)
        y = np.append(y, eval(data.iat[-1,-1].replace('x',"{}".format(top))))
    except Exception as err:
        print(err)
        pass
    return x, y


def tax_rate(year, status, state=""):
    import sys
    import os
    path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
    if not path in sys.path:
        sys.path.append(path)
    from basic.mathe import pwl_derivative
    dirs = os.path.dirname(os.path.abspath(__file__))
    x = np.array([])
    y = np.array([])
    bottom = 0
    try:
        data = pd.read_csv(dirs+'\\{}{}.csv'.format(year, state), header=0)
        x, y = pwl_derivative(data["But less than"].values, data[status].values)
    except:
        pass
    try:
        top = 5e6
        data = pd.read_csv(dirs+'\\{}{}.txt'.format(year, state), sep=' ',
                           header=None, index_col=None).fillna(top)
        data.columns = ['status','start','end','eval']
        data = data[(data.status==status.replace(' ','-'))]
        px = []
        py = []
        for i in data.index:
            if data.at[i,'start']>=bottom:
                px = np.append(px, data.at[i,'start'])
                py = np.append(py, eval(data.at[i,'eval'].replace('x',"{}".format(data.at[i,'start']))))
        px = np.append(px, top)
        py = np.append(py, eval(data.iat[-1,-1].replace('x',"{}".format(top))))
        a, b = pwl_derivative(px, py)
        x = np.append(x, a)
        y = np.append(y, b)
    except Exception as err:
        print(err)
        pass
    return x, y


def plot_tax(year, start=100, end=2e5, nums=200, log=False, ratio=False, state="", AGI=True):
    """
    input:  year    year to be calculated
            start   minimum value of taxable income, default $100
            end     maximum value of taxable income, default $200,000
            nums    number of points, default 200
            log     indicate whether to plot x axis in log scale
            ratio   indicate whether to plot ratio or value
            state   the state to calculate, default "" (federal)
            AGI     indicate whether the input is AGI, default True
    """
    import matplotlib.pyplot as plt
    if log:
        index = np.logspace(np.log10(start), np.log10(end), nums, endpoint=True)
    else:
        index = np.linspace(start, end, nums, endpoint=True)
    summary = pd.DataFrame(0.0, index=np.round(index), columns=status_list)
    for i in summary.index:
        for j in summary.columns:
            if AGI:
                summary.at[i,j] = calculate_tax(year, j, x=i, state=state)
            else:
                summary.at[i,j] = calculate_tax(year, j, y=i, state=state)
    if ratio:
        summary = summary.apply(lambda x:x/summary.index.values)
    summary.plot(figsize=(14,4), grid=True, logx=log)
    plt.xlabel('Taxable income')
    if ratio:
        plt.ylabel('Tax ratio')
    else:
        plt.ylabel('Tax value')


if __name__=="__main__":
    import argparse
    import functions
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Functions for tax calculation")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(functions, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(functions, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
