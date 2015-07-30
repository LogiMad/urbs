import pandas as pd
import os
import urbs_configuration_set as conf
from datetime import datetime

# region SCENARIOS
def scenario_base(data):
    """do nothing"""
    return data

def scenario_stock_prices(data):
    """change stock commodity prices"""
    co = data['commodity']
    stock_commodities_only = (co.index.get_level_values('Type') == 'Stock')
    co.loc[stock_commodities_only, 'price'] *= conf.stock_prices_multiplier
    return data

def scenario_co2_limit(data):
    """change global CO2 limit"""
    hacks = data['hacks']
    hacks.loc['Global CO2 limit', 'Value'] *= conf.co2_limit_multiplier
    return data

def scenario_north_process_caps(data):
    """change maximum installable capacity"""
    pro = data['process']
    pro.loc[('North', 'Hydro plant'), 'cap-up'] *= 0.5
    pro.loc[('North', 'Biomass plant'), 'cap-up'] *= 0.25
    return data

def scenario_all_together(data):
    """combine all other scenarios"""
    data = scenario_stock_prices(data)
    data = scenario_co2_limit(data)
    data = scenario_north_process_caps(data)
    return data
# endregion

def prepare_result_directory(result_name):
    """ create a time stamped directory within the result folder """
    # timestamp for result directory
    now = datetime.now().strftime('%Y%m%dT%H%M%S')

    # create result directory if not existent
    result_dir = os.path.join('result', '{}-{}'.format(result_name, now))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir

def setup_solver(optim, logfile='solver.log'):
    """ """
    if optim.name == 'gurobi':
        # reference with list of option names
        # http://www.gurobi.com/documentation/5.6/reference-manual/parameters
        optim.set_options("logfile={}".format(logfile))
        # optim.set_options("timelimit=7200")  # seconds
        # optim.set_options("mipgap=5e-4")  # default = 1e-4
    elif optim.name == 'glpk':
        # reference with list of options
        # execute 'glpsol --help'
        optim.set_options("log={}".format(logfile))
        # optim.set_options("tmlim=7200")  # seconds
        # optim.set_options("mipgap=.0005")
    else:
        print("Warning from setup_solver: no options set for solver "
              "'{}'!".format(optim.name))
    return optim

def read_excel(filename):
    """Read Excel input file and prepare URBS input dict.

    Reads an Excel spreadsheet that adheres to the structure shown in
    mimo-example.xlsx. Two preprocessing steps happen here:
    1. Column titles in 'Demand' and 'SupIm' are split, so that
    'Site.Commodity' becomes the MultiIndex column ('Site', 'Commodity').
    2. The attribute 'annuity-factor' is derived here from the columns 'wacc'
    and 'depreciation' for 'Process', 'Transmission' and 'Storage'.

    Args:
        filename: filename to an Excel spreadsheet with the required sheets
            'Commodity', 'Process', 'Transmission', 'Storage', 'Demand' and
            'SupIm'.

    Returns:
        a dict of 6 DataFrames

    Example:
        >>> data = read_excel('mimo-example.xlsx')
        >>> data['hacks'].loc['Global CO2 limit', 'Value']
        150000000
    """
    with pd.ExcelFile(filename) as xls:
        commodity = xls.parse(
            'Commodity',
            index_col=['Site', 'Commodity', 'Type'])
        process = xls.parse(
            'Process',
            index_col=['Site', 'Process'])
        process_commodity = xls.parse(
            'Process-Commodity',
            index_col=['Process', 'Commodity', 'Direction'])
        transmission = xls.parse(
            'Transmission',
            index_col=['Site In', 'Site Out', 'Transmission', 'Commodity'])
        storage = xls.parse(
            'Storage',
            index_col=['Site', 'Storage', 'Commodity'])
        demand = xls.parse(
            'Demand',
            index_col=['t'])
        supim = xls.parse(
            'SupIm',
            index_col=['t'])
        try:
            hacks = xls.parse(
                'Hacks',
                index_col=['Name'])
        except XLRDError:
            hacks = None

    # prepare input data
    # split columns by dots '.', so that 'DE.Elec' becomes the two-level
    # column index ('DE', 'Elec')
    demand.columns = split_columns(demand.columns, '.')
    supim.columns = split_columns(supim.columns, '.')

    # derive annuity factor from WACC and depreciation periods
    process['annuity-factor'] = annuity_factor(
        process['depreciation'], process['wacc'])
    transmission['annuity-factor'] = annuity_factor(
        transmission['depreciation'], transmission['wacc'])
    storage['annuity-factor'] = annuity_factor(
        storage['depreciation'], storage['wacc'])

    data = {
        'commodity': commodity,
        'process': process,
        'process_commodity': process_commodity,
        'transmission': transmission,
        'storage': storage,
        'demand': demand,
        'supim': supim}
    if hacks is not None:
        data['hacks'] = hacks

    # sort nested indexes to make direct assignments work, cf
    # http://pandas.pydata.org/pandas-docs/stable/indexing.html#the-need-for-sortedness-with-multiindex
    for key in data:
        if isinstance(data[key].index, pd.core.index.MultiIndex):
            data[key].sortlevel(inplace=True)
    return data

def load(filename):
    """Load a urbs model instance from a gzip'ed pickle file

    Args:
        filename: pickle file

    Returns:
        prob: the unpickled urbs model instance
    """
    import gzip
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with gzip.GzipFile(filename, 'r') as file_handle:
        prob = pickle.load(file_handle)
    return prob

# region Helper functions
def annuity_factor(n, i):
    """Annuity factor formula.

    Evaluates the annuity factor formula for depreciation duration
    and interest rate. Works also well for equally sized numpy arrays
    of values for n and i.

    Args:
        n: depreciation period (years)
        i: interest rate (percent, e.g. 0.06 means 6 %)

    Returns:
        Value of the expression :math:`\\frac{(1+i)^n i}{(1+i)^n - 1}`

    Example:
        >>> round(annuity_factor(20, 0.07), 5)
        0.09439

    """
    return (1+i)**n * i / ((1+i)**n - 1)

def split_columns(columns, sep='.'):
    """Split columns by separator into MultiIndex.

    Given a list of column labels containing a separator string (default: '.'),
    derive a MulitIndex that is split at the separator string.

    Args:
        columns: list of column labels, containing the separator string
        sep: the separator string (default: '.')

    Returns:
        a MultiIndex corresponding to input, with levels split at separator

    Example:
        >>> split_columns(['DE.Elec', 'MA.Elec', 'NO.Wind'])
        MultiIndex(levels=[[u'DE', u'MA', u'NO'], [u'Elec', u'Wind']],
                   labels=[[0, 1, 2], [0, 0, 1]])

    """
    column_tuples = [tuple(col.split('.')) for col in columns]
    return pd.MultiIndex.from_tuples(column_tuples)

# endregion
