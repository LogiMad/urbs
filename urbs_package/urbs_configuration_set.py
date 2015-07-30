import urbs_preprocessor as pre

input_file = 'mimo-example.xlsx'

# scenario multipliers
stock_prices_multiplier = 1.5
co2_limit_multiplier = 0.5

# select scenarios to be run
scenarios = [
    pre.scenario_base,
    pre.scenario_stock_prices,
    pre.scenario_co2_limit,
    pre.scenario_north_process_caps,
    pre.scenario_all_together]
#scenarios = scenarios[:1]  # select by slicing

optimization_solver='glpk' # cplex, glpk, gurobi, ...

# simulation timesteps
(offset, length) = (5000, 10*24)  # time step selection
timesteps = range(offset, offset+length+1)

# result's source for scenarios comparison
# (True: load the scenario results from urbs model instance. False: retrieve the results from recent spreadsheets.)
load_scenario = True

# specify comparison result filename
comp_filename = 'comparison'

# plotting timesteps
periods = {
    #'spr': range(1000, 1000+24*7),
    #'sum': range(3000, 3000+24*7),
    'aut': range(5000, 5000+24*7),
    #'win': range(7000, 7000+24*7),
}

# region Colors
COLORS = {
    'Biomass plant': (0, 122, 55),
    'Coal plant': (100, 100, 100),
    'Gas plant': (237, 227, 0),
    'Hydro plant': (198, 188, 240),
    'Lignite plant': (116, 66, 65),
    'Photovoltaics': (243, 174, 0),
    'Slack powerplant': (163, 74, 130),
    'Wind park': (122, 179, 225),
    'Decoration': (128, 128, 128),  # plot labels
    'Demand': (25, 25, 25),  # thick demand line
    'Grid': (128, 128, 128),  # background grid
    'Overproduction': (190, 0, 99),  # excess power
    'Storage': (60, 36, 154),  # storage area
    'Stock': (222, 222, 222)}  # stock commodity power
# endregion

# add or change plot colors
my_colors = {
    'South': (230, 200, 200),
    'Mid': (200, 230, 200),
    'North': (200, 200, 230)}

