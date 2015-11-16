# URBS User Interface
input_file = 'mimo-example.xlsx'  # e.g.: 'mimo-example.xlsx', 'mimo-example_new.xlsx'

opt_model_name = 'urbs.lp_model'  # e.g.: 'urbs.lp_model', 'urbs.pwlp_model'

optimization_solver = 'glpk'  # e.g.: cplex, glpk, gurobi, ...

# region SCENARIOS
def scenario_base(data):
    """do nothing"""
    return data

def scenario_stock_prices(data):
    """change stock commodity prices"""
    co = data['commodity']
    stock_commodities_only = (co.index.get_level_values('Type') == 'Stock')
    co.loc[stock_commodities_only, 'price'] *= 1.5
    return data

def scenario_co2_limit(data):
    """change global CO2 limit"""
    hacks = data['hacks']
    hacks.loc['Global CO2 limit', 'Value'] *= 0.5
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

# select scenarios to be run
scenarios = [
    scenario_base,
    scenario_stock_prices,
    scenario_co2_limit,
    scenario_north_process_caps,
    scenario_all_together]
# scenarios = scenarios[:1]  # select by slicing
# endregion

# region Simulation Timesteps
(offset, length) = (5000, 10*24)  # time step selection
timesteps = range(offset, offset+length+1)
# endregion

# region Scenarios Comparison
# result's source for scenarios comparison
# (True: load the scenario results from urbs model instance. False: retrieve the results from recent spreadsheets.)
load_scenario = True  # False

# specify comparison result filename
comp_filename = 'comparison'

# endregion

# region Plotting Timesteps
periods = {
    #'spr': range(1000, 1000+24*7),
    #'sum': range(3000, 3000+24*7),
    'aut': range(5000, 5000+24*7),
    #'win': range(7000, 7000+24*7),
}
# endregion

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

# add or change plot colors
my_colors = {
    'South': (230, 200, 200),
    'Mid': (200, 230, 200),
    'North': (200, 200, 230)}
# endregion

# region generate_pw_brk_pts function settings:
x_start = 0.0  # x_start: the domain lower bound of the normalized characteristic function.x_start = 0.0
x_end = 1.0  # x_end: the domain upper bound of the normalized characteristic function.
x_step = 0.01  # x_step: the step size of domain check points to find the break points taking the tolerance into account.
tolerance = 0.5  # tolerance: uncertainty tolerance of the piecewise linearization compared to characteristic function.
# endregion

if __name__ == '__main__':
    import urbs.urbs_main as main
    main.run_compare_scenarios(input_file, opt_model_name, timesteps, scenarios, comp_filename, periods)
