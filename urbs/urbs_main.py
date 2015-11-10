import os, sys
import time
from coopr.opt.base import SolverFactory
import pandas as pd
import preprocessor as pre
import postprocessor as post
from importlib import import_module
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import urbs_user_interface as conf

def run_scenario(data, opt_model_name, timesteps, scenario, result_dir, plot_periods={}):
    """ run an urbs model for given input, time steps and scenario

    Args:
        input_file: filename to an Excel spreadsheet for urbs.read_excel
        timesteps: a list of timesteps, e.g. range(0,8761)
        scenario: a scenario function that modifies the input data dict
        result_dir: directory name for result spreadsheet and plots

    Returns:
        the urbs model instance
    """
    sce_begin_time = time.clock()
    # scenario name and modify data for scenario
    sce = scenario.__name__
    print '###Working on', sce, 'optimization problem###'
    data = scenario(data)
    sce_data_time = time.clock()
    print sce, 'data was ready in', (sce_data_time - sce_begin_time), 'secs!'

    # create model
    opt_model = import_module(opt_model_name, package=None)
    model = opt_model.create_model(data, timesteps)
    prob = model.create()

    # refresh time stamp string and create filename for logfile
    now = prob.created
    log_filename = os.path.join(result_dir, '{}-{}.log').format(sce, now)
    sce_creat_time = time.clock()
    print sce, 'model was created in', (sce_creat_time - sce_data_time), 'secs!'
    # solve model and read results
    optim = SolverFactory(conf.optimization_solver)  # cplex, glpk, gurobi, ...
    optim = pre.setup_solver(optim, logfile=log_filename)
    result = optim.solve(prob, tee=True)
    prob.load(result)
    sce_solver_time = time.clock()
    print sce, 'problem was solved in', (sce_solver_time - sce_creat_time), 'secs!'
    # write report to spreadsheet
    post.report(
        prob,
        os.path.join(result_dir, '{}-{}.xlsx').format(sce, now),
        prob.com_demand, prob.sit)

    # store optimisation problem for later re-analysis
    post.save(
        prob,
        os.path.join(result_dir, '{}-{}.pgz').format(sce, now))

    post.result_figures(
        prob, 
        os.path.join(result_dir, '{}-{}'.format(sce, now)),
        plot_title_prefix=sce.replace('_', ' ').title(),
        periods=plot_periods)
    sce_post_time = time.clock()
    print sce, 'report and figures were generated and the model instance was saved in', (sce_post_time - sce_solver_time), 'secs!'
    print sce, 'runtime was', (sce_post_time - sce_begin_time), 'secs!'
    print
    return prob

def run_compare_scenarios(input_file, opt_model_name, timesteps, scenarios, comp_filename, periods):
    result_name = os.path.splitext(input_file)[0]  # cut away file extension
    result_dir = pre.prepare_result_directory(result_name)  # name + time stamp

    for country, color in conf.my_colors.iteritems():
        conf.COLORS[country] = color

    # Read Excel input file and prepare URBS input dict.
    excel_begin_time = time.clock()
    data = pre.read_excel(input_file)
    if opt_model_name == 'urbs_package.pwlp_model':
        data['process_commodity'] = pre.generate_pw_brk_pts(data['process_commodity'], 'ratio', 'charac-eq',
                                                            conf.x_start, conf.x_end, conf.x_step, conf.tolerance)
    excel_end_time = time.clock()
    print'The data was loaded in', (excel_end_time - excel_begin_time), 'secs!'

    costs = []
    esums = []
    for scenario in scenarios:
        prob = run_scenario(data, opt_model_name, timesteps, scenario,
                            result_dir, plot_periods=periods)

        cost, esum = post.get_esums_costs(prob, prob.com_demand, prob.sit)
        costs.append(cost)
        esums.append(esum)

    comp_begin_time = time.clock()
    scenario_names = post.derive_scenario_names(scenarios)
    # merge everything into one DataFrame each
    costs = pd.concat(costs, axis=1, keys=scenario_names)
    esums = pd.concat(esums, axis=1, keys=scenario_names)

    # specify comparison result filename
    output_filename = os.path.join(result_dir, comp_filename)
    # drop redundant 'costs' column label make index name nicer for plot
    # sort/transpose frame convert EUR/a to 1e9 EUR/a
    costs, esums = post.comp_analyse(costs, esums)
    # make plots and reports out of compared scenarios
    post.comp_plot(costs, esums, output_filename)
    post.comp_report(costs, esums, output_filename)
    comp_end_time = time.clock()
    print 'Comparison report files were generated in', (comp_end_time - comp_begin_time), 'secs!'

if __name__ == '__main__':
    run_compare_scenarios(conf.input_file, conf.opt_model_name, conf.timesteps, conf.scenarios, conf.comp_filename, conf.periods)
