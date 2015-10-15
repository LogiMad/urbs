import os
from coopr.opt.base import SolverFactory
import pandas as pd
import urbs_configuration_set as conf
import urbs_package.urbs_preprocessor as pre
import urbs_package.lp_model as lp_model
import urbs_package.pwlp_model as pwlp_model
import urbs_package.urbs_postprocessor as post
import importlib

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

    # scenario name and modify data for scenario
    sce = scenario.__name__
    data = scenario(data)

    # create model
    # opt_model = importlib.import_module(opt_model_name, package='urbs_package')
    # model = opt_model.create_model(data, timesteps)
    model = lp_model.create_model(data, timesteps)
    prob = model.create()

    # refresh time stamp string and create filename for logfile
    now = prob.created
    log_filename = os.path.join(result_dir, '{}-{}.log').format(sce, now)
    print('model created')
    # solve model and read results
    optim = SolverFactory(conf.optimization_solver)  # cplex, glpk, gurobi, ...
    optim = pre.setup_solver(optim, logfile=log_filename)
    result = optim.solve(prob, tee=True)
    prob.load(result)

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
    return prob

def run_compare_scenarios(input_file, opt_model_name, timesteps, scenarios, comp_filename, periods):
    result_name = os.path.splitext(input_file)[0]  # cut away file extension
    result_dir = pre.prepare_result_directory(result_name)  # name + time stamp

    for country, color in conf.my_colors.iteritems():
        conf.COLORS[country] = color

    # Read Excel input file and prepare URBS input dict.
    data = pre.read_excel(input_file)

    data['process_commodity'] = pre.generate_pw_brk_pts(data['process_commodity'], 'ratio', 'charac-eq',
                                                        conf.x_start, conf.x_end, conf.x_step, conf.tolerance)

    costs = []
    esums = []
    for scenario in scenarios:
        prob = run_scenario(data, opt_model_name, timesteps, scenario,
                            result_dir, plot_periods=periods)

        cost, esum = post.get_esums_costs(prob, prob.com_demand, prob.sit)
        costs.append(cost)
        esums.append(esum)

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


if __name__ == '__main__':
    run_compare_scenarios(conf.input_file, conf.opt_model_name, conf.timesteps, conf.scenarios, conf.comp_filename, conf.periods)
