import coopr.environ
import os
from coopr.opt.base import SolverFactory
from datetime import datetime
import urbs_configuration_set as conf
import urbs_package.urbs_preprocessor as pre
import urbs_package.lp_model as lp_model
import urbs_package.urbs_postprocessor as post

def run_scenario(input_file, timesteps, scenario, result_dir, plot_periods={}):
    """ run an urbs model for given input, time steps and scenario

    Args:
        input_file: filename to an Excel spreadsheet for urbs.read_excel
        timesteps: a list of timesteps, e.g. range(0,8761)
        scenario: a scenario function that modifies the input data dict
        result_dir: directory name for result spreadsheet and plots

    Returns:
        the urbs model instance
    """

    # scenario name, read and modify data for scenario
    sce = scenario.__name__
    data = pre.read_excel(input_file)
    data = scenario(data)

    # create model
    model = lp_model.create_model(data, timesteps)
    prob = model.create()

    # refresh time stamp string and create filename for logfile
    now = prob.created
    log_filename = os.path.join(result_dir, '{}-{}.log').format(sce, now)

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
        os.path.join(result_dir, '{}-{}'.format(sce,now)),
        plot_title_prefix=sce.replace('_', ' ').title(),
        periods=plot_periods)
    return prob

if __name__ == '__main__':
    result_name = os.path.splitext(conf.input_file)[0]  # cut away file extension
    result_dir = pre.prepare_result_directory(result_name)  # name + time stamp

    for country, color in conf.my_colors.iteritems():
        conf.COLORS[country] = color

    for scenario in conf.scenarios:
        prob = run_scenario(conf.input_file, conf.timesteps, scenario,
                            result_dir, plot_periods=conf.periods)

    # compare scenarios and make report and graph

    # specify comparison result filename
    # and run the comparison function
    comp_filename = os.path.join(directory, 'comparison')
    for scenario in conf.scenarios:
        sce_prob = pre.load(os.path.join(result_dir, '{}-*.pgz'.format(scenario)))
        get_esums_costs(sce_prob, sce_prob.com_demand, sce_prob.sit)
        compare_scenarios(sce_prob, comp_filename)
