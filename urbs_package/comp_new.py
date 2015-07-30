import glob
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import os
import pandas as pd
import sys
import urbs_configuration_set as conf
import urbs_preprocessor as pre
import urbs_postprocessor as post


def compare_scenarios(comp_filename, result_dir=None):
    """ Create report sheet and plots for given report spreadsheets.
    
    Args:
        result_files: a list of spreadsheet filenames generated by post.report
        output_filename: a spreadsheet filename that the comparison is to be 
                         written to
                         
     Returns:
        Nothing
    
    To do: 
        Don't use report spreadsheets, instead load pickled problem 
        instances. This would make this function less fragile and dependent
        on the output format of post.report().
    """

    def comp_analyse(costs, esums):
        """drop redundant 'costs' column label
        make index name nicer for plot
        sort/transpose frame
        convert EUR/a to 1e9 EUR/a
        """
        costs.columns = costs.columns.droplevel(1)
        costs.index.name = 'Cost type'
        costs = costs.sort().transpose()
        costs = costs / 1e9

        # sum up created energy over all locations, but keeping scenarios (level=0)
        # make index name 'Commodity' nicer for plot
        # drop all unused commodities and sort/transpose
        # convert MWh to GWh
        esums = esums.loc['Created'].sum(axis=1, level=0)
        esums.index.name = 'Commodity'
        used_commodities = (esums.sum(axis=1) > 0)
        esums = esums[used_commodities].sort().transpose()
        esums = esums / 1e3
        return costs, esums

    def comp_plot(costs, esums):
        fig = plt.figure(figsize=(20, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3])

        ax0 = plt.subplot(gs[0])
        bp0 = costs.plot(ax=ax0, kind='barh', stacked=True)

        ax1 = plt.subplot(gs[1])
        esums_colors = [post.to_color(commodity) for commodity in esums.columns]
        bp1 = esums.plot(ax=ax1, kind='barh', stacked=True, color=esums_colors)

        # remove scenario names from second plot
        ax1.set_yticklabels('')

        # make bar plot edges lighter
        for bp in [bp0, bp1]:
            for patch in bp.patches:
                patch.set_edgecolor(post.to_color('Decoration'))

        # set limits and ticks for both axes
        for ax in [ax0, ax1]:
            plt.setp(ax.spines.values(), color=post.to_color('Grid'))
            ax.yaxis.grid(False)
            ax.xaxis.grid(True, 'major', color=post.to_color('Grid'), linestyle='-')
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')

            # group 1,000,000 with commas
            group_thousands = tkr.FuncFormatter(lambda x, pos: '{:0,d}'.format(int(x)))
            ax.xaxis.set_major_formatter(group_thousands)

            # legend
            lg = ax.legend(frameon=False, loc='upper center',
                           ncol=5,
                           bbox_to_anchor=(0.5, 1.11))
            plt.setp(lg.get_patches(), edgecolor=post.to_color('Decoration'),
                     linewidth=0.15)

        ax0.set_xlabel('Total costs (1e9 EUR/a)')
        ax1.set_xlabel('Total energy produced (GWh)')

        for ext in ['png', 'pdf']:
            fig.savefig('{}.{}'.format(output_filename, ext),
                        bbox_inches='tight')
        return costs, esums

    def comp_report(costs, esums):
        with pd.ExcelWriter('{}.{}'.format(output_filename, 'xlsx')) as writer:
            costs.to_excel(writer, 'Costs')
            esums.to_excel(writer, 'Energy sums')

    if not result_dir:
        # get the directory of the supposedly last run
        search_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'result'))
        result_dir = get_most_recent_entry(search_dir)

    if conf.load_scenario:
        for scenario in conf.scenarios:
            inst_file = glob.glob(os.path.join(result_dir, '{}-*.pgz'.format(scenario.__name__)))[0]
            print(inst_file)
            sce_prob = pre.load(inst_file)
            print(sce_prob.sit)
            costs, esums = get_esums_costs(sce_prob, sce_prob.com_demand, sce_prob.sit)
    else:
        # retrieve (glob) a list of all result spreadsheets from result_dir
        result_files = glob_result_files(result_dir)
        scenario_names = derive_scenarios(result_files)
        costs, esums = read_files(scenario_names, result_files)
    output_filename = os.path.join(result_dir, comp_filename)

    costs, esums = comp_analyse(costs, esums)
    costs, esums = comp_plot(costs, esums)
    comp_report(costs, esums)

# region Helper functions
def get_most_recent_entry(search_dir):
    """ Return most recently modified entry from given directory.

    Args:
        search_dir: an absolute or relative path to a directory

    Returns:
        The file/folder in search_dir that has the most recent 'modified'
        datetime.
    """
    entries = glob.glob(os.path.join(search_dir, "*"))
    entries.sort(key=lambda x: os.path.getmtime(x))
    return entries[-1]

def glob_result_files(folder_name):
    """ Glob result spreadsheets from specified folder.

    Args:
        folder_name: an absolute or relative path to a directory

    Returns:
        list of filenames that match the pattern 'scenario_*.xlsx'
    """
    glob_pattern = os.path.join(folder_name, 'scenario_*.xlsx')
    result_files = sorted(glob.glob(glob_pattern))
    return result_files

def derive_scenarios(result_files):
        """derive list of scenario names for column labels/figure captions"""
        scenario_names = [os.path.basename(rf) # drop folder names, keep filename
                          .replace('_', ' ') # replace _ with spaces
                          .replace('.xlsx', '') # drop file extension
                          .replace('scenario ', '') # drop 'scenario ' prefix
                          for rf in result_files]
        scenario_names = [s[0:s.find('-')] for s in scenario_names] # drop everything after first '-'

        # find base scenario and put at first position
        try:
            base_scenario = scenario_names.index('base')
            result_files.insert(0, result_files.pop(base_scenario))
            scenario_names.insert(0, scenario_names.pop(base_scenario))
        except ValueError:
            pass  # do nothing if no base scenario is found

        return scenario_names

def read_files(scenario_names, result_files):
        costs = []  # total costs by type and scenario
        esums = []  # sum of energy produced by scenario

        for rf in result_files:
            with pd.ExcelFile(rf) as xls:
                cost = xls.parse('Costs', has_index_names=True)
                esum = xls.parse('Energy sums')

                # repair broken MultiIndex in the first column
                esum.reset_index(inplace=True)
                esum.fillna(method='ffill', inplace=True)
                esum.set_index(['level_0', 'level_1'], inplace=True)

                costs.append(cost)
                esums.append(esum)

        # merge everything into one DataFrame each
        costs = pd.concat(costs, axis=1, keys=scenario_names)
        esums = pd.concat(esums, axis=1, keys=scenario_names)
        return costs, esums

def get_esums_costs(instance, commodities=None, sites=None):
        # get the data
        costs = post.get_constants(instance)[0]
        esums = post.summaries_timeseries(instance, commodities=None, sites=None)[0]
        return costs, esums

# endregion
if __name__ == '__main__':
        # and run the comparison function
        compare_scenarios(conf.comp_filename, result_dir=None)
