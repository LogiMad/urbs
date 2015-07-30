
import coopr.pyomo as pyomo
import pandas as pd
import matplotlib.pyplot as plt
from random import random
import urbs_configuration_set as conf

def report(instance, filename, commodities=None, sites=None):
    """Write result summary to a spreadsheet file

    Args:
        instance: a urbs model instance
        filename: Excel spreadsheet filename, will be overwritten if exists
        commodities: optional list of commodities for which to write timeseries
        sites: optional list of sites for which to write timeseries

    Returns:
        Nothing
    """

    # create spreadsheet writer object
    with pd.ExcelWriter(filename) as writer:
        # get the data
        costs, cpro, ctra, csto = get_constants(instance)
        # write constants to spreadsheet
        costs.to_excel(writer, 'Costs')
        cpro.to_excel(writer, 'Process caps')
        ctra.to_excel(writer, 'Transmission caps')
        csto.to_excel(writer, 'Storage caps')

        # get the energy and timeseries summary
        energy_sums, timeseries = summaries_timeseries(instance, filename, commodities=None, sites=None)
        # write timeseries data (if any)
        if timeseries:
            energy_sums.to_excel(writer, 'Energy sums')
            # write timeseries to individual sheets
            for co in commodities:
                for sit in sites:
                    # sheet names cannot be longer than 31 characters...
                    sheet_name = "{}.{} timeseries".format(co, sit)[:31]
                    timeseries[(co, sit)].to_excel(writer, sheet_name)

def plot(prob, com, sit, timesteps=None):
    """Plot a stacked timeseries of commodity balance and storage.

    Creates a stackplot of the energy balance of a given commodity, together
    with stored energy in a second subplot.

    Args:
        prob: urbs model instance
        com: commodity name to plot
        sit: site name to plot
        timesteps: optional list of  timesteps to plot; default: prob.tm

    Returns:
        fig: figure handle
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if timesteps is None:
        # default to all simulated timesteps
        timesteps = sorted(get_entity(prob, 'tm').index)

    # FIGURE
    fig = plt.figure(figsize=(16, 8))
    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    created, consumed, stored, imported, exported = get_timeseries(
        prob, com, sit, timesteps)

    costs, cpro, ctra, csto = get_constants(prob)

    # move retrieved/stored storage timeseries to created/consumed and
    # rename storage columns back to 'storage' for color mapping
    created = created.join(stored['Retrieved'])
    consumed = consumed.join(stored['Stored'])
    created.rename(columns={'Retrieved': 'Storage'}, inplace=True)
    consumed.rename(columns={'Stored': 'Storage'}, inplace=True)

    # only keep storage content in storage timeseries
    stored = stored['Level']

    # add imported/exported timeseries
    created = created.join(imported)
    consumed = consumed.join(exported)

    # move demand to its own plot
    demand = consumed.pop('Demand')

    # remove all columns from created which are all-zeros in both created and
    # consumed (except the last one, to prevent a completely empty frame)
    for col in created.columns:
        if not created[col].any() and len(created.columns) > 1:
            if col not in consumed.columns or not consumed[col].any():
                created.pop(col)

    # PLOT CREATED
    ax0 = plt.subplot(gs[0])
    sp0 = ax0.stackplot(created.index, created.as_matrix().T, linewidth=0.15)

    # Unfortunately, stackplot does not support multi-colored legends itself.
    # Therefore, a so-called proxy artist - invisible objects that have the
    # correct color for the legend entry - must be created. Here, Rectangle
    # objects of size (0,0) are used. The technique is explained at
    # http://stackoverflow.com/a/22984060/2375855
    proxy_artists = []
    for k, commodity in enumerate(created.columns):
        commodity_color = to_color(commodity)

        sp0[k].set_facecolor(commodity_color)
        sp0[k].set_edgecolor(to_color('Decoration'))

        proxy_artists.append(mpl.patches.Rectangle(
            (0, 0), 0, 0, facecolor=commodity_color))

    # label
    ax0.set_title('Energy balance of {} in {}'.format(com, sit))
    ax0.set_ylabel('Power (MW)')

    # legend
    # add "only" consumed commodities to the legend
    lg_items = tuple(created.columns)
    for item in consumed.columns:
        # if item not in created add to legend, except items
        # from consumed which are all-zeros
        if item in created.columns or not consumed[item].any():
            pass
        else:
            # add item/commodity is not consumed
            commodity_color = to_color(item)
            proxy_artists.append(mpl.patches.Rectangle(
                (0, 0), 0, 0, facecolor=commodity_color))
            lg_items = lg_items + (item,)

    lg = ax0.legend(proxy_artists,
                    lg_items,
                    frameon=False,
                    ncol=len(proxy_artists),
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.01))
    plt.setp(lg.get_patches(), edgecolor=to_color('Decoration'),
             linewidth=0.15)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # PLOT CONSUMED
    sp00 = ax0.stackplot(consumed.index, -consumed.as_matrix().T,
                         linewidth=0.15)

    # color
    for k, commodity in enumerate(consumed.columns):
        commodity_color = to_color(commodity)

        sp00[k].set_facecolor(commodity_color)
        sp00[k].set_edgecolor((.5, .5, .5))

    # PLOT DEMAND
    ax0.plot(demand.index, demand.values, linewidth=1.2,
             color=to_color('Demand'))

    # PLOT STORAGE
    ax1 = plt.subplot(gs[1], sharex=ax0)
    sp1 = ax1.stackplot(stored.index, stored.values, linewidth=0.15)

    # color
    sp1[0].set_facecolor(to_color('Storage'))
    sp1[0].set_edgecolor(to_color('Decoration'))

    # labels & y-limits
    ax1.set_xlabel('Time in year (h)')
    ax1.set_ylabel('Energy (MWh)')
    try:
        ax1.set_ylim((0, csto.loc[sit, :, com]['C Total'].sum()))
    except KeyError:
        pass

    # make xtick distance duration-dependent
    if len(timesteps) > 26*168:
        steps_between_ticks = 168*4
    elif len(timesteps) > 3*168:
        steps_between_ticks = 168
    elif len(timesteps) > 2 * 24:
        steps_between_ticks = 24
    elif len(timesteps) > 24:
        steps_between_ticks = 6
    else:
        steps_between_ticks = 3
    xticks = timesteps[::steps_between_ticks]

    # set limits and ticks for both axes
    for ax in [ax0, ax1]:
        # ax.set_axis_bgcolor((0, 0, 0, 0))
        plt.setp(ax.spines.values(), color=to_color('Decoration'))
        ax.set_frame_on(False)
        ax.set_xlim((timesteps[0], timesteps[-1]))
        ax.set_xticks(xticks)
        ax.xaxis.grid(True, 'major', color=to_color('Grid'),
                      linestyle='-')
        ax.yaxis.grid(True, 'major', color=to_color('Grid'),
                      linestyle='-')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # group 1,000,000 with commas
        group_thousands = mpl.ticker.FuncFormatter(
            lambda x, pos: '{:0,d}'.format(int(x)))
        ax.yaxis.set_major_formatter(group_thousands)

    return fig

def result_figures(prob, figure_basename, plot_title_prefix=None, periods={}):
    """Create plot for each site and demand commodity and save to files.

    Args:
        prob: urbs model instance
        figure_basename: relative filename prefix that is shared
        plot_title_prefix: (optional) plot title identifier
        periods: (optional) dict of 'period name': timesteps_list items
                 if omitted, one period 'all' with all timesteps is assumed
    """
    # default to all timesteps if no
    if not periods:
        periods = {'all': sorted(get_entity(prob, 'tm').index)}

    # create timeseries plot for each demand (site, commodity) timeseries
    for sit, com in prob.demand.columns:
        for period, timesteps in periods.items():
            # do the plotting
            fig = plot(prob, com, sit, timesteps=timesteps)

            # change the figure title
            ax0 = fig.get_axes()[0]
            # if no custom title prefix is specified, use the figure
            if not plot_title_prefix:
                plot_title_prefix = os.path.basename(figure_basename)
            new_figure_title = ax0.get_title().replace(
                'Energy balance of ', '{}: '.format(plot_title_prefix))
            ax0.set_title(new_figure_title)

            # save plot to files
            for ext in ['png', 'pdf']:
                fig_filename = '{}-{}-{}-{}.{}'.format(
                                    figure_basename, com, sit, period, ext)
                fig.savefig(fig_filename, bbox_inches='tight')
            plt.close(fig)

def save(prob, filename):
    """Save urbs model instance to a gzip'ed pickle file.

    Pickle is the standard Python way of serializing and de-serializing Python
    objects. By using it, saving any object, in case of this function a
    Pyomo ConcreteModel, becomes a twoliner.
    <https://docs.python.org/2/library/pickle.html>
    GZip is a standard Python compression library that is used to transparently
    compress the pickle file further.
    <https://docs.python.org/2/library/gzip.html>
    It is used over the possibly more compact bzip2 compression due to the
    lower runtime. Source: <http://stackoverflow.com/a/18475192/2375855>

    Args:
        prob: a urbs model instance
        filename: pickle file to be written

    Returns:
        Nothing
    """
    import gzip
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    with gzip.GzipFile(filename, 'wb') as file_handle:
        pickle.dump(prob, file_handle, pickle.HIGHEST_PROTOCOL)

# region Helper functions
def get_entity(instance, name):
    """ Return a DataFrame for an entity in model instance.

    Args:
        instance: a Pyomo ConcreteModel instance
        name: name of a Set, Param, Var, Constraint or Objective

    Returns:
        a single-columned Pandas DataFrame with domain as index
    """

    # retrieve entity, its type and its onset names
    entity = instance.__getattribute__(name)
    labels = _get_onset_names(entity)

    # extract values
    if isinstance(entity, pyomo.Set):
        # Pyomo sets don't have values, only elements
        results = pd.DataFrame([(v, 1) for v in entity.value])

        # for unconstrained sets, the column label is identical to their index
        # hence, make index equal to entity name and append underscore to name
        # (=the later column title) to preserve identical index names for both
        # unconstrained supersets
        if not labels:
            labels = [name]
            name = name+'_'

    elif isinstance(entity, pyomo.Param):
        if entity.dim() > 1:
            results = pd.DataFrame([v[0]+(v[1],) for v in entity.iteritems()])
        else:
            results = pd.DataFrame(entity.iteritems())
    else:
        # create DataFrame
        if entity.dim() > 1:
            # concatenate index tuples with value if entity has
            # multidimensional indices v[0]
            results = pd.DataFrame(
                [v[0]+(v[1].value,) for v in entity.iteritems()])
        else:
            # otherwise, create tuple from scalar index v[0]
            results = pd.DataFrame(
                [(v[0], v[1].value) for v in entity.iteritems()])

    # check for duplicate onset names and append one to several "_" to make
    # them unique, e.g. ['sit', 'sit', 'com'] becomes ['sit', 'sit_', 'com']
    for k, label in enumerate(labels):
        if label in labels[:k]:
            labels[k] = labels[k] + "_"

    if not results.empty:
        # name columns according to labels + entity name
        results.columns = labels + [name]
        results.set_index(labels, inplace=True)

    return results

def get_entities(instance, names):
    """ Return one DataFrame with entities in columns and a common index.

    Works only on entities that share a common domain (set or set_tuple), which
    is used as index of the returned DataFrame.

    Args:
        instance: a Pyomo ConcreteModel instance
        names: list of entity names (as returned by list_entities)

    Returns:
        a Pandas DataFrame with entities as columns and domains as index
    """

    df = pd.DataFrame()
    for name in names:
        other = get_entity(instance, name)

        if df.empty:
            df = other
        else:
            index_names_before = df.index.names

            df = df.join(other, how='outer')

            if index_names_before != df.index.names:
                df.index.names = index_names_before

    return df

def list_entities(instance, entity_type):
    """ Return list of sets, params, variables, constraints or objectives

    Args:
        instance: a Pyomo ConcreteModel object
        entity_type: "set", "par", "var", "con" or "obj"

    Returns:
        DataFrame of entities

    Example:
        >>> data = read_excel('mimo-example.xlsx')
        >>> model = create_model(data, range(1,25))
        >>> list_entities(model, 'obj')  #doctest: +NORMALIZE_WHITESPACE
                                         Description Domain
        Name
        obj   minimize(cost = sum of all cost types)     []

    """

    # helper function to discern entities by type
    def filter_by_type(entity, entity_type):
        if entity_type == 'set':
            return isinstance(entity, pyomo.Set) and not entity.virtual
        elif entity_type == 'par':
            return isinstance(entity, pyomo.Param)
        elif entity_type == 'var':
            return isinstance(entity, pyomo.Var)
        elif entity_type == 'con':
            return isinstance(entity, pyomo.Constraint)
        elif entity_type == 'obj':
            return isinstance(entity, pyomo.Objective)
        else:
            raise ValueError("Unknown entity_type '{}'".format(entity_type))

    # iterate through all model components and keep only
    iter_entities = instance.__dict__.iteritems()
    entities = sorted(
        (name, entity.doc, _get_onset_names(entity))
        for (name, entity) in iter_entities
        if filter_by_type(entity, entity_type))

    # if something was found, wrap tuples in DataFrame, otherwise return empty
    if entities:
        entities = pd.DataFrame(entities,
                                columns=['Name', 'Description', 'Domain'])
        entities.set_index('Name', inplace=True)
    else:
        entities = pd.DataFrame()
    return entities

def _get_onset_names(entity):
    """
        Example:
            >>> data = read_excel('mimo-example.xlsx')
            >>> model = create_model(data, range(1,25))
            >>> _get_onset_names(model.e_co_stock)
            ['t', 'sit', 'com', 'com_type']
    """
    # get column titles for entities from domain set names
    labels = []

    if isinstance(entity, pyomo.Set):
        if entity.dimen > 1:
            # N-dimensional set tuples, possibly with nested set tuples within
            if entity.domain:
                domains = entity.domain.set_tuple
            else:
                domains = entity.set_tuple

            for domain_set in domains:
                labels.extend(_get_onset_names(domain_set))

        elif entity.dimen == 1:
            if entity.domain:
                # 1D subset; add domain name
                labels.append(entity.domain.name)
            else:
                # unrestricted set; add entity name
                labels.append(entity.name)
        else:
            # no domain, so no labels needed
            pass

    elif isinstance(entity, (pyomo.Param, pyomo.Var, pyomo.Constraint,
                    pyomo.Objective)):
        if entity.dim() > 0 and entity._index:
            labels = _get_onset_names(entity._index)
        else:
            # zero dimensions, so no onset labels
            pass

    else:
        raise ValueError("Unknown entity type!")

    return labels

def get_constants(instance):
    """Return summary DataFrames for important variables

    Usage:
        costs, cpro, ctra, csto = get_constants(instance)

    Args:
        instance: a urbs model instance

    Returns:
        (costs, cpro, ctra, csto) tuple

    Example:
        >>> import coopr.environ
        >>> from coopr.opt.base import SolverFactory
        >>> data = read_excel('mimo-example.xlsx')
        >>> model = create_model(data, range(1,25))
        >>> prob = model.create()
        >>> optim = SolverFactory('glpk')
        >>> result = optim.solve(prob)
        >>> prob.load(result)
        True
        >>> cap_pro = get_constants(prob)[1]['Total']
        >>> cap_pro.xs('Wind park', level='Process').apply(int)
        Site
        Mid      13000
        North    27271
        South     2674
        Name: Total, dtype: int64
    """
    costs = get_entity(instance, 'costs')
    cpro = get_entities(instance, ['cap_pro', 'cap_pro_new'])
    ctra = get_entities(instance, ['cap_tra', 'cap_tra_new'])
    csto = get_entities(instance, ['cap_sto_c', 'cap_sto_c_new',
                                   'cap_sto_p', 'cap_sto_p_new'])

    # better labels and index names and return sorted
    if not cpro.empty:
        cpro.index.names = ['Site', 'Process']
        cpro.columns = ['Total', 'New']
        cpro.sortlevel(inplace=True)
    if not ctra.empty:
        ctra.index.names = ['Site In', 'Site Out', 'Transmission', 'Commodity']
        ctra.columns = ['Total', 'New']
        ctra.sortlevel(inplace=True)
    if not csto.empty:
        csto.columns = ['C Total', 'C New', 'P Total', 'P New']
        csto.sortlevel(inplace=True)

    return costs, cpro, ctra, csto

def get_timeseries(instance, com, sit, timesteps=None):
    """Return DataFrames of all timeseries referring to given commodity

    Usage:
        create, consume, store, imp, exp = get_timeseries(instance, co,
                                                          sit, timesteps)

    Args:
        instance: a urbs model instance
        com: a commodity
        sit: a site
        timesteps: optional list of timesteps, defaults: all modelled timesteps

    Returns:
        a (created, consumed, storage, imported, exported) tuple of DataFrames
        timeseries. These are:

        * created: timeseries of commodity creation, including stock source
        * consumed: timeseries of commodity consumption, including demand
        * storage: timeseries of commodity storage (level, stored, retrieved)
        * imported: timeseries of commodity import (by site)
        * exported: timeseries of commodity export (by site)
    """
    if timesteps is None:
        # default to all simulated timesteps
        timesteps = sorted(get_entity(instance, 'tm').index)

    # DEMAND
    # default to zeros if commodity has no demand, get timeseries
    try:
        demand = instance.demand.loc[timesteps][sit, com]
    except KeyError:
        demand = pd.Series(0, index=timesteps)
    demand.name = 'Demand'

    # STOCK
    eco = get_entity(instance, 'e_co_stock')['e_co_stock'].unstack()['Stock']
    eco = eco.xs(sit, level='sit').unstack().fillna(0)
    try:
        stock = eco.loc[timesteps][com]
    except KeyError:
        stock = pd.Series(0, index=timesteps)
    stock.name = 'Stock'

    # PROCESS
    # select all entries of created and consumed desired commodity com and site
    # sit. Keep only entries with non-zero values and unstack process column.
    # Finally, slice to the desired timesteps.
    epro = get_entities(instance, ['e_pro_in', 'e_pro_out'])
    epro = epro.xs(sit, level='sit').xs(com, level='com')
    try:
        created = epro[epro['e_pro_out'] > 0]['e_pro_out'].unstack(level='pro')
        created = created.loc[timesteps].fillna(0)
    except KeyError:
        created = pd.DataFrame(index=timesteps)

    try:
        consumed = epro[epro['e_pro_in'] > 0]['e_pro_in'].unstack(level='pro')
        consumed = consumed.loc[timesteps].fillna(0)
    except KeyError:
        consumed = pd.DataFrame(index=timesteps)

    # TRANSMISSION
    etra = get_entities(instance, ['e_tra_in', 'e_tra_out'])
    try:
        etra.index.names = ['tm', 'sitin', 'sitout', 'tra', 'com']
        etra = etra.groupby(level=['tm', 'sitin', 'sitout', 'com']).sum()
        etra = etra.xs(com, level='com')

        imported = etra.xs(sit, level='sitout')['e_tra_out'].unstack().fillna(0)
        exported = etra.xs(sit, level='sitin')['e_tra_in'].unstack().fillna(0)

    except (ValueError, KeyError):
        imported = pd.DataFrame(index=timesteps)
        exported = pd.DataFrame(index=timesteps)

    # STORAGE
    # group storage energies by commodity
    # select all entries with desired commodity co
    esto = get_entities(instance, ['e_sto_con', 'e_sto_in', 'e_sto_out'])
    esto = esto.groupby(level=['t', 'sit', 'com']).sum()
    try:
        esto = esto.xs(sit, level='sit')
        stored = esto.xs(com, level='com')
        stored = stored.loc[timesteps]
        stored.columns = ['Level', 'Stored', 'Retrieved']
    except (KeyError, ValueError):
        stored = pd.DataFrame(0, index=timesteps,
                              columns=['Level', 'Stored', 'Retrieved'])

    # show stock as created
    created = created.join(stock)

    # show demand as consumed
    consumed = consumed.join(demand)

    return created, consumed, stored, imported, exported

def summaries_timeseries(instance, commodities=None, sites=None):
    # initialize timeseries tableaus
    energies = []
    timeseries = {}

    # collect timeseries data
    for co in commodities:
        for sit in sites:
            created, consumed, stored, imported, exported = get_timeseries(
                instance, co, sit)

            overprod = pd.DataFrame(
                columns=['Overproduction'],
                data=created.sum(axis=1) - consumed.sum(axis=1) +
                imported.sum(axis=1) - exported.sum(axis=1) +
                stored['Retrieved'] - stored['Stored'])

            tableau = pd.concat(
                [created, consumed, stored, imported, exported, overprod],
                axis=1,
                keys=['Created', 'Consumed', 'Storage',
                      'Import from', 'Export to', 'Balance'])
            timeseries[(co, sit)] = tableau.copy()

            # timeseries sums
            sums = pd.concat([created.sum(),
                              consumed.sum(),
                              stored.sum().drop('Level'),
                              imported.sum(),
                              exported.sum(),
                              overprod.sum()], axis=0,
                             keys=['Created', 'Consumed', 'Storage',
                             'Import', 'Export', 'Balance'])
            energies.append(sums.to_frame("{}.{}".format(co, sit)))

    # concatenate energy sums
    energy_sums = pd.concat(energies, axis=1).fillna(0)
    return energy_sums, timeseries

def to_color(obj=None):
    """Assign a deterministic pseudo-random color to argument.

    If COLORS[obj] is set, return that. Otherwise, create a random color from
    the hash(obj) representation string. For strings, this value depends only
    on the string content, so that same strings always yield the same color.

    Args:
        obj: any hashable object

    Returns:
        a (r, g, b) color tuple if COLORS[obj] is set, otherwise a hexstring
    """
    if obj is None:
        obj = random()
    try:
        color = tuple(rgb/255.0 for rgb in conf.COLORS[obj])
    except KeyError:
        # random deterministic color
        color = "#{:06x}".format(abs(hash(obj)))[:7]
    return color


# endregion


