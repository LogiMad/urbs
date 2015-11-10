import coopr.pyomo as pyomo
import math
from datetime import datetime
from pandas import isnull, notnull

def create_model(data, timesteps=None, dt=1):
    """Create a pyomo ConcreteModel URBS object from given input data.

    Args:
        data: a dict of 6 DataFrames with the keys 'commodity', 'process',
            'transmission', 'storage', 'demand' and 'supim'.
        timesteps: optional list of timesteps, default: demand timeseries
        dt: timestep duration in hours (default: 1)

    Returns:
        a pyomo ConcreteModel object
    """
    m = pyomo.ConcreteModel()
    m.name = 'URBS'
    m.created = datetime.now().strftime('%Y%m%dT%H%M%S')

    # Optional
    if not timesteps:
        timesteps = data['demand'].index.tolist()

    # region Preparations
    # ============
    # Data import. Syntax to access a value within equation definitions looks
    # like this:
    #
    #     m.storage.loc[site, storage, commodity][attribute]
    #
    m.commodity = data['commodity']
    m.process = data['process']
    m.process_commodity = data['process_commodity']
    m.transmission = data['transmission']
    m.storage = data['storage']
    m.demand = data['demand']
    m.supim = data['supim']
    m.timesteps = timesteps

    # process input/output ratios
    m.r_in = m.process_commodity.xs('In', level='Direction')['ratio']
    m.r_out = m.process_commodity.xs('Out', level='Direction')['ratio']
    m.pw_dmn_in = m.process_commodity.xs('In', level='Direction')['pw_domain']
    m.pw_rng_in = m.process_commodity.xs('In', level='Direction')['pw_range']
    m.pw_dmn_out = m.process_commodity.xs('Out', level='Direction')['pw_domain']
    m.pw_rng_out = m.process_commodity.xs('Out', level='Direction')['pw_range']
    # print(m.process_commodity)
    print(m.pw_dmn_out)
    # endregion

    # region Sets
    # ====
    # Syntax: m.{name} = Set({domain}, initialize={values})
    # where name: set name
    #       domain: set domain for tuple sets, a cartesian set product
    #       values: set values, a list or array of element tuples

    # generate ordered time step sets
    m.t = pyomo.Set(
        initialize=m.timesteps,
        ordered=True,
        doc='Set of timesteps')

    # modelled (i.e. excluding init time step for storage) time steps
    m.tm = pyomo.Set(
        within=m.t,
        initialize=m.timesteps[1:],
        ordered=True,
        doc='Set of modelled timesteps')

    # site (e.g. north, middle, south...)
    m.sit = pyomo.Set(
        initialize=m.commodity.index.get_level_values('Site').unique(),
        doc='Set of sites')

    # commodity (e.g. solar, wind, coal...)
    m.com = pyomo.Set(
        initialize=m.commodity.index.get_level_values('Commodity').unique(),
        doc='Set of commodities')

    # commodity type (i.e. SupIm, Demand, Stock, Env)
    m.com_type = pyomo.Set(
        initialize=m.commodity.index.get_level_values('Type').unique(),
        doc='Set of commodity types')

    # process (e.g. Wind turbine, Gas plant, Photovoltaics...)
    m.pro = pyomo.Set(
        initialize=m.process.index.get_level_values('Process').unique(),
        doc='Set of conversion processes')

    # tranmission (e.g. hvac, hvdc, pipeline...)
    m.tra = pyomo.Set(
        initialize=m.transmission.index.get_level_values('Transmission').unique(),
        doc='Set of tranmission technologies')

    # storage (e.g. hydrogen, pump storage)
    m.sto = pyomo.Set(
        initialize=m.storage.index.get_level_values('Storage').unique(),
        doc='Set of storage technologies')

    # cost_type
    m.cost_type = pyomo.Set(
        initialize=['Inv', 'Fix', 'Var', 'Fuel'],
        doc='Set of cost types (hard-coded)')

    # tuple sets
    m.com_tuples = pyomo.Set(
        within=m.sit*m.com*m.com_type,
        initialize=m.commodity.index,
        doc='Combinations of defined commodities, e.g. (Mid,Elec,Demand)')
    m.pro_tuples = pyomo.Set(
        within=m.sit*m.pro,
        initialize=m.process.index,
        doc='Combinations of possible processes, e.g. (North,Coal plant)')
    m.tra_tuples = pyomo.Set(
        within=m.sit*m.sit*m.tra*m.com,
        initialize=m.transmission.index,
        doc='Combinations of possible transmission, e.g. (South,Mid,hvac,Elec)')
    m.sto_tuples = pyomo.Set(
        within=m.sit*m.sto*m.com,
        initialize=m.storage.index,
        doc='Combinations of possible storage by site, e.g. (Mid,Bat,Elec)')

    # process input/output
    m.pro_input_tuples = pyomo.Set(
        within=m.sit*m.pro*m.com,
        initialize=[(site, process, commodity)
                    for (site, process) in m.pro_tuples
                    for (pro, commodity) in m.r_in.index
                    if process == pro],
        doc='Commodities consumed by process by site, e.g. (Mid,PV,Solar)')
    m.pro_output_tuples = pyomo.Set(
        within=m.sit*m.pro*m.com,
        initialize=[(site, process, commodity)
                    for (site, process) in m.pro_tuples
                    for (pro, commodity) in m.r_out.index
                    if process == pro],
        doc='Commodities produced by process by site, e.g. (Mid,PV,Elec)')

    # commodity type subsets
    m.com_supim = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'SupIm'),
        doc='Commodities that have intermittent (timeseries) input')
    m.com_stock = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'Stock'),
        doc='Commodities that can be purchased at some site(s)')
    m.com_demand = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'Demand'),
        doc='Commodities that have a demand (implies timeseries)')
    m.com_env = pyomo.Set(
        within=m.com,
        initialize=commodity_subset(m.com_tuples, 'Env'),
        doc='Commodities that (might) have a maximum creation limit')
    # endregion

    # region Parameters
    # weight = length of year (hours) / length of simulation (hours)
    # weight scales costs and emissions from length of simulation to a full
    # year, making comparisons among cost types (invest is annualized, fixed
    # costs are annual by default, variable costs are scaled by weight) and
    # among different simulation durations meaningful.
    m.weight = pyomo.Param(
        initialize=float(8760) / (len(m.t) * dt),
        doc='Pre-factor for variable costs and emissions for an annual result')

    # dt = spacing between timesteps. Required for storage equation that
    # converts between energy (storage content, e_sto_con) and power (all other
    # quantities that start with "e_")
    m.dt = pyomo.Param(
        initialize=dt,
        doc='Time step duration (in hours), default: 1')
    # endregion

    # region Variables
    # costs
    m.costs = pyomo.Var(
        m.cost_type,
        within=pyomo.NonNegativeReals,
        doc='Costs by type (EUR/a)')

    # commodity
    m.e_co_stock = pyomo.Var(
        m.tm, m.com_tuples,
        within=pyomo.NonNegativeReals,
        doc='Use of stock commodity source (MW) per timestep')

    # process
    m.cap_pro = pyomo.Var(
        m.pro_tuples,
        within=pyomo.NonNegativeReals,
        doc='Total process capacity (MW)')
    m.cap_pro_new = pyomo.Var(
        m.pro_tuples,
        within=pyomo.NonNegativeReals,
        doc='New process capacity (MW)')
    m.tau_pro = pyomo.Var(
        m.tm, m.pro_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow (MW) through process')
    m.e_pro_in = pyomo.Var(
        m.tm, m.pro_tuples, m.com,
        within=pyomo.NonNegativeReals,
        doc='Power flow of commodity into process (MW) per timestep')
    m.e_pro_out = pyomo.Var(
        m.tm, m.pro_tuples, m.com,
        within=pyomo.NonNegativeReals,
        doc='Power flow out of process (MW) per timestep')
    m.cf_pro = pyomo.Var(
        m.tm, m.pro_tuples,
        within=pyomo.NonNegativeReals,
        bounds=(0.0, 1.0),
        doc='Capacity factor of process at timestep')

    # transmission
    m.cap_tra = pyomo.Var(
        m.tra_tuples,
        within=pyomo.NonNegativeReals,
        doc='Total transmission capacity (MW)')
    m.cap_tra_new = pyomo.Var(
        m.tra_tuples,
        within=pyomo.NonNegativeReals,
        doc='New transmission capacity (MW)')
    m.e_tra_in = pyomo.Var(
        m.tm, m.tra_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow into transmission line (MW) per timestep')
    m.e_tra_out = pyomo.Var(
        m.tm, m.tra_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow out of transmission line (MW) per timestep')

    # storage
    m.cap_sto_c = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Total storage size (MWh)')
    m.cap_sto_c_new = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='New storage size (MWh)')
    m.cap_sto_p = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Total storage power (MW)')
    m.cap_sto_p_new = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='New  storage power (MW)')
    m.e_sto_in = pyomo.Var(
        m.tm, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow into storage (MW) per timestep')
    m.e_sto_out = pyomo.Var(
        m.tm, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow out of storage (MW) per timestep')
    m.e_sto_con = pyomo.Var(
        m.t, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Energy content of storage (MWh) in timestep')
    # endregion

    # region Equation declarations
    # equation bodies are defined in separate functions, refered to here by
    # their name in the "rule" keyword.

    # region Constraints declaration
    # region commodity
    m.res_vertex = pyomo.Constraint(
        m.tm, m.com_tuples,
        rule=res_vertex_rule,
        doc='storage + transmission + process + source >= demand')
    m.res_stock_step = pyomo.Constraint(
        m.tm, m.com_tuples,
        rule=res_stock_step_rule,
        doc='stock commodity input per step <= commodity.maxperstep')
    m.res_stock_total = pyomo.Constraint(
        m.com_tuples,
        rule=res_stock_total_rule,
        doc='total stock commodity input <= commodity.max')
    m.res_env_step = pyomo.Constraint(
        m.tm, m.com_tuples,
        rule=res_env_step_rule,
        doc='environmental output per step <= commodity.maxperstep')
    m.res_env_total = pyomo.Constraint(
        m.com_tuples,
        rule=res_env_total_rule,
        doc='total environmental commodity output <= commodity.max')
    # endregion

    # region process
    m.def_process_capacity = pyomo.Constraint(
        m.pro_tuples,
        rule=def_process_capacity_rule,
        doc='total process capacity = inst-cap + new capacity')
    m.def_process_input = pyomo.Constraint(
        m.tm, m.pro_input_tuples,
        rule=def_process_input_rule,
        doc='process input = process throughput * input ratio')
    m.def_process_output = pyomo.Constraint(
        m.tm, m.pro_output_tuples,
        rule=def_process_output_rule,
        doc='process output = process throughput * output ratio')
    m.def_intermittent_supply = pyomo.Constraint(
        m.tm, m.pro_input_tuples,
        rule=def_intermittent_supply_rule,
        doc='process output = process capacity * supim timeseries')
    m.res_process_throughput_by_capacity = pyomo.Constraint(
        m.tm, m.pro_tuples,
        rule=res_process_throughput_by_capacity_rule,
        doc='process throughput <= total process capacity')
    m.res_process_capacity = pyomo.Constraint(
        m.pro_tuples,
        rule=res_process_capacity_rule,
        doc='process.cap-lo <= total process capacity <= process.cap-up')
    m.res_process_ramp_up = pyomo.Constraint(
        m.tm, m.pro_output_tuples,
        rule=res_process_ramp_up_rule,
        doc='process output increase  <= relative ramp-up rate * time step duration')
    m.res_process_ramp_down = pyomo.Constraint(
        m.tm, m.pro_output_tuples,
        rule=res_process_ramp_down_rule,
        doc='process output decrease  <= relative ramp-down rate * time step duration')
    m.def_process_output_pw = pyomo.Piecewise(
        m.tm, m.pro_output_tuples,
        m.e_pro_out, m.cf_pro,  # range and domain variables
        pw_pts=m.pw_dmn_out[m.pro, m.com],
        pw_constr_type='EQ',
        pw_repn='SOS2',
        f_rule=def_process_output_pw_rule,
        doc='process throughput = f(capacity factor), "f" is a linear piecewise function')
    # endregion

    # region transmission
    m.def_transmission_capacity = pyomo.Constraint(
        m.tra_tuples,
        rule=def_transmission_capacity_rule,
        doc='total transmission capacity = inst-cap + new capacity')
    m.def_transmission_output = pyomo.Constraint(
        m.tm, m.tra_tuples,
        rule=def_transmission_output_rule,
        doc='transmission output = transmission input * efficiency')
    m.res_transmission_input_by_capacity = pyomo.Constraint(
        m.tm, m.tra_tuples,
        rule=res_transmission_input_by_capacity_rule,
        doc='transmission input <= total transmission capacity')
    m.res_transmission_capacity = pyomo.Constraint(
        m.tra_tuples,
        rule=res_transmission_capacity_rule,
        doc='transmission.cap-lo <= total transmission capacity <= '
            'transmission.cap-up')
    m.res_transmission_symmetry = pyomo.Constraint(
        m.tra_tuples,
        rule=res_transmission_symmetry_rule,
        doc='total transmission capacity must be symmetric in both directions')
    # endregion

    # region storage
    m.def_storage_state = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=def_storage_state_rule,
        doc='storage[t] = storage[t-1] + input - output')
    m.def_storage_power = pyomo.Constraint(
        m.sto_tuples,
        rule=def_storage_power_rule,
        doc='storage power = inst-cap + new power')
    m.def_storage_capacity = pyomo.Constraint(
        m.sto_tuples,
        rule=def_storage_capacity_rule,
        doc='storage capacity = inst-cap + new capacity')
    m.res_storage_input_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_input_by_power_rule,
        doc='storage input <= storage power')
    m.res_storage_output_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_output_by_power_rule,
        doc='storage output <= storage power')
    m.res_storage_state_by_capacity = pyomo.Constraint(
        m.t, m.sto_tuples,
        rule=res_storage_state_by_capacity_rule,
        doc='storage content <= storage capacity')
    m.res_storage_power = pyomo.Constraint(
        m.sto_tuples,
        rule=res_storage_power_rule,
        doc='storage.cap-lo-p <= storage power <= storage.cap-up-p')
    m.res_storage_capacity = pyomo.Constraint(
        m.sto_tuples,
        rule=res_storage_capacity_rule,
        doc='storage.cap-lo-c <= storage capacity <= storage.cap-up-c')
    m.res_initial_and_final_storage_state = pyomo.Constraint(
        m.t, m.sto_tuples,
        rule=res_initial_and_final_storage_state_rule,
        doc='storage content initial == and final >= storage.init * capacity')
    # endregion

    # region costs
    m.def_costs = pyomo.Constraint(
        m.cost_type,
        rule=def_costs_rule,
        doc='main cost function by cost type')
    # endregion

    # endregion

    # region Objective declaration
    m.obj = pyomo.Objective(
        rule=obj_rule,
        sense=pyomo.minimize,
        doc='minimize(cost = sum of all cost types)')
    # endregion

    # region possibly: add hack features
    if 'hacks' in data:
        m = add_hacks(m, data['hacks'])
    # endregion

    # endregion

    return m

# region Constraint Equations
# region commodity
def res_vertex_rule(m, tm, sit, com, com_type):
    """vertex equation: calculate balance for given commodity and site;

    contains implicit constraints for process activity, import/export and
    storage activity (calculated by function commodity_balance);
    contains implicit constraint for stock commodity source term
    """
    # environmental or supim commodities don't have this constraint (yet)
    if com in m.com_env:
        return pyomo.Constraint.Skip
    if com in m.com_supim:
        return pyomo.Constraint.Skip

    # helper function commodity_balance calculates balance from input to
    # and output from processes, storage and transmission.
    # if power_surplus > 0: production/storage/imports create net positive
    #                       amount of commodity com
    # if power_surplus < 0: production/storage/exports consume a net
    #                       amount of the commodity com
    power_surplus = - commodity_balance(m, tm, sit, com)

    # if com is a stock commodity, the commodity source term e_co_stock
    # can supply a possibly negative power_surplus
    if com in m.com_stock:
        power_surplus += m.e_co_stock[tm, sit, com, com_type]

    # if com is a demand commodity, the power_surplus is reduced by the
    # demand value; no scaling by m.dt or m.weight is needed here, as this
    # constraint is about power (MW), not energy (MWh)
    if com in m.com_demand:
        try:
            power_surplus -= m.demand.loc[tm][sit, com]
        except KeyError:
            pass
    return power_surplus >= 0

def res_stock_step_rule(m, tm, sit, com, com_type):
    """stock commodity purchase == commodity consumption, according to
    commodity_balance of current (time step, site, commodity);
    limit stock commodity use per time step
    """
    if com not in m.com_stock:
        return pyomo.Constraint.Skip
    else:
        return (m.e_co_stock[tm, sit, com, com_type] <=
                m.commodity.loc[sit, com, com_type]['maxperstep'])

def res_stock_total_rule(m, sit, com, com_type):
    """limit stock commodity use in total (scaled to annual consumption, thanks
    to m.weight)
    """
    if com not in m.com_stock:
        return pyomo.Constraint.Skip
    else:
        # calculate total consumption of commodity com
        total_consumption = 0
        for tm in m.tm:
            total_consumption += (
                m.e_co_stock[tm, sit, com, com_type] * m.dt)
        total_consumption *= m.weight
        return (total_consumption <=
                m.commodity.loc[sit, com, com_type]['max'])

def res_env_step_rule(m, tm, sit, com, com_type):
    """environmental commodity creation == - commodity_balance of that commodity

    used for modelling emissions (e.g. CO2) or other end-of-pipe results of
    any process activity;
    limit environmental commodity output per time step
    """

    if com not in m.com_env:
        return pyomo.Constraint.Skip
    else:
        environmental_output = - commodity_balance(m, tm, sit, com)
        return (environmental_output <=
                m.commodity.loc[sit, com, com_type]['maxperstep'])

def res_env_total_rule(m, sit, com, com_type):
    """limit environmental commodity output in total (scaled to annual
    emissions, thanks to m.weight)
    """
    if com not in m.com_env:
        return pyomo.Constraint.Skip
    else:
        # calculate total creation of environmental commodity com
        env_output_sum = 0
        for tm in m.tm:
            env_output_sum += (- commodity_balance(m, tm, sit, com) * m.dt)
        env_output_sum *= m.weight
        return (env_output_sum <=
                m.commodity.loc[sit, com, com_type]['max'])
    return
# endregion

# region process
def def_process_capacity_rule(m, sit, pro):
    """process capacity == new capacity + existing capacity"""
    return (m.cap_pro[sit, pro] ==
            m.cap_pro_new[sit, pro] +
            m.process.loc[sit, pro]['inst-cap'])

def def_process_input_rule(m, tm, sit, pro, co):
    """process input power == process throughput * input ratio"""
    if m.r_in.loc[pro, co] == -1:
        return pyomo.Constraint.Skip
    else:
        return (m.e_pro_in[tm, sit, pro, co] ==
                m.tau_pro[tm, sit, pro] * m.r_in.loc[pro, co])

def def_process_output_rule(m, tm, sit, pro, co):
    """process output power = process throughput * output ratio"""
    if m.r_out.loc[pro, co] == -1:
        return pyomo.Constraint.Skip
    else:
        return (m.e_pro_out[tm, sit, pro, co] ==
                m.tau_pro[tm, sit, pro] * m.r_out.loc[pro, co])

def def_intermittent_supply_rule(m, tm, sit, pro, coin):
    """process input (for supim commodity) = process capacity * timeseries"""
    if coin in m.com_supim:
        return (m.e_pro_in[tm, sit, pro, coin] ==
                m.cap_pro[sit, pro] * m.supim.loc[tm][sit, coin])
    else:
        return pyomo.Constraint.Skip

# def def_process_capacity_factor_rule(m, tm, sit, pro):
#     """process capacity factor = process throughput / process capacity"""
#     return (m.cf_pro[tm, sit, pro] ==
#             m.tau_pro[tm, sit, pro] / m.cap_pro[sit, pro])

def def_process_capacity_factor_rule(m, tm, sit, pro):
    """process capacity factor = process output power / process capacity"""
    return (m.cf_pro[tm, sit, pro] ==
            m.e_pro_out[tm, sit, pro, co] / m.cap_pro[sit, pro])

def res_process_throughput_by_capacity_rule(m, tm, sit, pro):
    """process throughput <= process capacity"""
    return (m.tau_pro[tm, sit, pro] <= m.cap_pro[sit, pro])

def res_process_capacity_rule(m, sit, pro):
    """lower bound <= process capacity <= upper bound"""
    return (m.process.loc[sit, pro]['cap-lo'],
            m.cap_pro[sit, pro],
            m.process.loc[sit, pro]['cap-up'])

def res_process_ramp_up_rule(m, tm, sit, pro, co):
    """process output increase  <= process capacity * relative ramp-up rate * time step duration"""
    if isnull(m.process.loc[sit, pro]['ramp-up']):
        return pyomo.Constraint.Skip
    elif tm-1 in m.e_pro_out:
        return (m.e_pro_out[tm, sit, pro, co] - m.e_pro_out[tm-1, sit, pro, co] <=
                m.cap_pro[sit, pro] * m.process.loc[sit, pro]['ramp-up'] * 60 * m.dt)
    else:
        return pyomo.Constraint.Skip

def res_process_ramp_down_rule(m, tm, sit, pro, co):
    """process output decrease  <= process capacity * relative ramp-down rate * time step duration"""
    if isnull(m.process.loc[sit, pro]['ramp-down']):
        return pyomo.Constraint.Skip
    elif tm-1 in m.e_pro_out:
        return (m.e_pro_out[tm-1, sit, pro, co] - m.e_pro_out[tm, sit, pro, co] <=
                m.cap_pro[sit, pro] * m.process.loc[sit, pro]['ramp-down'] * 60 * m.dt)
    else:
        return pyomo.Constraint.Skip

def def_process_output_pw_rule(m, sit, pro, dmn_pts):
    """Return the range value associated with input domain value
        as input of the linear Piecewise function dictionary.
    """
    if isnull(m.process_commodity.loc[sit, pro]['pw_range']):
        return pyomo.Constraint.Skip
    else:
        return m.process_commodity.loc[sit, pro]['pw_range'][dmn_pts]


# endregion

# region transmission
def def_transmission_capacity_rule(m, sin, sout, tra, com):
    """transmission capacity == new capacity + existing capacity"""
    return (m.cap_tra[sin, sout, tra, com] ==
            m.cap_tra_new[sin, sout, tra, com] +
            m.transmission.loc[sin, sout, tra, com]['inst-cap'])

def def_transmission_output_rule(m, tm, sin, sout, tra, com):
    """transmission output == transmission input * efficiency"""
    return (m.e_tra_out[tm, sin, sout, tra, com] ==
            m.e_tra_in[tm, sin, sout, tra, com] *
            m.transmission.loc[sin, sout, tra, com]['eff'])

def res_transmission_input_by_capacity_rule(m, tm, sin, sout, tra, com):
    """transmission input <= transmission capacity"""
    return (m.e_tra_in[tm, sin, sout, tra, com] <=
            m.cap_tra[sin, sout, tra, com])

def res_transmission_capacity_rule(m, sin, sout, tra, com):
    """lower bound <= transmission capacity <= upper bound"""
    return (m.transmission.loc[sin, sout, tra, com]['cap-lo'],
            m.cap_tra[sin, sout, tra, com],
            m.transmission.loc[sin, sout, tra, com]['cap-up'])

def res_transmission_symmetry_rule(m, sin, sout, tra, com):
    """transmission capacity from A to B == transmission capacity from B to A"""
    return m.cap_tra[sin, sout, tra, com] == m.cap_tra[sout, sin, tra, com]
# endregion

# region storage
def def_storage_state_rule(m, t, sit, sto, com):
    """storage content in timestep [t] == storage content[t-1]
    + newly stored energy * input efficiency
    - retrieved energy / output efficiency
    """
    return (m.e_sto_con[t, sit, sto, com] ==
            m.e_sto_con[t-1, sit, sto, com] +
            m.e_sto_in[t, sit, sto, com] *
            m.storage.loc[sit, sto, com]['eff-in'] * m.dt -
            m.e_sto_out[t, sit, sto, com] /
            m.storage.loc[sit, sto, com]['eff-out'] * m.dt)

def def_storage_power_rule(m, sit, sto, com):
    """storage power == new storage power + existing storage power"""
    return (m.cap_sto_p[sit, sto, com] ==
            m.cap_sto_p_new[sit, sto, com] +
            m.storage.loc[sit, sto, com]['inst-cap-p'])

def def_storage_capacity_rule(m, sit, sto, com):
    """storage capacity == new storage capacity + existing storage capacity"""
    return (m.cap_sto_c[sit, sto, com] ==
            m.cap_sto_c_new[sit, sto, com] +
            m.storage.loc[sit, sto, com]['inst-cap-c'])

def res_storage_input_by_power_rule(m, t, sit, sto, com):
    """storage input <= storage power"""
    return m.e_sto_in[t, sit, sto, com] <= m.cap_sto_p[sit, sto, com]

def res_storage_output_by_power_rule(m, t, sit, sto, co):
    """storage output <= storage power"""
    return m.e_sto_out[t, sit, sto, co] <= m.cap_sto_p[sit, sto, co]

def res_storage_power_rule(m, sit, sto, com):
    """lower bound <= storage power <= upper bound"""
    return (m.storage.loc[sit, sto, com]['cap-lo-p'],
            m.cap_sto_p[sit, sto, com],
            m.storage.loc[sit, sto, com]['cap-up-p'])

def res_storage_capacity_rule(m, sit, sto, com):
    """lower bound <= storage capacity <= upper bound"""
    return (m.storage.loc[sit, sto, com]['cap-lo-c'],
            m.cap_sto_c[sit, sto, com],
            m.storage.loc[sit, sto, com]['cap-up-c'])

def res_initial_and_final_storage_state_rule(m, t, sit, sto, com):
    """initialization of storage content in first timestep t[1]
    forced minimun  storage content in final timestep t[len(m.t)]
    content[t=1] == storage capacity * fraction <= content[t=final]
    """
    if t == m.t[1]:  # first timestep (Pyomo uses 1-based indexing)
        return (m.e_sto_con[t, sit, sto, com] ==
                m.cap_sto_c[sit, sto, com] *
                m.storage.loc[sit, sto, com]['init'])
    elif t == m.t[len(m.t)]:  # last timestep
        return (m.e_sto_con[t, sit, sto, com] >=
                m.cap_sto_c[sit, sto, com] *
                m.storage.loc[sit, sto, com]['init'])
    else:
        return pyomo.Constraint.Skip

def res_storage_state_by_capacity_rule(m, t, sit, sto, com):
    """storage content <= storage capacity"""
    return m.e_sto_con[t, sit, sto, com] <= m.cap_sto_c[sit, sto, com]
# endregion
# endregion

# region Objective function(s)
def def_costs_rule(m, cost_type):
    """Calculate total costs by cost type.

    Sums up process activity and capacity expansions
    and sums them in the cost types that are specified in the set
    m.cost_type. To change or add cost types, add/change entries
    there and modify the if/elif cases in this function accordingly.

    Cost types are
      - Investment costs for process power, storage power and
        storage capacity. They are multiplied by the annuity
        factors.
      - Fixed costs for process power, storage power and storage
        capacity.
      - Variables costs for usage of processes, storage and transmission.
      - Fuel costs for stock commodity purchase.

    """
    if cost_type == 'Inv':
        return m.costs['Inv'] == \
            sum(m.cap_pro_new[p] *
                m.process.loc[p]['inv-cost'] *
                m.process.loc[p]['annuity-factor']
                for p in m.pro_tuples) + \
            sum(m.cap_tra_new[t] *
                m.transmission.loc[t]['inv-cost'] *
                m.transmission.loc[t]['annuity-factor']
                for t in m.tra_tuples) + \
            sum(m.cap_sto_p_new[s] *
                m.storage.loc[s]['inv-cost-p'] *
                m.storage.loc[s]['annuity-factor'] +
                m.cap_sto_c_new[s] *
                m.storage.loc[s]['inv-cost-c'] *
                m.storage.loc[s]['annuity-factor']
                for s in m.sto_tuples)

    elif cost_type == 'Fix':
        return m.costs['Fix'] == \
            sum(m.cap_pro[p] * m.process.loc[p]['fix-cost']
                for p in m.pro_tuples) + \
            sum(m.cap_tra[t] * m.transmission.loc[t]['fix-cost']
                for t in m.tra_tuples) + \
            sum(m.cap_sto_p[s] * m.storage.loc[s]['fix-cost-p'] +
                m.cap_sto_c[s] * m.storage.loc[s]['fix-cost-c']
                for s in m.sto_tuples)

    elif cost_type == 'Var':
        return m.costs['Var'] == \
            sum(m.tau_pro[(tm,) + p] * m.dt *
                m.process.loc[p]['var-cost'] *
                m.weight
                for tm in m.tm for p in m.pro_tuples) + \
            sum(m.e_tra_in[(tm,) + t] * m.dt *
                m.transmission.loc[t]['var-cost'] *
                m.weight
                for tm in m.tm for t in m.tra_tuples) + \
            sum(m.e_sto_con[(tm,) + s] *
                m.storage.loc[s]['var-cost-c'] * m.weight +
                (m.e_sto_in[(tm,) + s] + m.e_sto_out[(tm,) + s]) * m.dt *
                m.storage.loc[s]['var-cost-p'] * m.weight
                for tm in m.tm for s in m.sto_tuples)

    elif cost_type == 'Fuel':
        return m.costs['Fuel'] == sum(
            m.e_co_stock[(tm,) + c] * m.dt *
            m.commodity.loc[c]['price'] *
            m.weight
            for tm in m.tm for c in m.com_tuples
            if c[1] in m.com_stock)

    else:
        raise NotImplementedError("Unknown cost type.")

def obj_rule(m):
    return pyomo.summation(m.costs)

# Hacks

def add_hacks(model, hacks):
    """ add hackish features to model object

    This function is reserved for corner cases/features that still lack a
    satisfyingly general solution that could become part of create_model.
    Use hack features sparingly and think about how to incorporate into main
    model function before adding here. Otherwise, these features might become
    a maintenance burden.
    """

    # Store hack data
    model.hacks = hacks

    # Global CO2 limit
    try:
        global_co2_limit = hacks.loc['Global CO2 limit', 'Value']
    except KeyError:
        global_co2_limit = float('inf')

    # only add constraint if limit is finite
    if not math.isinf(global_co2_limit):
        model.res_global_co2_limit = pyomo.Constraint(
            rule=res_global_co2_limit_rule,
            doc='total co2 commodity output <= hacks.Glocal CO2 limit')

    return model

def res_global_co2_limit_rule(m):
    """total CO2 output <= Global CO2 limit"""
    co2_output_sum = 0
    for tm in m.tm:
        for sit in m.sit:
            # minus because negative commodity_balance represents creation of
            # that commodity.
            co2_output_sum += (- commodity_balance(m, tm, sit, 'CO2') * m.dt)

    # scaling to annual output (cf. definition of m.weight)
    co2_output_sum *= m.weight
    return (co2_output_sum <= m.hacks.loc['Global CO2 limit', 'Value'])
# endregion

# region Helper functions
def commodity_balance(m, tm, sit, com):
    """Calculate commodity balance at given timestep.

    For a given commodity co and timestep tm, calculate the balance of
    consumed (to process/storage/transmission, counts positive) and provided
    (from process/storage/transmission, counts negative) power. Used as helper
    function in create_model for constraints on demand and stock commodities.

    Args:
        m: the model object
        tm: the timestep
        site: the site
        com: the commodity

    Returns
        balance: net value of consumed (positive) or provided (negative) power
    """
    balance = 0
    for site, process in m.pro_tuples:
        if site == sit and com in m.r_in.loc[process].index:
            # usage as input for process increases balance
            balance += m.e_pro_in[(tm, site, process, com)]
        if site == sit and com in m.r_out.loc[process].index:
            # output from processes decreases balance
            balance -= m.e_pro_out[(tm, site, process, com)]
    for site_in, site_out, transmission, commodity in m.tra_tuples:
        # exports increase balance
        if site_in == sit and commodity == com:
            balance += m.e_tra_in[(tm, site_in, site_out, transmission, com)]
        # imports decrease balance
        if site_out == sit and commodity == com:
            balance -= m.e_tra_out[(tm, site_in, site_out, transmission, com)]
    for site, storage, commodity in m.sto_tuples:
        # usage as input for storage increases consumption
        # output from storage decreases consumption
        if site == sit and commodity == com:
            balance += m.e_sto_in[(tm, site, storage, com)]
            balance -= m.e_sto_out[(tm, site, storage, com)]
    return balance

def commodity_subset(com_tuples, type_name):
    """ Unique list of commodity names for given type.

    Args:
        com_tuples: a list of (site, commodity, commodity type) tuples
        type_name: a commodity type ('Stock', 'SupIm', 'Env' or 'Demand')

    Returns:
        The set (unique elements) of commodity names of the desired type
    """
    return set(com for sit, com, com_type in com_tuples
               if com_type == type_name)

# endregion

