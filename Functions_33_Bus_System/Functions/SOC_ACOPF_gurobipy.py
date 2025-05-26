import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB

# Function used to create incidence matrix 
def Incidence_matrices(num_nodes, num_lines, sending_end, receiving_end):
    A_plus = np.zeros((num_lines,num_nodes)) # A+ matrix (num_nodes x num_lines)
    for i in range(num_lines):
        A_plus[i,int(sending_end[i])] = 1
        A_plus[i,int(receiving_end[i])] = -1

    A_minus = np.zeros((num_lines,num_nodes)) # A- matrix (num_nodes x num_lines)
    for i in range(num_lines):
        A_minus[i,int(sending_end[i])] = 0
        A_minus[i,int(receiving_end[i])] = -1

    A_plus = np.transpose(A_plus)
    A_minus = np.transpose(A_minus)

    return A_plus, A_minus


def SOC_ACOPF_2D_alocation_single_scenario(baseMVA, NT, num_nodes, num_lines, Yp, sending_node, receiving_node, IndPV,
               R_l, X_l, B_l, Pd, Qd, pn_bound, qn_bound, v_bound, G_n, B_n, K_l, quad_cost, lin_cost, const_cost,
               ESS_soc0, ESS_cha_bound, ESS_dis_bound, ESS_soc_bound, 
               theta_n_min=-1, theta_n_max=-1, theta_l_min=-1, theta_l_max=-1, eta_dis = 1, eta_cha = 1):

    ######################################################
    """This model is very usefull since it allows to compute the optimal power flow for any grid topology."""
    ######################################################
    """Inputs"""
    # baseMVA : size(1) : Power base in MVA
    # NT : size(1) : Number of time steps
    # num_nodes : size(1) : Number of buses in the grid (NB)
    # num_lines : size(1) : Number of lines in the grid (NL)
    # Yp : size(1) : Planning period
    # sending_node : size(NB) : array with the sending node with number from 0 to N_bus-1 for line l 
    # sending_node : size(NL) : array with the receiving node with number from 0 to N_bus-1 for line l
    # IndPV : Indexes of PV : List where are the PV
    """All values in p.u. except the cost functions [CHF/MW]"""
    # quad_cost : Quadratic cost coefficient
    # lin_cost : Linear cost coefficient
    # const_cost : Constant cost coefficient
    # R_l, X_l, B_l, K_l : size(NL,NT) : r, x, b, ampacity limit for each line at each time step for each line
    # p_d, q_d, G_n, B_n : size(NB,NT) : active power, reactive power, g and b, at each time step for each node
    # pn_bound, qn_bound, v_bound : size(2,NB,NT) : active power, reactive power and voltage magnitude bounds at each time step for each node
    # quad_cost, lin_cost, const_cost : size(NB,NT) : cost of each buses.
    # ESS_soc0, ESS_cha_bound, ESS_dis_bound, ESS_soc_bound : size(2,NB,NT) : ESS initial state and bounds for rating and capacity
    ######################################################
    """Output"""
    # problem : size(1) : Total cost of operation
    # p_n, q_n, v_n : size(NB,NT) : active power injection, reactive power injection and voltage magnitude at each time step for each node
    # p_sl, q_sl, p_ol, q_ol, K_ol : size(NL,NT) : active and reactive power flow on lines, active and reactive power flow losses on lines, ampacity losses
    # theta_n, theta_l : size(NB or NL,NT) : angles on buses and lines
    # ESS_soc, ESS_cha, ESS_dis, q_ESS : size(NB,NT) : ESS operation variables 
    # lambda_, mu_ : size(NB,NT) : Dual values of ESS constraints on rating and capacity
    ######################################################
    
    """Initialisation"""
    A_plus, A_minus = Incidence_matrices(num_nodes, num_lines, sending_node, receiving_node)

    # Unfolding nodes data
    p_n_min = pn_bound[0]  # Minimum active power generation at each node
    p_n_max = pn_bound[1]  # Maximum active power generation at each node
    q_n_min = qn_bound[0]  # Minimum reactive power generation at each node
    q_n_max = qn_bound[1]  # Maximum reactive power generation at each node
    V_min = v_bound[0]**2  # Minimum voltage squared at each node
    V_max = v_bound[1]**2  # Maximum voltage squared at each node
    ESS_cha_min = ESS_cha_bound[0] # Minimum charging rate at each node 
    ESS_cha_max = ESS_cha_bound[1] # Maximum charging rate at each node
    ESS_dis_min = ESS_dis_bound[0] # Minimum discharging rate at each node 
    ESS_dis_max = ESS_dis_bound[1] # Maximum discharging rate at each node
    ESS_soc_min = ESS_soc_bound[0] # Minimum state of charge at each node 
    ESS_soc_max = ESS_soc_bound[1] # Maximum state of charge at each node

    # theta_n min and max
    if theta_n_min == -1 : theta_n_min = - np.pi / 2 * np.ones((num_nodes,NT))  # Minimum bus angle 
    if theta_n_max == -1 : theta_n_max = np.pi / 2 * np.ones((num_nodes,NT))  # Maximum bus angle

    # theta_l min and max
    if theta_l_min == -1 : theta_l_min = - np.pi / 2 * np.ones((num_lines,NT))  # Minimum line angle (from relaxation assumption)
    if theta_l_max == -1 : theta_l_max = np.pi / 2 * np.ones((num_lines,NT))  # Maximum line angle (from relaxation assumption)

    # for q_ESS
    lin = 2 # from data
    xx = np.linspace(0, 1/2 * np.pi, lin + 1)
    slope = np.zeros(lin)
    offset = np.zeros(lin)
    for i in range(lin):
        slope[i]=(np.sin(xx[i+1])-np.sin(xx[i]))/(np.cos(xx[i+1])-np.cos(xx[i]))
        offset[i]=(np.sin(xx[i])*np.cos(xx[i+1])-np.sin(xx[i+1])*np.cos(xx[i]))/(np.cos(xx[i+1])-np.cos(xx[i]))

    ######################################################
    """Create Gurobi Model"""
    model = gp.Model("SOC_ACOPF")
    
    ######################################################
    """Variables"""
    # Active power at node n
    # Active power bounds (1n)
    p_n = model.addVars(num_nodes, NT, lb=p_n_min, ub=p_n_max, name="p_n")
    p_curtailment = model.addVars(num_nodes, NT, lb=0, name="p_curtailment")

    p_imp = model.addVars(NT, lb=0, name="p_imp")
    p_exp = model.addVars(NT, lb=0, name="p_exp")
    
    # Reactive power at node n
    # Reactive power bounds (1o)
    q_n = model.addVars(num_nodes, NT, lb=q_n_min, ub=q_n_max, name="q_n")
    
    # Voltage magnitude squared at node n
    # Voltage Magnitude bounds (1k)
    V_n = model.addVars(num_nodes, NT, lb=V_min, ub=V_max, name="V_n")
    
    # Voltage angles at node n
    # Node angle bounds (1m)
    theta_n = model.addVars(num_nodes, NT, lb=theta_n_min, ub=theta_n_max, name="theta_n")  

    # Power flow variables
    p_sl = model.addVars(num_lines, NT, lb=-GRB.INFINITY, name="p_sl")
    q_sl = model.addVars(num_lines, NT, lb=-GRB.INFINITY, name="q_sl")
    p_ol = model.addVars(num_lines, NT, lb=-GRB.INFINITY, name="p_ol")
    q_ol = model.addVars(num_lines, NT, lb=-GRB.INFINITY, name="q_ol")
    K_ol = model.addVars(num_lines, NT, lb=-GRB.INFINITY, name="K_ol")
    # Line angle bounds (1l):
    theta_l = model.addVars(num_lines, NT, lb=theta_l_min, ub=theta_l_max, name="theta_l")

    # ESS variables
    # ESS charging and discharging rate lower bounds
    ESS_cha = model.addVars(num_nodes, NT, lb=ESS_cha_min, name="ESS_cha")
    ESS_dis = model.addVars(num_nodes, NT, lb=ESS_dis_min, name="ESS_dis")
    # ESS SOC lower bounds
    ESS_soc = model.addVars(num_nodes, NT, lb=ESS_soc_min, name="ESS_soc")
    q_ESS = model.addVars(num_nodes, NT, lb=-GRB.INFINITY, name="q_ESS")
    abs_diff = model.addVars(num_nodes, NT, lb=0, name="abs_diff")
    
    # Variables for allocation
    Cmax = model.addVars(num_nodes, NT, lb=-GRB.INFINITY, name="Cmax")
    Rmax = model.addVars(num_nodes, NT, lb=-GRB.INFINITY, name="Rmax")

    t_cone = model.addVars(num_lines, NT, lb=0, name="t_cone")
    rhs_sq = model.addVars(num_lines, NT, lb=0, name="rhs_sq")

    # Create the Incidence Matrices used in the sending end and receiving end voltage
    Inc_sending_cvx = np.zeros((num_lines, num_nodes))
    Inc_receiving_cvx = np.zeros((num_lines, num_nodes))
    for l in range(num_lines):
        Inc_sending_cvx[l, sending_node[l]] = 1
        Inc_receiving_cvx[l, receiving_node[l]] = 1

    ######################################################
    """Constraints"""

    # p_slack = p_n[0, :]
    # p_imp = cp.pos(p_slack)
    # p_exp = cp.pos(-p_slack)
    for time in range(NT):
        model.addConstr(p_n[0, time] == p_imp[time] - p_exp[time], name=f"decompose_{time}")

    ### Bus constraints ###

    # p_n + p_curtailment = p_n_max
    for time in range(NT):
        if len(IndPV[time]) > 0:
            for idx_pv in IndPV[time]:
                model.addConstr(p_n[idx_pv, time] + p_curtailment[idx_pv, time] == p_n_max[idx_pv, time])

    Rmax_constr = {}
    Cmax_constr = {}
    # ESS constraints
    for n in range(num_nodes):
        for time in range(NT):
            # Rating and capacity constraints
            Cmax_constr[n, time] = model.addConstr(Cmax[n, time] == ESS_soc_max[n, time])
            Rmax_constr[n, time] = model.addConstr(Rmax[n, time] == ESS_cha_max[n, time])
            
            # ESS charging and discharging rate upper bounds
            model.addConstr(ESS_cha[n, time] <= Rmax[n, time])
            model.addConstr(ESS_dis[n, time] <= Rmax[n, time])
            
            # ESS SOC upper bounds
            model.addConstr(ESS_soc[n, time] <= Cmax[n, time])
            
            # ESS reactive power computation and bounds
            for i in range(lin):
                model.addConstr(q_ESS[n, time] <= slope[i] * (ESS_cha[n, time] - ESS_dis[n, time]) + offset[i] * Rmax[n, time])
                model.addConstr(q_ESS[n, time] <= -slope[i] * (ESS_cha[n, time] - ESS_dis[n, time]) + offset[i] * Rmax[n, time])
                model.addConstr(q_ESS[n, time] >= slope[i] * (ESS_cha[n, time] - ESS_dis[n, time]) - offset[i] * Rmax[n, time])
                model.addConstr(q_ESS[n, time] >= -slope[i] * (ESS_cha[n, time] - ESS_dis[n, time]) - offset[i] * Rmax[n, time])

    # ESS time linking constraints
    for n in range(num_nodes):
        # Linking time steps for ESS (excluding the last time step)
        for time in range(NT-1):
            model.addConstr(ESS_soc[n, time+1] == ESS_soc[n, time] + ESS_cha[n, time] - ESS_dis[n, time])
        model.addConstr(ESS_soc0[n] == ESS_soc[n, NT-1] + ESS_cha[n, NT-1] - ESS_dis[n, NT-1])

        # Initializing ESS SOC for the first time step
        model.addConstr(ESS_soc[n, 0] == ESS_soc0[n])
        model.addConstr(ESS_soc[n, NT-1] == ESS_soc0[n])
        model.addConstr(gp.quicksum(ESS_cha[n, time] for time in range(NT)) == 
                       gp.quicksum(ESS_dis[n, time] for time in range(NT)))
        
    # Battery aging constraints --> battery has to do at maximum 1.1 cycles per day
    for n in range(num_nodes):
        for time in range(NT):
            diff = ESS_cha[n, time] - ESS_dis[n, time]
            model.addConstr(abs_diff[n, time] >= diff)
            model.addConstr(abs_diff[n, time] >= -diff)       
        model.addConstr(gp.quicksum(abs_diff[n, time] for time in range(NT)) <= 2 * 1.1 * Cmax[n, 0])

    # Power balance constraints
    for n in range(num_nodes):
        for time in range(NT):
            # Active Power Balance (1b)
            model.addConstr(
                p_n[n, time] + ESS_dis[n, time] - Pd[n, time] - ESS_cha[n, time]
                ==
                gp.quicksum(A_plus[n, l] * p_sl[l, time] for l in range(num_lines))
                - gp.quicksum(A_minus[n, l] * p_ol[l, time] for l in range(num_lines))
                + G_n[n, time] * V_n[n, time]
            )
            
            # Reactive Power Balance (1c)
            model.addConstr(
                q_n[n, time] - q_ESS[n, time] - Qd[n, time] 
                == 
                gp.quicksum(A_plus[n, l] * q_sl[l, time] for l in range(num_lines)) 
                - gp.quicksum(A_minus[n, l] * q_ol[l, time] for l in range(num_lines)) 
                - B_n[n, time] * V_n[n, time]
            )

    ### line constraints ###

    for l in range(num_lines):
        for time in range(NT):
            # Voltage drop constraint (1d):
            model.addConstr(
                gp.quicksum(Inc_sending_cvx[l, n] * V_n[n, time] for n in range(num_nodes)) -
                gp.quicksum(Inc_receiving_cvx[l, n] * V_n[n, time] for n in range(num_nodes))
                ==
                2 * R_l[l, time] * p_sl[l, time] + 2 * X_l[l, time] * q_sl[l, time]
                - R_l[l, time] * p_ol[l, time] - X_l[l, time] * q_ol[l, time]
            )
            
            # Conic active and reactive power losses constraint (2b):
            model.addConstr(
                K_ol[l, time] == (
                    K_l[l, time] - gp.quicksum(Inc_sending_cvx[l, n] * V_n[n, time] for n in range(num_nodes)) * B_l[l, time] ** 2 
                    + 2 * q_sl[l, time] * B_l[l, time]) * X_l[l, time]
            )
            model.addConstr(K_ol[l, time] >= q_ol[l, time])

            # Power loss constraint (2c):
            model.addConstr(p_ol[l, time] * X_l[l, time] == q_ol[l, time] * R_l[l, time])
            
            # Line angle constraint (1h):
            model.addConstr(
                theta_l[l, time] ==
                gp.quicksum(Inc_sending_cvx[l, n] * theta_n[n, time] for n in range(num_nodes))
                - gp.quicksum(Inc_receiving_cvx[l, n] * theta_n[n, time] for n in range(num_nodes))
            )

            # Linearized angle constraint (2d):
            model.addConstr(theta_l[l, time] == X_l[l, time] * p_sl[l, time] - R_l[l, time] * q_sl[l, time])

            # Feasibility solution recovery equation (4g):
            model.addQConstr(
                (2 * theta_l[l, time] / np.sin(theta_l_max[l, time]))**2 + 
                (V_n[sending_node[l], time] - V_n[receiving_node[l], time])**2 <= 
                (V_n[sending_node[l], time] + V_n[receiving_node[l], time])**2
            )

    # Conic active and reactive power losses constraint (2b): (rest of ineq)
    # gurobipy没法直接写SOCP constraint, 而且加t_cone松弛会出现t_cone浮荡的问题，会导致objective负的太多
    for l in range(num_lines):
        for time in range(NT):
            model.addQConstr(X_l[l, time] * (p_sl[l, time] **2 + q_sl[l, time] **2) <= q_ol[l, time] * V_n[sending_node[l], time])

    ######################################################
    """Objective Function"""
    curtailment_cost = 26
    price_imp = [620, 620, 620, 620, 620, 620, 620, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 890, 620, 620, 620]
    
    # Quadratic terms
    quad_terms = gp.quicksum(quad_cost[n, t] * (p_n[n, t] * baseMVA)**2 
                            for n in range(num_nodes) for t in range(NT))   
    # Linear terms
    lin_terms = gp.quicksum(lin_cost[n, t] * p_n[n, t] * baseMVA 
                           for n in range(num_nodes) for t in range(NT))   
    # Constant terms
    const_terms = gp.quicksum(const_cost[n, t] 
                             for n in range(num_nodes) for t in range(NT))
    
    # ESS terms
    ess_terms = gp.quicksum(ESS_cha[n, t] + ESS_dis[n, t] 
                           for n in range(num_nodes) for t in range(NT)) * baseMVA
    
    # Power loss terms
    loss_terms = gp.quicksum(p_ol[l, t] 
                            for l in range(num_lines) for t in range(NT)) * baseMVA
    
    # Import/Export terms
    imp_terms = gp.quicksum(p_imp[t] * price_imp[t] 
                           for t in range(NT)) * baseMVA
    exp_terms = gp.quicksum(p_exp[t] 
                           for t in range(NT)) * baseMVA
    
    # Curtailment terms
    curt_terms = gp.quicksum(p_curtailment[n, t] 
                            for n in range(num_nodes) for t in range(NT)) * baseMVA

    # Set objective
    model.setObjective(
        Yp * 365 * (quad_terms + lin_terms + const_terms) +
        Yp * 2910 * ess_terms +
        loss_terms * 100 * 365 * Yp +
        imp_terms * 365 * Yp +
        exp_terms * 100 * 365 * Yp +
        curt_terms * curtailment_cost * Yp * 365,
        GRB.MINIMIZE
    )

    # Optimize
    model.setParam("FeasibilityTol", 1e-7)
    model.setParam("OptimalityTol", 1e-7)
    model.setParam("QCPDual", 1)
    model.optimize()

    # Get results
    if model.status == GRB.OPTIMAL:
        # Extract solution values
        p_n_val = np.array([[p_n[n, time].x for time in range(NT)] for n in range(num_nodes)])
        q_n_val = np.array([[q_n[n, time].x for time in range(NT)] for n in range(num_nodes)])
        V_n_val = np.array([[V_n[n, time].x for time in range(NT)] for n in range(num_nodes)])
        p_sl_val = np.array([[p_sl[l, time].x for time in range(NT)] for l in range(num_lines)])
        q_sl_val = np.array([[q_sl[l, time].x for time in range(NT)] for l in range(num_lines)])
        p_ol_val = np.array([[p_ol[l, time].x for time in range(NT)] for l in range(num_lines)])
        q_ol_val = np.array([[q_ol[l, time].x for time in range(NT)] for l in range(num_lines)])
        K_ol_val = np.array([[K_ol[l, time].x for time in range(NT)] for l in range(num_lines)])
        theta_n_val = np.array([[theta_n[n, time].x for time in range(NT)] for n in range(num_nodes)])
        theta_l_val = np.array([[theta_l[l, time].x for time in range(NT)] for l in range(num_lines)])
        ESS_soc_val = np.array([[ESS_soc[n, time].x for time in range(NT)] for n in range(num_nodes)])
        ESS_cha_val = np.array([[ESS_cha[n, time].x for time in range(NT)] for n in range(num_nodes)])
        ESS_dis_val = np.array([[ESS_dis[n, time].x for time in range(NT)] for n in range(num_nodes)])
        q_ESS_val = np.array([[q_ESS[n, time].x for time in range(NT)] for n in range(num_nodes)])

        # Get dual values
        lambda_ = np.array([[Rmax_constr[n, time].Pi for time in range(NT)] for n in range(num_nodes)])
        mu_ = np.array([[Cmax_constr[n, time].Pi for time in range(NT)] for n in range(num_nodes)])

        return (model.objVal, p_n_val, q_n_val, np.sqrt(V_n_val), p_sl_val, q_sl_val, 
                p_ol_val, q_ol_val, K_ol_val, theta_n_val, theta_l_val, ESS_soc_val, 
                ESS_cha_val, ESS_dis_val, q_ESS_val, lambda_, mu_)
    else:
        raise Exception("Optimization failed with status: " + str(model.status))