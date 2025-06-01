import cvxpy as cp
import numpy as np
import math
import gurobipy
import mosek

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


def check_feasibility(baseMVA, NT, num_nodes, num_lines, Yp, sending_node, receiving_node, IndPV,
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
    """Variables"""
    p_n = cp.Variable((num_nodes,NT))  # Active power at node n
    p_curtailment = cp.Variable((num_nodes, NT), nonneg=True)
    p_slack = p_n[0, :]
    p_imp = cp.pos(p_slack)
    p_exp = cp.pos(-p_slack)
    q_n = cp.Variable((num_nodes,NT))  # Reactive power at node n
    V_n = cp.Variable((num_nodes, NT))  # Voltage magnitude squared at node n
    theta_n = cp.Variable((num_nodes,NT))  # Voltage angles at node n
    p_sl = cp.Variable((num_lines,NT))  # Active power at sending end of line l
    q_sl = cp.Variable((num_lines,NT))  # Reactive power at sending end of line l
    p_ol = cp.Variable((num_lines,NT))  # Active power losses on line l
    q_ol = cp.Variable((num_lines,NT))  # Reactive power losses on line l
    K_ol = cp.Variable((num_lines,NT))  # Branch Equivalent ampacity constraint on line l
    theta_l = cp.Variable((num_lines,NT))  # Voltage angles at line l

    ESS_cha = cp.Variable((num_nodes,NT)) 
    ESS_dis = cp.Variable((num_nodes,NT))
    ESS_soc = cp.Variable((num_nodes,NT))
    q_ESS = cp.Variable((num_nodes,NT))
    
    # Variables for allocation
    Cmax = cp.Variable((num_nodes,NT))
    Rmax = cp.Variable((num_nodes,NT))

    # Create the Incidence Matrices used in the sending end and receiving end voltage
    Inc_sending_cvx = np.zeros((num_lines, num_nodes))
    Inc_receiving_cvx = np.zeros((num_lines, num_nodes))
    for l in range(num_lines):
        Inc_sending_cvx[l, sending_node[l]] = 1
        Inc_receiving_cvx[l, receiving_node[l]] = 1

    ######################################################
    """ Constraints"""
    constraints = []
    objective = 0
    
    ### Bus constraints ###

    # Voltage Magnitude bounds (1k)
    constraints.append(V_n >= V_min)
    constraints.append(V_n <= V_max)

    # Node angle bounds (1m)
    constraints.append(theta_n >= theta_n_min)
    constraints.append(theta_n <= theta_n_max)

    # Active power bounds (1n)
    constraints.append(p_n >= p_n_min)
    constraints.append(p_n <= p_n_max)

    # for time in range(NT):
    #     if len(IndPV[time])>0:
    #         constraints.append(p_n[IndPV[time],:] == p_n_max[IndPV[time],:]) # enforces the power injection to be equal to PV production

    for time in range(NT):
        if len(IndPV[time]) > 0:
            # p_n + p_curtailment = p_n_max
            constraints.append(p_n[IndPV[time], :] + p_curtailment[IndPV[time], :] == p_n_max[IndPV[time], :])

    # Reactive power bounds (1o)
    constraints.append(q_n >= q_n_min)
    constraints.append(q_n <= q_n_max)

    # Alocation related constraints to get the dual variables
    constraint_capacity = Cmax == ESS_soc_max
    constraint_rating = Rmax == ESS_cha_max
    constraints += [constraint_capacity, constraint_rating]

    # ESS charging and discharging rate bounds
    constraints.append(ESS_cha >= ESS_cha_min)
    constraints.append(ESS_cha <= Rmax)
    constraints.append(ESS_dis >= ESS_dis_min)
    constraints.append(ESS_dis <= Rmax)

    # ESS SOC bounds
    constraints.append(ESS_soc >= ESS_soc_min)
    constraints.append(ESS_soc <= Cmax)

    # Linking time steps for ESS (excluding the last time step)
    constraints.append(ESS_soc[:, 1:] == ESS_soc[:, :-1] + ESS_cha[:, :-1] - ESS_dis[:, :-1])
    constraints.append(ESS_soc0 == ESS_soc[:,-1] + ESS_cha[:,-1] - ESS_dis[:,-1]) #last timestep to reset battery charge for next day

    # Initializing ESS SOC for the first time step
    constraints.append(ESS_soc[:, 0] == ESS_soc0)
    constraints.append(ESS_soc[:,-1] == ESS_soc0)
    constraints.append(cp.sum(ESS_cha,axis=1) == cp.sum(ESS_dis,axis=1))

    # Battery aging constraints --> battery has to do at maximum 1.1 cycles per day
    constraints.append(cp.sum(cp.abs(ESS_cha-ESS_dis),axis=1) <= 2*1.1*Cmax[:,0])

    # ESS reactive power computation and bounds
    for i in range(lin):
        constraints.append(q_ESS <= slope[i] * (ESS_cha - ESS_dis) + offset[i] * Rmax)
        constraints.append(q_ESS <= -slope[i] * (ESS_cha - ESS_dis) + offset[i] * Rmax)
        constraints.append(q_ESS >= slope[i] * (ESS_cha - ESS_dis) - offset[i] * Rmax)
        constraints.append(q_ESS >= -slope[i] * (ESS_cha - ESS_dis) - offset[i] * Rmax)

    # Active Power Balance (1b)
    constraints.append(p_n + ESS_dis - Pd - ESS_cha == A_plus @ p_sl - A_minus @ p_ol + cp.multiply(G_n, V_n))

    # Reactive Power Balance (1c)
    constraints.append(q_n - q_ESS - Qd == A_plus @ q_sl - A_minus @ q_ol - cp.multiply(B_n, V_n))

    ### line constraints ###

    # Line angle bounds (1l):
    constraints.append(theta_l >= theta_l_min)
    constraints.append(theta_l <= theta_l_max)

    # Voltage drop constraint (1d):
    constraints.append(Inc_sending_cvx @ V_n - Inc_receiving_cvx @ V_n== 2 * cp.multiply(R_l, p_sl) + 2 * cp.multiply(X_l, q_sl) - cp.multiply(R_l, p_ol) - cp.multiply(X_l, q_ol))
    
    # Conic active and reactive power losses constraint (2b):
    constraints.append(K_ol == cp.multiply((K_l - cp.multiply(Inc_sending_cvx @ V_n, B_l**2) + 2 * cp.multiply(q_sl, B_l)), X_l))
    constraints.append(K_ol >= q_ol)

    # Power loss constraint (2c):
    constraints.append(cp.multiply(p_ol,X_l) == cp.multiply(q_ol,R_l))

    # Line angle constraint (1h):
    constraints.append(theta_l == Inc_sending_cvx @ theta_n - Inc_receiving_cvx @ theta_n)

    # Linearized angle constraint (2d):
    constraints.append(theta_l == cp.multiply(X_l,p_sl) - cp.multiply(R_l,q_sl))

    # Constraints that requiere a loop because of dimensionality limit of cp.norm().
    for time in range(NT):
        for l in range(num_lines):
            # Conic active and reactive power losses constraint (2b): (rest of ineq)
            constraints.append(
                cp.norm(
                    cp.vstack([
                        2 *np.sqrt(X_l[l,time])* cp.vstack([p_sl[l,time], q_sl[l,time]]),
                        cp.reshape(q_ol[l,time] - V_n[sending_node[l],time], (1, 1))
                    ]),2
                ) <= q_ol[l,time] + V_n[sending_node[l],time]
            )

            # Feasibility solution recovery equation (4g):
            constraints.append(
                V_n[sending_node[l],time] + V_n[receiving_node[l],time] >= cp.norm(
                    cp.vstack([
                        2*theta_l[l,time]/np.sin(theta_l_max[l,time]), 
                        V_n[sending_node[l],time] - V_n[receiving_node[l],time]
                    ]), 2)
            )
    

    #####################################################################
    """Objective Function""" 
    objective = cp.Minimize(0)

    # Defining the optimization problem
    problem = cp.Problem(objective, constraints)
    #####################################################################

    #####################################################################
    # Solve the problem
    problem.solve(solver=cp.MOSEK)

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return True
    else:
        return False
