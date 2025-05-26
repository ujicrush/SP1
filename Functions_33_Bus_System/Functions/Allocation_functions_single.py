import cvxpy as cp
import numpy as np
import math
import gurobipy

def Allocation(iter, N_Bus, ESS_candidate, R_bounds, C_bounds, obj_2nd, lambda_2nd, mu_2nd, previous_rating, previous_cap, Fixed_cost, Power_rating_cost, Energy_capacity_cost,C_rate=1):
    # Used for allocation problem where number of scenario is 1

    # Unfolding data:
    R_min = R_bounds[0]
    R_max = R_bounds[1]
    C_min = C_bounds[0]
    C_max = C_bounds[1]

    # Adding variables for MILP
    U = cp.Variable((N_Bus),boolean=True) # Location for ESS solutions
    Cap_U = cp.Variable(N_Bus)  # Continuous variable representing the maximum energy storage capacity at each node.
    Rating_U = cp.Variable(N_Bus)  # Continuous variable representing the maximum energy storage rating at each node.
    alpha = cp.Variable() # using for benders decomposition

    #Values that need to be calculated
    non_candidate = np.ones(N_Bus)
    non_candidate[ESS_candidate] = 0 # Usefull to constraint the location of ESS storage system to only candidate nodes

    # Constraints: 
    constraints = []
    # Physical limits on power and capacity
    constraints += [Rating_U >= cp.multiply(R_min,U), Rating_U <= cp.multiply(R_max,U)] # Constraint rating to max and min values
    constraints += [Cap_U >= cp.multiply(C_min,U), Cap_U <= cp.multiply(C_max,U)] # Constraint capacity to max and min values
    constraints.append(cp.multiply(non_candidate,U) == np.zeros(N_Bus))
    constraints.append(C_rate * Cap_U >= Rating_U)

    # Benders decompositions 
    bdcut2 = [] 
    for k in range(iter):
        bdcut2.append(alpha >= obj_2nd[k] + cp.sum(lambda_2nd[:,:,k],axis=1).T@(Rating_U - previous_rating[:,k]) + cp.sum(mu_2nd[:,:,k],axis=1).T@(Cap_U - previous_cap[:, k]))

    constraints += bdcut2 + [alpha >=0] # +bdcut

    # Objective function
    objective = cp.multiply(Fixed_cost, cp.sum(U)) + (cp.multiply(Power_rating_cost,cp.sum(Rating_U)) + cp.multiply(Energy_capacity_cost,cp.sum(Cap_U))) + alpha
    
    # Solve
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)

    Investment = Fixed_cost*np.sum(U.value) + (Power_rating_cost*np.sum(Rating_U.value)+Energy_capacity_cost*np.sum(Cap_U.value))

    return problem.value, Investment, U.value, Rating_U.value, Cap_U.value, alpha.value

# Used when there are multiple number of scenarios
def Allocation_2D(
        iter, NP, N_Bus, ESS_candidate, R_bounds, C_bounds, obj_2nd, lambda_2nd, mu_2nd, 
        previous_rating, previous_cap, Fixed_cost, Power_rating_cost, Energy_capacity_cost, C_rate=1):

    # Unfolding data:
    R_min = R_bounds[0]
    R_max = R_bounds[1]
    C_min = C_bounds[0]
    C_max = C_bounds[1]

    # Adding variables for MILP
    U = cp.Variable((N_Bus), boolean=True) # Location for ESS solutions
    Cap_U = cp.Variable(N_Bus)  # Continuous variable representing the maximum energy storage capacity at each node.
    Rating_U = cp.Variable(N_Bus)  # Continuous variable representing the maximum energy storage rating at each node.
    alpha = cp.Variable() # using for benders decomposition

    #Values that need to be calculated
    non_candidate = np.ones(N_Bus)
    non_candidate[ESS_candidate] = 0 # Usefull to constraint the location of ESS storage system to only candidate nodes

    # Constraints: 
    constraints = []
    # Physical limits on power and capacity
    constraints += [Rating_U >= cp.multiply(R_min, U), Rating_U <= cp.multiply(R_max, U)] # Constraint rating to max and min values
    constraints += [Cap_U >= cp.multiply(C_min, U), Cap_U <= cp.multiply(C_max, U)] # Constraint capacity to max and min values
    constraints.append(cp.multiply(non_candidate,U) == np.zeros(N_Bus))
    constraints.append(C_rate * Cap_U >= Rating_U)

    # Benders decompositions 
    bdcut2 = []  # Single-cut

    for k in range(iter):
        bdcut2.append(alpha >= cp.sum(obj_2nd[:, k]) + 
                    np.sum(np.sum(lambda_2nd[:, :, :, k], axis=2), axis=0).T @ (Rating_U - previous_rating[:, k]) + 
                    np.sum(np.sum(mu_2nd[:, :, :, k], axis=2), axis=0).T @ (Cap_U - previous_cap[:, k]))

    constraints += bdcut2 + [alpha >= 0]

    # Objective function
    objective = (
        cp.multiply(Fixed_cost, cp.sum(U)) + 
        cp.multiply(Power_rating_cost, cp.sum(Rating_U)) + 
        cp.multiply(Energy_capacity_cost, cp.sum(Cap_U)) + 
        alpha
    )

    # Solve
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)

    Investment = (
        Fixed_cost * np.sum(U.value) + 
        Power_rating_cost * np.sum(Rating_U.value) + 
        Energy_capacity_cost * np.sum(Cap_U.value)
    )

    return problem.value, Investment, U.value, Rating_U.value, Cap_U.value, alpha.value