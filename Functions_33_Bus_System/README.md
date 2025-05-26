# Semester project

### Functions :

Allocation_functions : functions used to perform 1st stage of benders decomposition, i.e. ESS allocation and cost

SOC_ACOPF : functions that computes the 2nd stage of benders decomposition, i.e. ACOPF operationnion and cost

run : function that is used to perferm overall allocation : imports data, and perform the benders decomposition

--------------------
### Other codes : 

Cluster_build : How are the buildings clustered to a bus on the grid

Chauderon_scenario_process : How the data of clusters, scenario and demand have been merged

-------------------
### Data : 
#### groups folder:

Contains data of buildings for different scenarios groups, clean_demand are the finals files used in run.py

#### 632_0_ data:

These 3 files branch_data, bus_data, generator_data are data in p.u. for respectively lines, nodes and generators that are used in run.py

#### Other data:
Intermediate steps to get "Clean_demand_groupx.csv" 
