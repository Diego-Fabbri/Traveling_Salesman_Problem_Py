import sys
import pandas as pd
import time, numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
import math 
import random
import pandas as pd
import networkx as nx

random.seed(10) # Set seed to make results reproducible

#Parameters

n = 10 # define problem size
  #Generate Random Coordinates
max_val, min_val = 25, 1
range_size = (max_val - min_val)  # 24

X_Pos = np.random.rand(n) * range_size + min_val
Y_Pos = np.random.rand(n) * range_size + min_val

c = np.zeros((len(X_Pos),len(Y_Pos))) #matrix of distances (Cost)

  #Compute distances
for i in range(len(X_Pos)):
    for j in range(len(Y_Pos)):
        c[i][j] = math.sqrt(math.pow(X_Pos[j]- X_Pos[i],2) + math.pow(Y_Pos[j]- Y_Pos[i],2))

range_i = range(0,n)
range_j = range(0,n)

#Create Model
model = pyo.ConcreteModel()

#Define variables
model.u = pyo.Var(range(0,n), # index i
                  bounds = (0,None),
                  initialize = 0)

model.x = pyo.Var(range(0,n), # index i
                  range(0,n), # index j
                  within = Binary,
                  initialize=0)

u = model.u
x = model.x

#Constraints 
model.C1 = pyo.ConstraintList() 
for j in range_j:
    model.C1.add(expr = sum(x[i,j] for i in range_i if i!= j)  == 1)
model.C2 = pyo.ConstraintList() 
for i in range_i:
    model.C2.add(expr = sum(x[i,j] for j in range_j if i!= j)  == 1)

model.C3 = pyo.ConstraintList() 
for i in range(1,n):
    for j in range(1,n):
        if i!= j:
            model.C3.add(expr = u[i] - u[j]+ (n-1)*x[i,j] <= n - 2)
            
# Define Objective Function
model.obj = pyo.Objective(expr = sum(c[i,j]*x[i,j] for i in range_i for j in range_j), 
                          sense = minimize)

begin = time.time()
opt = SolverFactory('cplex')
results = opt.solve(model)

deltaT = time.time() - begin # Compute Exection Duration

model.pprint()

sys.stdout = open("Traveling_Salesman_Problem_CLSP_Problem_Results.txt", "w") #Print Results on a .txt file

print('Time =', np.round(deltaT,2))

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):

    print('Total Cost (Obj value) =', pyo.value(model.obj))
    print('Solver Status is =', results.solver.status)
    print('Termination Condition is =', results.solver.termination_condition)
    print(" " )
    for i in range_i:
        for j in range_j:
            if  pyo.value(x[i,j]) != 0:
               print('x[' ,i+1, '][ ', j+1,']: ', round(pyo.value(x[i,j]),2))
    print(" " )
    for i in range_i:
        print('u[' ,i ,']: ', round(pyo.value(u[i]),2))
elif (results.solver.termination_condition == TerminationCondition.infeasible):
   print('Model is unfeasible')
  #print('Solver Status is =', results.solver.status)
   print('Termination Condition is =', results.solver.termination_condition)
else:
    # Something else is wrong
    print ('Solver Status: ',  result.solver.status)
    print('Termination Condition is =', results.solver.termination_condition)
    
sys.stdout.close()
