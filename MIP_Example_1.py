# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 22:04:49 2021
MIP PACKAGE EXAMPLE -VRPTW
@author: KRISHNENDU MUKHERJEE
"""
from mip import *
import pandas as pd
import numpy as np
import os
import time


PATH = "ENTER PATH OF C101_200 FILE HERE"
FILENAME = "C101_200"


def load_data(fpath,fname):
    path = os.path.join(fpath,fname + ".csv")
    data = pd.read_csv(path)
    name = "Solomon_"+fname
    print(f"Probem Number {name} \n")
    print(f"Number of rows {data.shape[0]} and columns {data.shape[1]}\n")
    print(data.info())
    return data

def distance(data,cust1,cust2):
    print(f"Selecting customer1 {cust1} and customer2 {cust2}")
    cust1_xcoord = data["XCOORD"][cust1]
    cust1_ycoord = data["YCOORD"][cust1]
    cust2_xcoord = data["XCOORD"][cust2]
    cust2_ycoord = data["YCOORD"][cust2]
    #print(cust1_xcoord,cust1_ycoord,cust2_xcoord,cust2_ycoord)
    dist = round(np.sqrt((cust1_xcoord-cust2_xcoord)**2 + (cust1_ycoord-cust2_ycoord)**2 ),2)
    return dist

def mip_model(data):
    START_TIME=time.time()
    customers = [i for i in range(1,len(data))]
    vertex = [0] + customers
    Arc = [(i,j) for i in vertex for j in vertex if i!=j]
    
    model = Model()
    "x_bin(i,j)"
    x_bin = [[model.add_var(var_type=BINARY,name =f"x({i},{j})") for j in vertex] for i in vertex]
    model.objective= minimize(xsum(distance(data,i,j)*x_bin[i][j] for i in vertex for j in vertex if i!=j))
    "Constraint 1"
    for i in customers:
        model += xsum(x_bin[i][j] for j in vertex if i!=j)==1
    for j in customers:
        model += xsum(x_bin[i][j] for i in vertex if i!=j)==1
    
    bigM=99999
    "Indicator Constraint Equivalent:"
    "If bin_x[i,j]==1 then u[i] + demand[j] <= u[j]"
    "Where q[i] <= u[i] <= Capcity"
    u = [model.add_var(name=f"u({i})",lb=data["DEMAND"].iloc[i],ub=200) for i in vertex]     
    for i,j in Arc:
        if i!=j and j!=0:
             model += u[i] + data["DEMAND"].iloc[j] + bigM * x_bin[i][j] <= u[j] + bigM
             
    
    "Indicator Constraint Equivalent:"
    "If bin_x[i,j]==1 then t[i] + service[j] <= t[j]"
    "Where q[i] <= u[i] <= Capcity"  
    "Arrival time at t[i]"
    t = [model.add_var(name = f"t({i})", lb = data["READY_TIME"].iloc[i], ub = data["DUE_DATE"].iloc[i]) for i in vertex]
    t[0] = data["READY_TIME"].iloc[0]
    bigM=99999
    for i,j in Arc:
        if i!=j and j!=0:
            model += t[i] + data["SERVICE_TIME"].iloc[i] + 0.04 * distance(data,i,j) + bigM*x_bin[i][j] <= t[j] + bigM
    
    "Sub-tour Elimination MTZ:"
    U = [model.add_var(name = f"U({i})",lb=0,ub=np.inf) for i in range(1,len(customers)+2)]
    for i in vertex:
        for j in vertex:
            if i !=j and i!=0 and j!=0:
                
                model += U[i]-U[j] + (len(vertex)-1)*x_bin[i][j] <= (len(vertex)-2)
    model.max_gap = 0.16
    print("Starting the Optimization Solver...\n")
    print("Searching for optimum route...\n")
    status = model.optimize(max_seconds=3000)
    print(f"Status : {status}")
    if status == OptimizationStatus.OPTIMAL:
        print(f"Optimal solution  {model.objective_value}\n")
    elif status == OptimizationStatus.FEASIBLE:
        print(f"Optimal solution {model.obective_value} and best possible {model.objective_bound}\n")
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print(f" No feasible solution found, lower bound {model.objective_value}\n")
    else:
        print(f"Infeasible Problem\n ")
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
       print('solution:')
       for v in model.vars:
           #print(v)
           if abs(v.x) > 1e-6: # only printing non-zeros
              print('{} = {}'.format(v.name, v.x))
    
    fpath = PATH[:-4] + "LP" + "/" + FILENAME +"_VRPTW_MIP.lp"
    model.write(fpath)
    END_TIME = time.time()
    print(f"Computational Time: {END_TIME - START_TIME} Secs")
    
    
data = load_data(PATH,FILENAME)
mip_model(data)


