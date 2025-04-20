from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import Session
from qiskit_ibm_runtime import EstimatorV2 
from qiskit_ibm_runtime.fake_provider import FakeOsaka

from qiskit_aer import AerSimulator

import numpy as np
import pandas as pd
import itertools as it
import random as rd

"""

Functions that we need

Algorithm Implementation Functions
  Ansatz Circuit
  Hamiltonian Operator
  Optimization functions
  Qiskit Aer Functions
  Running on a fake backend
  

Data Processing Functions
"""


"""

Test 1: 
Scale the system size N and get the following
  The runtime of the variational algorithm
  The estimated ground-state energy values of the Hamiltonian
  Iteration of the algorithm

"""

def run_routine(Nrange, j):


  evals_aer = []
  evals_osaka = []
  analytical_energyvals = []
  time_scaling = []

  aer_sim = AerSimulator()
  fake_osaka = FakeOsaka()
  pm_aer = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
  pm_osaka = generate_preset_pass_manager(backend=fake_osaka, optimization_level=1)
  ##TODO: incorporate proper backends
  estimator_aer = EstimatorV2()
  # estimator_osaka = EstimatorV2(backend=fake_osaka)



  for N in Nrange:

    iteration_vals = []
    evs_evolution_vals = []

    #set up the hamiltonian, quantum circuit

    #we assume the depth scales with the size of the system

    H = tfi_hamiltonian(N, j, j)

    ndepth = N
    parms0 = [2*np.pi * rd.random() for _ in range(N*ndepth)]
    qc = ansatz(N, ndepth, ["ry"])

    #convert to isa architecture 

    isa_aer = pm_aer.run(qc)
    isa_osaka = pm_osaka.run(qc)

    #convert observable to isa layout 
    hamiltonian_aer = H.apply_laout(isa_aer.layout)
    hamiltonian_osaka = H.apply_layout(isa_osaka.layout)




#We use a gradient-free COBYLA method in finding the ground state
def run_optimization(parms0, qc, hamiltonian, estimator):

  return minimize(cost, parms0, args=(ansatz, hamiltonian, estimator), method="COBYLA")

def CNOt_layer(qc:QuantumCircuit, N:int, start_ind:int):

  #if N is odd, then N = 2k + 1 for some k. If N is even, then N = 2k
  #we only need k for the 
  k = N // 2

  for i in range(start_ind, k, step=2):
    qc.cx(i, i + 1)

  
#TODO: create the ansatz circuit without assigning the parameters

def ansatz(N:int, ndepth:int, rgates:list[str]):

  """
  parms: input parameters to the circuit
  N: number of qubits
  ndepth: amount of layers to add
  rgates: list containing rgates of interest
  
  """

  qc = QuantumCircuit(N)

  #create the parameter for every gate we would like to add
  parm = [Parameter(f"th{i}") for i in range(N*ndepth)]

  rgate_iter = 0
  amount_rgates = len(rgates)

  for i in range(ndepth):

    if rgate_iter >= amount_rgates:
      rgate_iter = 0

    #apply control gates 

    if rgates[rgate_iter] == "rx":
      for j in range(N):
        #apply rx gates
        qc.rx(parm[j + N*i], j)
    elif rgates[rgate_iter] == "ry":
      for j in range(N):
        #apply rx gates
        qc.ry(parm[j + N*i], j)
    else:
      for j in range(N):
        #apply rx gates
        qc.rz(parm[j + N*i], j)

    #apply CNOT gates
    #if i = 0, 2, 4, ... then apply CNOT gates starting at 0
    # else, for odd i, start at index 1 
    CNOt_layer(qc, N, (i + 1) % 2) 

  return qc

def tfi_hamiltonian(N:int, h:float, j:float) -> SparsePauliOp:

  #we have 2N - 1 terms, N terms from the transverse field, N-1 from interactions

  interacting_string = [ "".join(["I" for _ in range(N)]) for _ in range(N-1) ]

  for i in range(N-1):
    interacting_string[i][i] = "Z"
    interacting_string[i][i + 1] = "Z"

  transverse_string = [ "".join(["I" for _ in range(N)]) for _ in range(N) ]

  for i in range(N):
    transverse_string[i][i] = "X"

  interacting_coefficients = [-h for _ in range(N-1)]
  transverse_coefficients = [-j for _ in range(N)]

  coeffs = interacting_coefficients + transverse_coefficients
  paulis = interacting_string + transverse_string

  return SparsePauliOp(paulis, coeffs=coeffs)  

#Straight off ripped from qiskit
def cost(params, circuit, hamiltonian, estimator):

  pub = (circuit, hamiltonian, params)

  cost = estimator.run([pub]).result()[0].data.evs
  return cost


if __name__ == "__main__":

  #define the system size 
  N = 4

  

    

