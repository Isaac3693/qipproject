from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import TwoLocal
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator, StatevectorSimulator

import numpy as np
import pandas as pd
import itertools as it
import random as rd
import time
import matplotlib.pyplot as plt
from scipy.special import ellipe

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

def run_routine(Nrange, h, j, samples=1):
    """
    Run the simulation for different system sizes and compare results
    
    Args:
        Nrange: List of system sizes to simulate
        h: Field strength parameter
        j: Coupling strength parameter
        samples: Number of samples for averaging
        
    Returns:
        Dictionary with results
    """
    results = {
        'N': [],
        'vqe_energy': [],
        'analytical_energy': [],
        'vqe_time': [],
        'iterations': []
    }
    
    for N in Nrange:
        print(f"Processing N = {N}")
        results['N'].append(N)
        
        # Calculate analytical energy for comparison
        analytical_energy = analytical_gs_energy(N, h, j)
        results['analytical_energy'].append(analytical_energy)
        
        # VQE simulation timing and energy
        vqe_times = []
        vqe_energies = []
        vqe_iterations = []
        
        for _ in range(samples):
            # Set up the Hamiltonian and circuit
            H = tfi_hamiltonian(N, h, j)
            ndepth = N
            
            # Create parameterized circuit
            qc, params = ansatz(N, ndepth, ["ry"])
            
            # Random initial parameters
            parms0 = [2*np.pi * rd.random() for _ in range(len(params))]
            
            # Run optimization
            start_time = time.time()
            opt_result = run_optimization(parms0, qc, H, params)
            end_time = time.time()
            
            vqe_times.append(end_time - start_time)
            vqe_energies.append(opt_result.fun)
            
            # Safely get iteration count
            iteration_count = get_iteration_count(opt_result)
            vqe_iterations.append(iteration_count)
        
        # Average results
        results['vqe_time'].append(np.mean(vqe_times))
        results['vqe_energy'].append(np.mean(vqe_energies))
        results['iterations'].append(np.mean(vqe_iterations))
    
    return results

def run_optimization(parms0, qc, hamiltonian, params, method="COBYLA"):
    """Run optimization to find ground state"""
    def cost_wrapper(parameter_values):
        return cost_local(parameter_values, qc, hamiltonian, params)
    
    return minimize(cost_wrapper, parms0, method=method, options={'maxiter': 200})

def CNOt_layer(qc: QuantumCircuit, N: int, start_ind: int):
    """Apply a layer of CNOTs"""
    for i in range(start_ind, N-1, 2):
        qc.cx(i, i + 1)

def ansatz(N: int, ndepth: int, rgates: list[str]):
    """
    Create a parameterized ansatz circuit
    
    Args:
        N: number of qubits
        ndepth: amount of layers to add
        rgates: list containing rgates of interest
    
    Returns:
        qc: Parameterized quantum circuit
        params: List of all parameters
    """
    qc = QuantumCircuit(N)
    
    # Calculate total number of parameters needed
    total_params = N * ndepth
    params = [Parameter(f"th{i}") for i in range(total_params)]
    
    rgate_iter = 0
    amount_rgates = len(rgates)
    
    for i in range(ndepth):
        if rgate_iter >= amount_rgates:
            rgate_iter = 0
            
        # Apply rotation gates
        if rgates[rgate_iter] == "rx":
            for j in range(N):
                qc.rx(params[j + N*i], j)
        elif rgates[rgate_iter] == "ry":
            for j in range(N):
                qc.ry(params[j + N*i], j)
        else:
            for j in range(N):
                qc.rz(params[j + N*i], j)
                
        # Apply CNOT gates
        CNOt_layer(qc, N, (i + 1) % 2)
        
        rgate_iter += 1
    
    return qc, params

def tfi_hamiltonian(N: int, h: float, j: float) -> SparsePauliOp:
    """
    Create the Transverse Field Ising model Hamiltonian
    
    Args:
        N: Number of qubits
        h: Field strength
        j: Coupling strength
        
    Returns:
        SparsePauliOp representation of the Hamiltonian
    """
    # We have 2N - 1 terms, N terms from the transverse field, N-1 from interactions
    interacting_string = []
    for i in range(N-1):
        pauli_str = ['I'] * N
        pauli_str[i] = 'Z'
        pauli_str[i+1] = 'Z'
        interacting_string.append(''.join(pauli_str))

    transverse_string = []
    for i in range(N):
        pauli_str = ['I'] * N
        pauli_str[i] = 'X'
        transverse_string.append(''.join(pauli_str))

    interacting_coefficients = [-j for _ in range(N-1)]
    transverse_coefficients = [-h for _ in range(N)]

    coeffs = interacting_coefficients + transverse_coefficients
    paulis = interacting_string + transverse_string

    return SparsePauliOp(paulis, coeffs=coeffs)

def cost_local(params, circuit, hamiltonian, circuit_params):
    """
    Calculate energy expectation value using local statevector simulation
    """
    # Create parameter dictionary
    param_dict = dict(zip(circuit_params, params))
    
    # Bind parameters
    # vnot sure why, but bind_parameters wasnt showing up as a method
    bound_circuit = circuit.copy()
    bound_circuit.assign_parameters(param_dict, inplace=True)
    
    # Use statevector simulation to compute expectation value
    # This is the key part that replaces the EstimatorV2
    simulator = StatevectorSimulator()
    job = simulator.run(bound_circuit)
    result = job.result()
    sv = result.get_statevector()
    
    # Calculate expectation value <ψ|H|ψ>
    expectation_value = sv.expectation_value(hamiltonian)
    
    return expectation_value.real

def analytical_gs_energy(N, h, j):
    """
    Calculate the analytical ground state energy for the TFI model
    using the elliptic integral formula
    """
    m = 4*h*j/((h+j)**2)
    # Use scipy's implementation of the complete elliptic integral
    E = ellipe(m)
    return -(2*N/np.pi) * abs(h + j) * E

def plot_ansatz(N=4, depth=1):
    """Generate a visualization of the ansatz circuit"""
    qc, _ = ansatz(N, depth, ["ry"])
    qc.draw('mpl', filename='ansatz_circuit.png')
    print("Ansatz circuit visualization saved as 'ansatz_circuit.png'")

def get_iteration_count(opt_result):
    """
    Safely extract iteration count from optimization result
    Different optimizers store this information in different attributes
    """
    # First, print the entire result object to inspect its structure
    print("Optimization result attributes:")
    print(dir(opt_result))
    
    # Try different common attributes for iteration count
    if hasattr(opt_result, 'nit'):
        return opt_result.nit
    elif hasattr(opt_result, 'iterations'):
        return opt_result.iterations
    elif hasattr(opt_result, 'nfev'):
        # If no iteration count, use function evaluation count as a proxy
        return opt_result.nfev
    else:
        # Default value if no iteration information is available
        print("Warning: Could not find iteration count in optimization result")
        return 0

def compare_optimization_methods(N=7, h=1.0, j=1.0, methods=["COBYLA", "BFGS"]):
    """Compare different optimization methods"""
    results = {method: [] for method in methods}
    iterations = {method: [] for method in methods}
    
    # Set up the system
    H = tfi_hamiltonian(N, h, j)
    ndepth = N
    qc, params = ansatz(N, ndepth, ["ry"])
    
    # Initial parameters (use same for all methods)
    np.random.seed(42)  # For reproducibility
    parms0 = [2*np.pi * rd.random() for _ in range(len(params))]
    
    for method in methods:
        print(f"Running optimization with {method}")
        opt_result = run_optimization(parms0, qc, H, params, method=method)
        
        # Print full result object to see what's available
        print(f"Full optimization result for {method}:")
        print(opt_result)
        
        # Store convergence data
        results[method].append(opt_result.fun)
        
        # Get iteration count safely
        iterations[method].append(get_iteration_count(opt_result))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for method in methods:
        plt.bar(method, results[method][0])
    plt.ylabel('Ground State Energy')
    plt.title(f'Comparison of Optimization Methods (N={N}, h={h}, j={j})')
    plt.savefig('optimization_comparison.png')
    
    return results, iterations

def scaling_study(max_N=11, h=1.0, j=1.0, samples=100):
    """
    Study the scaling of VQE vs diagonalization with system size
    
    Args:
        max_N: Maximum system size to simulate
        h, j: Hamiltonian parameters
        samples: Number of samples for averaging
        
    Returns:
        pandas DataFrame with results
    """
    Nrange = list(range(2, max_N+1))
    results = run_routine(Nrange, h, j, samples)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate relative error
    df['rel_error'] = np.abs(df['vqe_energy'] - df['analytical_energy']) / np.abs(df['analytical_energy'])
    
    # Plot runtime scaling
    plt.figure(figsize=(10, 6))
    plt.semilogy(df['N'], df['vqe_time'], 'o-', label='VQE')
    plt.xlabel('System Size (N)')
    plt.ylabel('Runtime (s) - log scale')
    plt.grid(True)
    plt.legend()
    plt.title('Runtime Scaling with System Size')
    plt.savefig('runtime_scaling.png')
    
    # Plot energy comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df['N'], df['vqe_energy'], 'o-', label='VQE')
    plt.plot(df['N'], df['analytical_energy'], 'k--', label='Analytical')
    plt.xlabel('System Size (N)')
    plt.ylabel('Ground State Energy')
    plt.legend()
    plt.grid(True)
    plt.title('Ground State Energy vs System Size')
    plt.savefig('energy_comparison.png')
    
    # Plot relative error
    plt.figure(figsize=(10, 6))
    plt.semilogy(df['N'], df['rel_error'], 'o-')
    plt.xlabel('System Size (N)')
    plt.ylabel('Relative Error (log scale)')
    plt.grid(True)
    plt.title('VQE Energy Relative Error')
    plt.savefig('relative_error.png')
    
    return df

if __name__ == "__main__":
    # Generate ansatz circuit visualization
    plot_ansatz(N=4, depth=1)
    
    # Compare optimization methods
    opt_results, opt_iterations = compare_optimization_methods(N=4, h=1.0, j=1.0, 
                                                             methods=["COBYLA", "BFGS"])
    print(f"Optimization results: {opt_results}")
    print(f"Iterations: {opt_iterations}")
    
    # For a quicker test, use fewer samples and smaller system sizes
    results_df = scaling_study(max_N=6, h=1.0, j=1.0, samples=2)
    print(results_df)



