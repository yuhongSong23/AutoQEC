import pandas as pd
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
import math


# ------------------Obtain the Noise Information from IBM Backend Noise File-------------------
# Read the csv file and compute the average error rate for single qubit gate error (Basic gate: ID, X, SX, RZ)
def read_csv_compute_average_q1_error(csv_file, keywords):
    data = pd.read_csv(csv_file)
    # Find column names containing specific keywords
    selected_columns = [col for col in data.columns if any(keyword in col for keyword in keywords)]
    print(selected_columns)
    # Select the specified column
    selected_data = data[selected_columns]
    # compute the average
    averages = selected_data.mean()
    return averages


# Print IBM backend average noise information (Basic gate: ID, X, SX, RZ)
def ibm_noise_information():
    csv_file = './noise_data/ibm_torino_calibrations_2024-10-31T17_49_32Z.csv'
    keywords = ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error']
    gates_avg = read_csv_compute_average_q1_error(csv_file, keywords)
    q1_error = gates_avg.mean()
    print(gates_avg)
    print("Average q1 error: ", q1_error)


# Obtain the transpiled circuit for RX, RY, RZ logical gates
def qiskit_transpile_rotation_gate(gate_name, angle):
    '''
        for single-qubit gate, all the 11 IBM backends are ID, RZ, SX, X
        for 2-qubit gate, ibm_torino utilize CZ gate; other backends utilize ECR gate
    '''
    basic_gate = ['id', 'rz', 'sx', 'x', 'cx']
    qc = QuantumCircuit(3)
    if gate_name == "RX":
        qc.rx(angle, 0)
    elif gate_name == "RY":
        qc.ry(angle, 0)
    elif gate_name == "RZ":
        qc.rz(angle, 0)
    elif gate_name == "CCZ":
        qc.ccz(0, 1, 2)
    qc_trans = transpile(qc, basis_gates=basic_gate)
    print(qc_trans.draw())

'''
(2024-11-07) Transpiled Circuit for RX, RY, RZ
RX(0.1) gate:
   ┌─────────┐┌────┐┌────────────┐┌────┐┌──────────┐
q: ┤ Rz(π/2) ├┤ √X ├┤ Rz(3.2416) ├┤ √X ├┤ Rz(5π/2) ├
   └─────────┘└────┘└────────────┘└────┘└──────────┘
RY(0.1) gate:
   ┌────────┐┌────┐┌────────────┐┌────┐
q: ┤ Rz(-π) ├┤ √X ├┤ Rz(3.0416) ├┤ √X ├
   └────────┘└────┘└────────────┘└────┘
RZ(0.1) gate:
   ┌─────────┐
q: ┤ Rz(0.1) ├
   └─────────┘
'''


# -------------Compute error probability---------------
# compute the probability of QEC code invalid for given number of noisy gates, qec code distance and average depolarizing error
def qec_code_invalid_prob(m, d, p):
    # m is the number of noisy gate, d is the distance of qec code, p is the error rate of each logical gate
    prob = 1.0
    max_error_num = math.floor((d-1)/2) # the ability of qec code corrector
    for k in range(max_error_num+1):
        prob -= math.comb(m, k) * pow(p, k) * pow(1-p, m-k)
    return prob

# Compute average depolarizing noise, given the target overall error probability
def compute_avg_depolarizing_gate_error(m, target_error_prob):
    # m is the number of noisy gate
    p_list = []
    for target in target_error_prob:
        # (1-p)^n=1-target
        ln_result = math.log(1.0 - target) / m
        p = 1.0 - math.exp(ln_result)
        print(f"Average depolarizing noise: {p}")
        print(f"Verification: {1.0 - pow(1 - p, m)}")
        p_list.append(p)
    return p_list

# Compute overall error probability for given average depolarizing noise
def compute_overall_probability_given_avg_depolarizing_noise(m, p_list):
    error_prob_list = []
    for p in p_list:
        error_prob = 1.0 - pow(1 - p, m)
        error_prob_list.append(error_prob)
    return error_prob_list


























