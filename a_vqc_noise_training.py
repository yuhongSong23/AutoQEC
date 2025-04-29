import pandas as pd
from a_circuits import CODE, Logical_Circuit, Physical_Circuit_with_Given_Each_Qubit_Error_Probability, Physical_Circuit_wo_Noise
from pennylane import NesterovMomentumOptimizer
from a_vqc_funcs import train_and_test_circuit
from itertools import combinations
import numpy as np

# --------Gate Error class--------
class GateErrorModel:
    def __init__(self):
        self.model = {} # {gate: {qubit: prob, qubit: prob, ...}, ...}

    def add_error(self, gate, qubit, error_prob):
        if gate not in self.model:
            self.model[gate] = {}
        self.model[gate][qubit] = error_prob

    def get_error(self, gate, qubit):
        return self.model.get(gate, {}).get(qubit, None)

    def print_model(self):
        if not self.model:
            print("  The model is empty.")
            return
        for gate, qubits in self.model.items():
            print(f"  Gate: {gate}")
            sorted_qubits = sorted(qubits.items(), key=lambda item: item[1]) # sorted error probability
            # sorted_qubits = sorted(qubits.items())  # sorted qubits
            for qubit, error_prob in sorted_qubits:
                print(f"    Qubit {qubit}: Error Probability = {error_prob}")


# --------build noise model from csv file---------
def get_gate_from_col(col):
    gate_mapping = {
        "ID error": "id",
        "Z-axis rotation (rz) error": "rz",
        "(sx) error": "sx",
        "Pauli-X error": "x",
        "ECR error": "ecr",
        "CZ error": "cz"
    }
    return next((gate_mapping[key] for key in gate_mapping if key in col), None)

def build_gate_error_model_from_csv(error_model, csv_file, keywords):
    # keywords: ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error', 'ECR error'/'CZ error']
    data = pd.read_csv(csv_file)
    selected_columns = [col for col in data.columns if any(keyword in col for keyword in keywords)]
    selected_data = data[selected_columns]
    for col in selected_data:
        if 'ECR' in col: # two-qubit error
            pass
        else: # single-qubit error
            gate = get_gate_from_col(col)
            for qubit, error_prob in enumerate(data[col]):
                error_model.add_error(gate, qubit, error_prob)

def generate_sub_gate_error_model(error_model, gates, noise_qubits, sub_error_model):
    for gate in gates:
        for qubit in noise_qubits:
            error_prob = error_model.get_error(gate, qubit)
            sub_error_model.add_error(gate, qubit, error_prob)


# --------Extract error list for each qubit from given error model
def compute_rotation_gate_error(noise_qubits, error_model):
    # RX/RY gate can be transpiled 2 sx gates and 3/2 rz gates
    error_list = []
    gate = 'sx'
    if gate in error_model.model:
        for qubit in noise_qubits:
            p = error_model.get_error(gate, qubit)
            error_list.append(1.0 - pow(1 - p, 2)) # 2 sx
    return error_list

def compute_noise_level_for_given_noise_qubits(csv_file, keywords, noise_qubits):
    # build the whole noise model
    error_model = GateErrorModel()
    build_gate_error_model_from_csv(error_model, csv_file, keywords)
    # compute the error probability for RX/RY gates
    error_list = compute_rotation_gate_error(noise_qubits, error_model)
    print(f"Rotation gate error for qubit {noise_qubits}: ", error_list)
    return error_list


# ---------Compute overall error probability for given error of each qubit--------
def compute_error_prob(rotation_error_list):
    # 1-p(0)
    p0 = 1.0
    for rot_err in rotation_error_list:
        p0 *= (1.0 - rot_err) * (1.0 - rot_err)  # probability of no error
    err_prob = 1.0 - p0  # probability of occuring errors
    return err_prob

def prob_r_errors(error_list, r):
    # compute P(#error=r)
    total_probability = 0
    n = len(error_list)
    # any error combination
    for indices in combinations(range(n), r):
        error_prob = np.prod([error_list[i] for i in indices]) # occur errors
        no_error_prob = np.prod([1 - error_list[j] for j in range(n) if j not in indices]) # doesn't occur errors
        total_probability += error_prob * no_error_prob
    return total_probability


# --------Group noise levels for given backend csv file--------
def group_different_noise_qubit_to_form_noise_levels(csv_file, keywords, n_phy_qubits):
    error_model = GateErrorModel()
    build_gate_error_model_from_csv(error_model, csv_file, keywords)

    noise_levels = []
    sx_sorted_errors = sorted(error_model.model['sx'].items(), key=lambda item: item[1])
    qubits_list = [qubit for qubit, _ in sx_sorted_errors]
    for i in range(0, len(sx_sorted_errors), n_phy_qubits):
        noise_qubits = qubits_list[i:i + n_phy_qubits]
        rotation_error_list = compute_rotation_gate_error(noise_qubits, error_model)
        err_prob = compute_error_prob(rotation_error_list)
        error_prob1 = prob_r_errors(rotation_error_list+rotation_error_list, 1)
        noise_levels.append((noise_qubits, rotation_error_list, err_prob, error_prob1))
    noise_levels = noise_levels[:-1]
    noise_qubits = qubits_list[-n_phy_qubits:]
    rotation_error_list = compute_rotation_gate_error(noise_qubits, error_model)
    err_prob = compute_error_prob(rotation_error_list)
    error_prob1 = prob_r_errors(rotation_error_list + rotation_error_list, 1)
    noise_levels.append((noise_qubits, rotation_error_list, err_prob, error_prob1))
    return noise_levels

def test_to_build_noise_levels():
    csv_file_list = ['ibm_brussels_calibrations_2025-03-18T10_07_09Z.csv', 'ibm_strasbourg_calibrations_2025-03-19T18_37_44Z.csv']
    for name in csv_file_list:
        csv_file = './noise_data/' + name
        keywords = ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error']
        n_phy_qubits = CODE.phy_n_qubits
        noise_levels = group_different_noise_qubit_to_form_noise_levels(csv_file, keywords, n_phy_qubits)
        for e in noise_levels: # noise qubits
            print(e[0])
        for e in noise_levels: # rotation error list
            print(e[1])
        for e in noise_levels: # overall error probability
            print(e[2])
        print('-' * 20)
        for e in noise_levels: # P(#error=1)
            print(e[3])
        print('*' * 40)


# --------------Training with given noise qubits-----------------
def train_VQC_with_given_noise_qubits(noise_qubits, csv_file, keywords, gate_name, vqc_shape):
    task_name = f"Given Gate-Qubit Noise Level {noise_qubits} for {gate_name} Gate VQC Training"
    print("Task: " + task_name)
    error_list = compute_noise_level_for_given_noise_qubits(csv_file, keywords, noise_qubits)

    criterion = NesterovMomentumOptimizer
    trained_weights = train_and_test_circuit(Physical_Circuit_with_Given_Each_Qubit_Error_Probability, Logical_Circuit, gate_name, task_name, vqc_shape, criterion,
                                             epochs=50, data_num=200, lr=0.5, noise_level=error_list)
    return trained_weights