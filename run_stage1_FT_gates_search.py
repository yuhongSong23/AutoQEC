from pennylane import NesterovMomentumOptimizer
from a_vqc_funcs import train_and_test_circuit, dataset, test
from a_circuits import CODE, Logical_Circuit, Physical_Circuit_with_Fixed_1_Random_Noise, Physical_Circuit_wo_Noise
from a_vqc_noise_training import train_VQC_with_given_noise_qubits
from evaluate_FT_gates_training_methods import test_under_noise_free_environment, test_under_given_physical_circuit
import math
from datetime import datetime
from contextlib import redirect_stdout


# Noise-Free (NF) Training for FT Gates Search
def train_and_inference_for_noise_free_training():
    n_layer = 1
    if CODE.log_n_qubits == 1:
        vqc_shape = [n_layer, CODE.phy_n_qubits, 3]

    # gate_name_list = ['H', 'S', 'T', 'HS', 'SH', 'HSdagger', 'SdaggerH', 'HT', 'TH', 'HTdagger', 'TdaggerH', 'ST', 'SdaggerTdagger']
    gate_name_list = ['H']
    ft_gates_list = []
    for gate_name in gate_name_list:
        task_name = f"Noise-Free Training for {gate_name} Gate"
        print(f"Task: {task_name}")
        trained_weights = train_and_test_circuit(Physical_Circuit_wo_Noise, Logical_Circuit, gate_name, task_name, vqc_shape, NesterovMomentumOptimizer, angle=None, epochs=50, data_num=200, lr=0.5, noise_level=None)
        cost_nf = test_under_noise_free_environment(gate_name, trained_weights, 'NF Training', datanum=5000, angle=None)
        cost_1pe = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, 'NF Training')
        if math.floor(math.log10(abs(cost_nf))) <= -9 and math.floor(math.log10(abs(cost_1pe))) <= -9:
            ft_gates_list.append(gate_name)
    print(f"Supported FT Gates for Code [{CODE.code_name}] using NF-training is {ft_gates_list}")


# Hardware-Specific Noisy training for FT Gates Search
def train_and_inference_for_given_noise_qubits():
    if CODE.code_name == '7_1_3':
        noise_qubits_list = [[27, 104, 72, 41, 122, 119, 52], [118, 109, 113, 110, 70, 60, 48]]
    if CODE.code_name == '5_1_3':
        noise_qubits_list = [[41, 122, 119, 52, 117], [113, 110, 70, 60, 48]]
    label_list = [str(noise_qubits) for noise_qubits in noise_qubits_list]
    # noise information from ibm_brussels (IBM-BXL) and ibm_strasbourge (IBM-SXB) for Table 2
    csv_file_list = ['./noise_data/ibm_brussels_calibrations_2025-03-18T10_07_09Z.csv', './noise_data/ibm_strasbourg_calibrations_2025-03-19T18_37_44Z.csv']
    keywords = ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error']

    n_layer = 1  # the number of ansatz layers
    if CODE.log_n_qubits == 1:
        vqc_shape = [n_layer, CODE.phy_n_qubits, 3]

    # gate_name_list = ['H', 'S', 'T', 'HS', 'SH', 'HSdagger', 'SdaggerH', 'HT', 'TH', 'HTdagger', 'TdaggerH', 'ST', 'SdaggerTdagger']
    gate_name_list = ['H']
    ft_gates_list = []
    for noise_qubits, label, csv_file in zip(noise_qubits_list, label_list, csv_file_list):
        for gate_name in gate_name_list:
            print(f"Hardware-Specific Noisy VQC Training for {gate_name} Gate")
            trained_weights = train_VQC_with_given_noise_qubits(noise_qubits, csv_file, keywords, gate_name, vqc_shape) # training
            cost_nf = test_under_noise_free_environment(gate_name, trained_weights, f'HN-{label}-{csv_file}', datanum=5000, angle=None)
            cost_1pe = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, f'HN-{label}-{csv_file}')
            if math.floor(math.log10(abs(cost_nf))) <= -9 and math.floor(math.log10(abs(cost_1pe))) <= -9:
                ft_gates_list.append(gate_name)
        print(f"Supported FT Gates for Code [{CODE.code_name}] using HN-training{noise_qubits}{csv_file} is {ft_gates_list}")


# Our FTVQC-I: Error-Bounded Training for FT Gates Search
def train_and_inference_for_error_bounded_training():
    n_layer = 1  # the number of ansatz layers
    if CODE.log_n_qubits == 1:
        vqc_shape = [n_layer, CODE.phy_n_qubits, 3]

    # gate_name_list = ['H', 'S', 'T', 'HS', 'SH', 'HSdagger', 'SdaggerH', 'HT', 'TH', 'HTdagger', 'TdaggerH', 'ST', 'SdaggerTdagger']
    gate_name_list = ['H']
    ft_gates_list = []
    for gate_name in gate_name_list:
        task_name = f"Error-Bounded VQC Training for {gate_name} Gate"
        print(f"Task: {task_name}")
        trained_weights = train_and_test_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, Logical_Circuit, gate_name, task_name, vqc_shape, NesterovMomentumOptimizer, angle=None, epochs=50, data_num=200, lr=0.5, noise_level=None)
        cost_nf = test_under_noise_free_environment(gate_name, trained_weights, 'EB Training', datanum=5000, angle=None)
        cost_1pe = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, 'EB Training')
        if math.floor(math.log10(abs(cost_nf))) <= -9 and math.floor(math.log10(abs(cost_1pe))) <= -9:
            ft_gates_list.append(gate_name)
    print(f"Supported FT Gates for Code [{CODE.code_name}] using EB-training is {ft_gates_list}")


def FT_gates_search():
    # todo: for each QEC code, you need to change the variable CODE in a_circuit.py file
    start_time = datetime.now()
    train_and_inference_for_noise_free_training()
    print("Execution time (s): ", (datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    train_and_inference_for_given_noise_qubits()
    print("Execution time (s): ", (datetime.now() - start_time).total_seconds())

    start_time = datetime.now()
    train_and_inference_for_error_bounded_training()
    print("Execution time (s): ", (datetime.now() - start_time).total_seconds())



if __name__ == '__main__':
    out_file = './results/Stage1_FT_Gates_Search_' + str(datetime.now()) + '.txt'

    with open(out_file, 'w') as f, redirect_stdout(f):
        start_time = datetime.now()
        print(start_time)
        FT_gates_search()
        end_time = datetime.now()
        print(end_time)
        print("Execution time (s): ", (end_time - start_time).total_seconds())






