from a_vqc_funcs import dataset, test
from a_circuits import CODE, Logical_Circuit, Physical_Circuit_wo_Noise, \
    Physical_Circuit_with_Given_Each_Qubit_Error_Probability, Physical_Circuit_with_Fixed_1_Random_Noise
from pennylane import numpy as pnp
from a_vqc_noise_training import compute_noise_level_for_given_noise_qubits, train_VQC_with_given_noise_qubits
from datetime import datetime
from contextlib import redirect_stdout
from a_trained_weights_713 import H_7_1_3_trained_weights, S_7_1_3_trained_weights, trained_weights_wo_noise_H_7_1_3, trained_weights_wo_noise_S_7_1_3
from a_trained_weights_513 import HS_5_1_3_weights, HSdagger_5_1_3_weights, \
    trained_weights_wo_noise_HS_5_1_3, trained_weights_wo_noise_HSdagger_5_1_3

'''
Noise Levels	Noise Qubits/Avg Error per Qubit
IBM-BXL-O1	[67, 38, 2, 40, 43, 116, 100]
IBM-BXL-O2	[28, 17, 85, 99, 101, 70, 18]
IBM-BXL-O3	[109, 16, 58, 115, 120, 97, 77]
IBM-BXL-T1	[78, 79, 80, 91, 97, 98, 99]
IBM-BXL-T2	[57, 58, 59, 71, 76, 77, 78]
IBM-BXL-T3	[27, 28, 29, 35, 46, 47, 48]
IBM-BXL-R1	[21, 45, 63, 92, 119, 68, 36]
IBM-BXL-R2	[58, 77, 86, 47, 28, 75, 105]
IBM-BXL-R3	[90, 97, 102, 55, 18, 13, 50]
IBM-SXB-O1	[54, 85, 7, 121, 36, 58, 4]
IBM-SXB-O2	[100, 56, 47, 13, 115, 75, 112]
IBM-SXB-O3	[37, 126, 91, 28, 97, 105, 38]
IBM-SXB-T1	[19, 20, 21, 33, 38, 39, 40]
IBM-SXB-T2	[61, 62, 63, 72, 80, 81, 82]
IBM-SXB-T3	[23, 24, 25, 34, 42, 43, 44]
IBM-SXB-R1	[23, 28, 65, 86, 69, 97, 91]
IBM-SXB-R2	[12, 77, 38, 73, 74, 108, 114]
IBM-SXB-R3	[59, 45, 68, 1, 19, 91, 48]
IBM-BXL-O4	[116, 100, 3, 63, 60]
IBM-BXL-O5	[103, 68, 6, 46, 87]
IBM-BXL-O6	[58, 115, 120, 97, 77]
IBM-BXL-T4	[71, 77, 78, 79, 91]
IBM-BXL-T5	[109, 96, 97, 98, 91]
IBM-BXL-T6	[16, 26, 27, 28, 35]
IBM-BXL-R4	[18, 21, 33, 84, 63]
IBM-BXL-R5	[28, 45, 63, 72, 109]
IBM-BXL-R6	[97, 77, 31, 70, 48]
IBM-SXB-O4	[119, 35, 50, 122, 55]
IBM-SXB-O5	[69, 64, 33, 19, 78]
IBM-SXB-O6	[91, 28, 97, 105, 38]
IBM-SXB-T4	[33, 39, 38, 37, 52]
IBM-SXB-T5	[72, 62, 63, 64, 54]
IBM-SXB-T6	[15, 22, 23, 24, 34]
IBM-SXB-R4	[28, 84, 62, 91, 109]
IBM-SXB-R5	[85, 30, 23, 103, 93]
IBM-SXB-R6	[114, 28, 35, 58, 70]
IonQ-Forte	0.0002
IonQ-Aria2	0.0003
'''


# Test unider Noise-free Environment
def test_under_noise_free_environment(gate_name, trained_weight, label, datanum, angle=None):
    train_x, train_y, test_x, test_y = dataset(Logical_Circuit, gate_name, datanum, angle, cx_flag=None)
    task_name = f"Model Test for {gate_name} Gate under Noise-free Environment using {label}"
    costs = test(test_x, Physical_Circuit_wo_Noise, trained_weight, test_y, task_name)
    average_cost = pnp.mean(costs)
    print(f"\nAverage cost is {average_cost}")
    return average_cost


# Test under Given Physical Circuit
def test_under_given_physical_circuit(phy_circuit, circuit_name, gate_name, trained_weight, label):
    train_x, train_y, test_x, test_y = dataset(Logical_Circuit, gate_name, data_num=5000, angle=None, cx_flag=None)
    task_name = f"Model Test for {gate_name} Gate under {circuit_name} Circuit using {label}"
    costs = test(test_x, phy_circuit, trained_weight, test_y, task_name)
    average_cost = pnp.mean(costs)
    print(f"\nAverage cost is {average_cost}")
    return average_cost


# Test under Given Noise Qubits
def test_under_given_noise_qubits(gate_name, trained_weights, label, noise_qubits, csv_file, keywords, data_num):
    train_x, train_y, test_x, test_y = dataset(Logical_Circuit, gate_name, data_num, angle=None, cx_flag=None)
    task_name = f"Noise-aware Model Test for {gate_name} Gate under Given Noise Qubits {noise_qubits} using {label}"
    # compute noise level
    error_list = compute_noise_level_for_given_noise_qubits(csv_file, keywords, noise_qubits)
    costs = test(test_x, Physical_Circuit_with_Given_Each_Qubit_Error_Probability, trained_weights, test_y, task_name, error_list)
    average_cost = pnp.mean(costs)
    print(f"\nAverage cost is {average_cost}")
    return average_cost


# Test under Given Error Probability
def test_under_given_error_probability(gate_name, weights, label, error_list):
    train_x, train_y, test_x, test_y = dataset(Logical_Circuit, gate_name, data_num=5000, angle=None, cx_flag=None)
    task_name = f"Noise-aware Model Test for {gate_name} Gate under Given Error Probability {error_list} using {label}"
    costs = test(test_x, Physical_Circuit_with_Given_Each_Qubit_Error_Probability, weights, test_y, task_name, error_list)
    average_cost = pnp.mean(costs)
    print(f"\nAverage cost is {average_cost}")
    return average_cost


# test noise-free/error-bounded training model under different noise levels
def test_NF_EB_trained_model(trained_weights, gate_name, label, datanum=5000):
    avg_costs_list = []
    if trained_weights.shape[1] == 7:
        # test under noise-free environment
        avg_cost = test_under_noise_free_environment(gate_name, trained_weights, label, datanum, angle=None)
        avg_costs_list.append(avg_cost)
        # test under noise qubits, IBM-Brussels (IBM BXL)
        noise_qubits_bxl = [[67, 38, 2, 40, 43, 116, 100], [28, 17, 85, 99, 101, 70, 18], [109, 16, 58, 115, 120, 97, 77],
                            [78, 79, 80, 91, 97, 98, 99], [57, 58, 59, 71, 76, 77, 78], [27, 28, 29, 35, 46, 47, 48],
                            [21, 45, 63, 92, 119, 68, 36], [58, 77, 86, 47, 28, 75, 105], [90, 97, 102, 55, 18, 13, 50]]
        csv_file_bxl = './noise_data/ibm_brussels_calibrations_2025-03-18T10_07_09Z.csv'
        keywords = ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error']
        for noise_qubits in noise_qubits_bxl:
            avg_cost = test_under_given_noise_qubits(gate_name, trained_weights, label, noise_qubits, csv_file_bxl, keywords, datanum)
            avg_costs_list.append(avg_cost)
        # test under noise qubits, IBM-Strasbourg (IBM SXB)
        noise_qubits_sxb = [[54, 85, 7, 121, 36, 58, 4], [100, 56, 47, 13, 115, 75, 112], [37, 126, 91, 28, 97, 105, 38],
                            [19, 20, 21, 33, 38, 39, 40], [61, 62, 63, 72, 80, 81, 82], [23, 24, 25, 34, 42, 43, 44],
                            [23, 28, 65, 86, 69, 97, 91], [12, 77, 38, 73, 74, 108, 114], [59, 45, 68, 1, 19, 91, 48]]
        csv_file_sxb = './noise_data/ibm_strasbourg_calibrations_2025-03-19T18_37_44Z.csv'
        for noise_qubits in noise_qubits_sxb:
            avg_cost = test_under_given_noise_qubits(gate_name, trained_weights, label, noise_qubits, csv_file_sxb, keywords, datanum)
            avg_costs_list.append(avg_cost)
        # test under IonQ backends
        error_list_list = [[0.0002] * CODE.phy_n_qubits, [0.0003] * CODE.phy_n_qubits]
        for error_list in error_list_list:
            avg_cost = test_under_given_error_probability(gate_name, trained_weights, label, error_list)
            avg_costs_list.append(avg_cost)
        # test under 1 physical error environment
        avg_cost = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, label)
        avg_costs_list.append(avg_cost)
    elif trained_weights.shape[1] == 5:
        # test under noise-free environment
        avg_cost = test_under_noise_free_environment(gate_name, trained_weights, label, datanum, angle=None)
        avg_costs_list.append(avg_cost)
        # test under noise qubits, IBM-Brussels (IBM BXL)
        noise_qubits_bxl = [[116, 100, 3, 63, 60], [103, 68, 6, 46, 87], [58, 115, 120, 97, 77],
                            [71, 77, 78, 79, 91], [109, 96, 97, 98, 91], [16, 26, 27, 28, 35],
                            [18, 21, 33, 84, 63], [28, 45, 63, 72, 109], [97, 77, 31, 70, 48]]
        csv_file_bxl = './noise_data/ibm_brussels_calibrations_2025-03-18T10_07_09Z.csv'
        keywords = ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error']
        for noise_qubits in noise_qubits_bxl:
            avg_cost = test_under_given_noise_qubits(gate_name, trained_weights, label, noise_qubits, csv_file_bxl, keywords, datanum)
            avg_costs_list.append(avg_cost)
        # test under noise qubits, IBM-Strasbourg (IBM SXB)
        noise_qubits_sxb = [[119, 35, 50, 122, 55], [69, 64, 33, 19, 78], [91, 28, 97, 105, 38],
                            [33, 39, 38, 37, 52], [72, 62, 63, 64, 54], [15, 22, 23, 24, 34],
                            [28, 84, 62, 91, 109], [85, 30, 23, 103, 93], [114, 28, 35, 58, 70]]
        csv_file_sxb = './noise_data/ibm_strasbourg_calibrations_2025-03-19T18_37_44Z.csv'
        for noise_qubits in noise_qubits_sxb:
            avg_cost = test_under_given_noise_qubits(gate_name, trained_weights, label, noise_qubits, csv_file_sxb, keywords, datanum)
            avg_costs_list.append(avg_cost)
        # test under IonQ backends
        error_list_list = [[0.0002] * CODE.phy_n_qubits, [0.0003] * CODE.phy_n_qubits]
        for error_list in error_list_list:
            avg_cost = test_under_given_error_probability(gate_name, trained_weights, label, error_list)
            avg_costs_list.append(avg_cost)
        # test under 1 physical error environment
        avg_cost = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, label)
        avg_costs_list.append(avg_cost)
    print("Average Costs List: ", avg_costs_list)


# Train and Test hardware-specific noisy training method under different noise levels
def train_test_HN_trained_model(code_name, gate_name):
    avg_costs_list = []
    if code_name == '7_1_3':
        n_layer = 1  # the number of ansatz layers
        if CODE.log_n_qubits == 1:
            vqc_shape = [n_layer, CODE.phy_n_qubits, 3]
        # train and test using IBM-Brussels noise information
        noise_qubits_bxl = [[67, 38, 2, 40, 43, 116, 100], [28, 17, 85, 99, 101, 70, 18], [109, 16, 58, 115, 120, 97, 77],
                            [78, 79, 80, 91, 97, 98, 99], [57, 58, 59, 71, 76, 77, 78], [27, 28, 29, 35, 46, 47, 48],
                            [21, 45, 63, 92, 119, 68, 36], [58, 77, 86, 47, 28, 75, 105], [90, 97, 102, 55, 18, 13, 50]]
        csv_file_bxl = './noise_data/ibm_brussels_calibrations_2025-03-18T10_07_09Z.csv'
        keywords = ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error']

        label_list_bxl = [str(noise_qubits) for noise_qubits in noise_qubits_bxl]
        for noise_qubits, label in zip(noise_qubits_bxl, label_list_bxl):
            trained_weights = train_VQC_with_given_noise_qubits(noise_qubits, csv_file_bxl, keywords, gate_name, vqc_shape)  # training
            cost_nf = test_under_noise_free_environment(gate_name, trained_weights, f'HN-{label}-{csv_file_bxl}', datanum=5000, angle=None)
            cost_hn = test_under_given_noise_qubits(gate_name, trained_weights, f'HN-{label}-{csv_file_bxl}', noise_qubits, csv_file_bxl, keywords, data_num=5000)
            cost_1pe = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, f'HN-{label}-{csv_file_bxl}')
            avg_costs_list.append((cost_nf, cost_hn, cost_1pe))

        # train and test under IBM-Strabourg noise information
        noise_qubits_sxb = [[54, 85, 7, 121, 36, 58, 4], [100, 56, 47, 13, 115, 75, 112], [37, 126, 91, 28, 97, 105, 38],
                            [19, 20, 21, 33, 38, 39, 40], [61, 62, 63, 72, 80, 81, 82], [23, 24, 25, 34, 42, 43, 44],
                            [23, 28, 65, 86, 69, 97, 91], [12, 77, 38, 73, 74, 108, 114], [59, 45, 68, 1, 19, 91, 48]]
        csv_file_sxb = './noise_data/ibm_strasbourg_calibrations_2025-03-19T18_37_44Z.csv'
        label_list_sxb = [str(noise_qubits) for noise_qubits in noise_qubits_sxb]
        for noise_qubits, label in zip(noise_qubits_sxb, label_list_sxb):
            trained_weights = train_VQC_with_given_noise_qubits(noise_qubits, csv_file_sxb, keywords, gate_name, vqc_shape)  # training
            cost_nf = test_under_noise_free_environment(gate_name, trained_weights, f'HN-{label}-{csv_file_sxb}', datanum=5000, angle=None)
            cost_hn = test_under_given_noise_qubits(gate_name, trained_weights, f'HN-{label}-{csv_file_sxb}', noise_qubits, csv_file_sxb, keywords, data_num=5000)
            cost_1pe = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, f'HN-{label}-{csv_file_sxb}')
            avg_costs_list.append((cost_nf, cost_hn, cost_1pe))
    elif code_name == '5_1_3':
        n_layer = 1  # the number of ansatz layers
        if CODE.log_n_qubits == 1:
            vqc_shape = [n_layer, CODE.phy_n_qubits, 3]
        # train and test using IBM-Brussels noise information
        noise_qubits_bxl = [[116, 100, 3, 63, 60], [103, 68, 6, 46, 87], [58, 115, 120, 97, 77],
                            [71, 77, 78, 79, 91], [109, 96, 97, 98, 91], [16, 26, 27, 28, 35],
                            [18, 21, 33, 84, 63], [28, 45, 63, 72, 109], [97, 77, 31, 70, 48]]
        csv_file_bxl = './noise_data/ibm_brussels_calibrations_2025-03-18T10_07_09Z.csv'
        keywords = ['ID error', 'Z-axis rotation (rz) error', '(sx) error', 'Pauli-X error']

        label_list_bxl = [str(noise_qubits) for noise_qubits in noise_qubits_bxl]
        for noise_qubits, label in zip(noise_qubits_bxl, label_list_bxl):
            trained_weights = train_VQC_with_given_noise_qubits(noise_qubits, csv_file_bxl, keywords, gate_name, vqc_shape)  # training
            cost_nf = test_under_noise_free_environment(gate_name, trained_weights, f'HN-{label}-{csv_file_bxl}', datanum=5000, angle=None)
            cost_hn = test_under_given_noise_qubits(gate_name, trained_weights, f'HN-{label}-{csv_file_bxl}', noise_qubits, csv_file_bxl, keywords, data_num=5000)
            cost_1pe = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, f'HN-{label}-{csv_file_bxl}')
            avg_costs_list.append((cost_nf, cost_hn, cost_1pe))

        # train and test under IBM-Strabourg noise information
        noise_qubits_sxb = [[119, 35, 50, 122, 55], [69, 64, 33, 19, 78], [91, 28, 97, 105, 38],
                            [33, 39, 38, 37, 52], [72, 62, 63, 64, 54], [15, 22, 23, 24, 34],
                            [28, 84, 62, 91, 109], [85, 30, 23, 103, 93], [114, 28, 35, 58, 70]]
        csv_file_sxb = './noise_data/ibm_strasbourg_calibrations_2025-03-19T18_37_44Z.csv'
        label_list_sxb = [str(noise_qubits) for noise_qubits in noise_qubits_sxb]
        for noise_qubits, label in zip(noise_qubits_sxb, label_list_sxb):
            trained_weights = train_VQC_with_given_noise_qubits(noise_qubits, csv_file_sxb, keywords, gate_name, vqc_shape)  # training
            cost_nf = test_under_noise_free_environment(gate_name, trained_weights, f'HN-{label}-{csv_file_sxb}', datanum=5000, angle=None)
            cost_hn = test_under_given_noise_qubits(gate_name, trained_weights, f'HN-{label}-{csv_file_sxb}', noise_qubits, csv_file_sxb, keywords, data_num=5000)
            cost_1pe = test_under_given_physical_circuit(Physical_Circuit_with_Fixed_1_Random_Noise, 'Physical_Circuit_with_Fixed_1_Random_Noise', gate_name, trained_weights, f'HN-{label}-{csv_file_sxb}')
            avg_costs_list.append((cost_nf, cost_hn, cost_1pe))

    print("Average Costs List: ", avg_costs_list)


def test_NF_HN_EB():
    # todo: for each QEC code, you need to change the variable CODE in a_circuit.py file
    code_name = CODE.code_name
    if code_name == '7_1_3':
        gate_name = 'H'
        test_NF_EB_trained_model(trained_weights_wo_noise_H_7_1_3, gate_name, 'NF Training')
        train_test_HN_trained_model(code_name, gate_name)
        test_NF_EB_trained_model(H_7_1_3_trained_weights, gate_name, 'EB Training')
        gate_name = 'S'
        test_NF_EB_trained_model(trained_weights_wo_noise_S_7_1_3, gate_name, 'NF Training')
        train_test_HN_trained_model(code_name, gate_name)
        test_NF_EB_trained_model(S_7_1_3_trained_weights, gate_name, 'EB Training')
    if code_name == '5_1_3':
        gate_name = 'HS'
        test_NF_EB_trained_model(trained_weights_wo_noise_HS_5_1_3, gate_name, 'NF Training')
        train_test_HN_trained_model(code_name, gate_name)
        test_NF_EB_trained_model(HS_5_1_3_weights, gate_name, 'EB Training')
        gate_name = 'HSdagger'
        test_NF_EB_trained_model(trained_weights_wo_noise_HSdagger_5_1_3, gate_name, 'NF Training')
        train_test_HN_trained_model(code_name, gate_name)
        test_NF_EB_trained_model(HSdagger_5_1_3_weights, gate_name, 'EB Training')



if __name__ == '__main__':
    out_file = './results/ft_gates_inference_evaluation_' + str(datetime.now()) + '.txt'

    with open(out_file, 'w') as f, redirect_stdout(f):
        start_time = datetime.now()
        print(start_time)
        test_NF_HN_EB()
        end_time = datetime.now()
        print(end_time)
        print("Execution time (s): ", (end_time - start_time).total_seconds())