import pennylane as qml
from pennylane import numpy as pnp
from a_qec_code import QEC_Code
import numpy as np

# todo: need to be changed when different evaluation are executed
CODE = QEC_Code('5_1_3', 0)
# CODE = QEC_Code('7_1_3', 0)
print(f"QEC Code: {CODE.code_name}")


# Logical circuit for given gate_name, k qubits
dev_log = qml.device('default.qubit', wires=CODE.log_n_qubits)
@qml.qnode(dev_log)
def Logical_Circuit(alpha, gate_name, angle=None):
    CODE.state_init(alpha, dev_log)
    CODE.log_circuit(gate_name, angle)
    return qml.density_matrix(wires=list(range(CODE.log_n_qubits)))


# Physical circuit under noise-free environment, n data qubit + (n-k) ancillary qubits
dev_phy = qml.device('default.qubit', wires=CODE.total_qubits)
@qml.qnode(dev_phy)
def Physical_Circuit_wo_Noise(alpha, weights):
    CODE.state_init(alpha, dev_phy)
    CODE.encoder()
    CODE.ansatz(weights)
    CODE.checker()
    CODE.corrector()
    CODE.decoder()
    return qml.density_matrix(wires=CODE.space_A)


# Physical circuit with given gate-qubit pair (error list), n data qubit + (n-k) ancillary qubits
dev_phy_given_gate_qubit_noise_level = qml.device('default.qubit', wires=CODE.total_qubits)
@qml.qnode(dev_phy_given_gate_qubit_noise_level)
def Physical_Circuit_with_Given_Each_Qubit_Error_Probability(alpha, weights, error_list):
    CODE.state_init(alpha, dev_phy_given_gate_qubit_noise_level)
    CODE.encoder()
    CODE.ansatz_given_each_qubit_error_probability(weights, error_list)
    CODE.checker()
    CODE.corrector()
    CODE.decoder()
    return qml.density_matrix(wires=CODE.space_A)


# Physical circuit with fixed 1 random error, n data qubit + (n-k) ancillary qubits
dev_phy_fixed_1_random_noise = qml.device("default.qubit", wires=CODE.total_qubits)
@qml.qnode(dev_phy_fixed_1_random_noise)
def Physical_Circuit_with_Fixed_1_Random_Noise(alpha, weights):
    CODE.state_init(alpha, dev_phy_fixed_1_random_noise)
    CODE.encoder()
    CODE.random_fixed_number_noisy_ansatz(weights, n_errors=1)
    CODE.checker()
    CODE.corrector()
    CODE.decoder()
    return qml.density_matrix(wires=CODE.space_A)


# ----------------Physical Circuit for CX Gate Verification--------------------
code_name = '7_1_3'
if code_name == '7_1_3':
    code1 = QEC_Code('7_1_3', 0)
    code2 = QEC_Code('7_1_3', code1.total_qubits)
elif code_name == '5_1_3':
    code1 = QEC_Code('5_1_3', 0)
    code2 = QEC_Code('5_1_3', code1.total_qubits)

dev_log_cir_cx = qml.device("default.qubit", wires=2)
@qml.qnode(dev_log_cir_cx)
def Logical_Circuit_CX(alpha1, alpha2):
    theta1 = 2 * pnp.arccos(alpha1)  # alpha|0> + beta|1>
    qml.RY(theta1, wires=0)
    theta2 = 2 * pnp.arccos(alpha2)  # alpha|0> + beta|1>
    qml.RY(theta2, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.density_matrix(wires=[0]), qml.density_matrix(wires=[1])

def cx_ansatz(): # vanilla transversal CX
    for i in range(code1.start_qubit_idx, code1.phy_n_qubits):
        qml.CNOT(wires=[i, code2.start_qubit_idx+i])

dev_phy_cx = qml.device("default.qubit", wires=code1.total_qubits+code2.total_qubits)
@qml.qnode(dev_phy_cx)
def Physical_Circuit_CX_wo_Noise(alpha1, alpha2):
    code1.state_init(alpha1, dev_phy_cx)
    code2.state_init(alpha2, dev_phy_cx)
    code1.encoder()
    code2.encoder()
    cx_ansatz()
    code1.checker()
    code2.checker()
    code1.corrector()
    code2.corrector()
    code1.decoder()
    code2.decoder()
    return qml.density_matrix(wires=[code1.start_qubit_idx]), qml.density_matrix(wires=[code2.start_qubit_idx])

def verify_for_CX_Gate():
    print(f"Task: VQC Verification for CX Gate using {code1.code_name}")
    alpha1_list = np.linspace(0, 1, 500)
    # for alpha1, alpha2 in itertools.product([0, 1], repeat=2):
    fidelity_1_list, fidelity_2_list = [], []
    for alpha1 in alpha1_list:
        alpha2 = 1 - alpha1
        print(f"alpha1: {alpha1}, alpha2: {alpha2}")
        DM_truth = Logical_Circuit_CX(alpha1, alpha2)
        # print(f"True DM for Qubit 0 and 1: {DM_truth}")
        DM_output = Physical_Circuit_CX_wo_Noise(alpha1, alpha2)
        # print(f"Approximated DM for Qubit 0 and 1: {DM_output}")
        fidelity_error1 = 1.0 - qml.math.fidelity(DM_truth[0], DM_output[0])
        fidelity_error2 = 1.0 - qml.math.fidelity(DM_truth[1], DM_output[1])
        print(f"Fidelity Error for Qubit 0 and 1: {fidelity_error1}, {fidelity_error2}")
        fidelity_1_list.append(fidelity_error1)
        fidelity_2_list.append(fidelity_error2)
    print(f"Average Test Cost: {np.mean(np.array(fidelity_1_list))}, {np.mean(np.array(fidelity_2_list))}")
