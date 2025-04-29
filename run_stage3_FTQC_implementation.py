from collections import Counter
import pennylane as qml
from a_trained_weights_713 import HS_7_1_3_trained_weights, SH_7_1_3_trained_weights
from a_trained_weights_513 import HS_5_1_3_weights, HSdagger_5_1_3_weights
import numpy as np
from functools import reduce
import operator
from run_baseline_EfficientSU2 import pauli_string_H2
from a_qec_code import QEC_Code
from datetime import datetime
from contextlib import redirect_stdout


# ------------Molecular Pauli String--------------------
def pauli_string_to_operator(pauli_str, wires):
    ops = []
    for i, p in enumerate(pauli_str):
        if p == "I":
            ops.append(qml.Identity(wires[i]))
        elif p == "X":
            ops.append(qml.PauliX(wires[i]))
        elif p == "Y":
            ops.append(qml.PauliY(wires[i]))
        elif p == "Z":
            ops.append(qml.PauliZ(wires[i]))
    return ops

def generate_Pauli_String(molecular_pauli_string, wires):
    coeffs = np.array([coeff for _, coeff in molecular_pauli_string])
    pauli_ops = [pauli_string_to_operator(ps, wires) for ps, _ in molecular_pauli_string] # [[op1, op2, op3], [], ....]
    combined_ops = [reduce(operator.matmul, ops) for ops in pauli_ops] # [op1 @ op2 @ op3, ......]

    H = qml.Hamiltonian(coeffs, combined_ops) # Hamiltonian
    return H


# -------------Logical Circuit for H2 Molecule---------------
H2_Hamiltonian_logical = generate_Pauli_String(pauli_string_H2, wires=[0, 1])
dev_H2_713 = qml.device('default.qubit', wires=2)
@qml.qnode(dev_H2_713)
def logical_circuit_H2_713():
    #      ┌───┐┌───┐┌───┐┌───┐
    # q_1: ┤ H ├┤ S ├┤ S ├┤ H ├
    #      └───┘└───┘└───┘└───┘
    # q_0: ────────────────────
    qml.Identity(wires=0)
    qml.Hadamard(wires=1)
    qml.S(wires=1)
    qml.S(wires=1)
    qml.Hadamard(wires=1)
    return qml.density_matrix(wires=1), qml.expval(H2_Hamiltonian_logical)

dev_H2_513 = qml.device('default.qubit', wires=2)
@qml.qnode(dev_H2_513)
def logical_circuit_H2_513():
    #      ┌───┐┌─────┐┌───┐┌───┐┌───┐┌─────┐
    # q_1: ┤ H ├┤ Sdg ├┤ H ├┤ S ├┤ H ├┤ Sdg ├
    #      └───┘└─────┘└───┘└───┘└───┘└─────┘
    # q_0: ──────────────────────────────────
    qml.Identity(wires=0)

    qml.Hadamard(wires=1)
    qml.adjoint(qml.S)(wires=1)
    qml.Hadamard(wires=1)
    qml.S(wires=1)
    qml.Hadamard(wires=1)
    qml.adjoint(qml.S)(wires=1)
    return qml.density_matrix(wires=1), qml.expval(H2_Hamiltonian_logical)


# --------------Build QEC-based FTQC for H2 (Noise-free, 1 physical error)---------------
# todo: need to be changed when different evaluation are executed
CODE_NAME = '7_1_3'
# CODE_NAME = '5_1_3'
print(f"QEC Code {CODE_NAME} for FTQC Implementation")
code1_H2 = QEC_Code(CODE_NAME, 0)
code2_H2 = QEC_Code(CODE_NAME, code1_H2.total_qubits)

H2_Hamiltonian_phy = generate_Pauli_String(pauli_string_H2, wires=code1_H2.space_A+code2_H2.space_A)
trash_qubits = code2_H2.anc_n_qubits # for qubit reset
dev_phy_H2_713 = qml.device('default.qubit', wires=code1_H2.total_qubits+code2_H2.total_qubits+trash_qubits)
@qml.qnode(dev_phy_H2_713)
def QEC_circuit_H2_713(env):
    # Gates from different training method
    trained_weights_HS, trained_weights_SH = HS_7_1_3_trained_weights, SH_7_1_3_trained_weights

    # Circuit
    code1_H2.encoder()
    code2_H2.encoder()

    if env == 'NF':  # noise-free environment
        code2_H2.ansatz(trained_weights_HS)
    elif env == '1PE':  # 1 physical error environment
        code2_H2.random_fixed_number_noisy_ansatz(trained_weights_HS, n_errors=1)
    code1_H2.checker()
    code2_H2.checker()
    code1_H2.corrector()
    code2_H2.corrector()
    # reset syndrome qubits
    for i, w in enumerate(range(code1_H2.total_qubits+code2_H2.total_qubits, code1_H2.total_qubits+code2_H2.total_qubits+trash_qubits)):
        qml.SWAP(wires=[w, code2_H2.start_qubit_idx+code2_H2.phy_n_qubits+i])

    if env == 'NF':  # noise-free environment
        code2_H2.ansatz(trained_weights_SH)
    elif env == '1PE':  # 1 physical error environment
        code2_H2.random_fixed_number_noisy_ansatz(trained_weights_SH, n_errors=1)
    code1_H2.checker()
    code2_H2.checker()
    code1_H2.corrector()
    code2_H2.corrector()

    code1_H2.decoder()
    code2_H2.decoder()

    return qml.density_matrix(wires=code2_H2.start_qubit_idx), qml.expval(H2_Hamiltonian_phy)


dev_phy_H2_513 = qml.device('default.qubit', wires=code1_H2.total_qubits + code2_H2.total_qubits + 2 * trash_qubits)
@qml.qnode(dev_phy_H2_513)
def QEC_circuit_H2_513(env):
    # Trained weights from different training method
    trained_weights_HS, trained_weights_HSdagger = HS_5_1_3_weights, HSdagger_5_1_3_weights

    # Circuit
    code1_H2.encoder()
    code2_H2.encoder()

    if env == 'NF':  # noise-free environment
        code2_H2.ansatz(trained_weights_HSdagger)
    elif env == '1PE':  # 1 physical error environment
        code2_H2.random_fixed_number_noisy_ansatz(trained_weights_HSdagger, n_errors=1)
    code1_H2.checker()
    code2_H2.checker()
    code1_H2.corrector()
    code2_H2.corrector()
    # reset syndrome qubits
    for i, w in enumerate(range(code1_H2.total_qubits+code2_H2.total_qubits, code1_H2.total_qubits+code2_H2.total_qubits+trash_qubits)):
        qml.SWAP(wires=[w, code2_H2.start_qubit_idx+code2_H2.phy_n_qubits+i])

    if env == 'NF':  # noise-free environment
        code2_H2.ansatz(trained_weights_HS)
    elif env == '1PE':  # 1 physical error environment
        code2_H2.random_fixed_number_noisy_ansatz(trained_weights_HS, n_errors=1)
    code1_H2.checker()
    code2_H2.checker()
    code1_H2.corrector()
    code2_H2.corrector()
    # reset syndrome qubits
    for i, w in enumerate(range(code1_H2.total_qubits+code2_H2.total_qubits+trash_qubits, code1_H2.total_qubits+code2_H2.total_qubits+2*trash_qubits)):
        qml.SWAP(wires=[w, code2_H2.start_qubit_idx+code2_H2.phy_n_qubits+i])

    if env == 'NF':  # noise-free environment
        code2_H2.ansatz(trained_weights_HSdagger)
    elif env == '1PE':  # 1 physical error environment
        code2_H2.random_fixed_number_noisy_ansatz(trained_weights_HSdagger, n_errors=1)
    code1_H2.checker()
    code2_H2.checker()
    code1_H2.corrector()
    code2_H2.corrector()

    code1_H2.decoder()
    code2_H2.decoder()

    return qml.density_matrix(wires=code2_H2.start_qubit_idx), qml.expval(H2_Hamiltonian_phy)


# -----------------------FTVQC Implementation-----------------------------
def FTQC_implementation(env='1PE'):
    if CODE_NAME == '7_1_3':
        print(f"Code: {CODE_NAME}")
        log_dm, log_energy = logical_circuit_H2_713()
        print(log_dm)
        print(f"1. Energy of Logical Circuit: {log_energy}")

        dm, energy = QEC_circuit_H2_713(env)
        # number of gates
        ops = QEC_circuit_H2_713.qtape.operations
        gate_names = [op.name for op in ops]
        gate_counts = Counter(gate_names)
        total_gate_count = sum(gate_counts.values())
        print(f'Gate Count: {total_gate_count}, Dict: {gate_counts}')
        # fidelity
        fidelity_error = 1.0 - qml.math.fidelity(log_dm, dm)
        print(dm, fidelity_error)
        print(f"2. Energy using {CODE_NAME} code {env} env.: {energy}")
        print('-' * 40)
    if CODE_NAME == '5_1_3':
        print(f"Code: {CODE_NAME}")
        log_dm, log_energy = logical_circuit_H2_513()
        print(log_dm)
        print(f"1. Energy of Logical Circuit: {log_energy}")

        dm, energy = QEC_circuit_H2_513(env)
        # number of gates
        ops = QEC_circuit_H2_513.qtape.operations
        gate_names = [op.name for op in ops]
        gate_counts = Counter(gate_names)
        total_gate_count = sum(gate_counts.values())
        print(f'Gate Count: {total_gate_count}, Dict: {gate_counts}')
        # fidelity
        fidelity_error = 1.0 - qml.math.fidelity(log_dm, dm)
        print(dm, fidelity_error)
        print(f"2. Energy using {CODE_NAME} code under {env} env.: {energy}")
        print('-' * 40)


if __name__ == '__main__':
    out_file = './results/Stage3_FTQC_Implementation_H2_' + str(datetime.now()) + '.txt'

    with open(out_file, 'w') as f, redirect_stdout(f):
        start_time = datetime.now()
        print(start_time)
        FTQC_implementation()
        end_time = datetime.now()
        print(end_time)
        print("Execution time (s): ", (end_time - start_time).total_seconds())