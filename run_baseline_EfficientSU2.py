import pennylane as qml
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.circuit.library import EfficientSU2
from matplotlib import pyplot as plt
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
import numpy as np
from qiskit_algorithms import VQE
from datetime import datetime
from contextlib import redirect_stdout


# Molecule Pauli String
pauli_string_H2 = [
    ("II", -1.052373245772859),
    ("IZ", 0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", 0.18093119978423156)]
H2_op = SparsePauliOp.from_list(pauli_string_H2)

pauli_string_LiH = [
    ("IIII", -7.4984),
    ("IIZI", 0.9885),
    ("IZII", -0.2318),
    ("ZIII", -0.3270),
    ("IIIZ", -0.5431),
    ("IIZZ", 0.3311),
    ("IZIZ", 0.0479),
    ("IZZI", 0.0673),
    ("ZIIZ", 0.5712),
    ("ZIZI", 0.2351),
    ("ZZII", 0.2321),
    ("XXYY", 0.1225),
    ("YYXX", 0.1225),
    ("XYXY", -0.1744),
    ("YXYX", -0.1744)]
LiH_op = SparsePauliOp.from_list(pauli_string_LiH)

pauli_string_CO2 = [
    ("IIII", -0.73604741),
    ("IIIZ", 0.12480727),
    ("IIZI", -0.13646287),
    ("IIZZ", 0.11658726),
    ("IZII", 0.12480727),
    ("IZIZ", 0.14273505),
    ("IZZI", 0.12152000),
    ("XXXX", 0.00493274),
    ("XXYY", 0.00493274),
    ("YYXX", 0.00493274),
    ("YYYY", 0.00493274),
    ("ZIII", -0.13646287),
    ("ZIIZ", 0.12152000),
    ("ZIZI", 0.14040905),
    ("ZZII", 0.11658726)]
CO2_op = SparsePauliOp.from_list(pauli_string_CO2)


# Clifford + T Decomposition (https://docs.pennylane.ai/en/stable/code/api/pennylane.clifford_t_decomposition.html)
def translate_qiskit_to_qml_and_decompose(ansatz, params):
    # EfficientSU2 Circuit
    dev = qml.device("default.qubit", wires=ansatz.num_qubits)
    @qml.qnode(dev)
    def circuit_wrapper(params):
        for j in range(ansatz.reps+1):
            for i in range(ansatz.num_qubits):
                qml.RY(params[2*j*ansatz.num_qubits + i], wires=i)
                qml.RZ(params[(2*j+1)*ansatz.num_qubits + i], wires=i)
                if (j < ansatz.reps) and (i < ansatz.num_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
        return qml.state()
    print("Optimized EfficientSU2 Circuit: ")
    print(qml.draw(circuit_wrapper)(params))

    # clifford + T decomposition
    decomposed_circuit = qml.transforms.clifford_t_decomposition(circuit_wrapper, epsilon=1e-03)
    specs = qml.specs(decomposed_circuit)(params)
    print("Decomposed Circuit Resources:")
    print(f"{specs['resources']}")


# Rotation-based VQE Training
def rotation_VQE_training_SPSA(ansatz, molecule_operator, iterations):
    # build Estimator
    estimator_1 = AerEstimator(
        run_options={"seed": algorithm_globals.random_seed, "shots": 1000},
        transpile_options={"seed_transpiler": algorithm_globals.random_seed}
    )

    # optimizer parameters
    learning_rate_array = np.linspace(0.5, 0.01, iterations)
    perturbation_array = np.linspace(0.1, 0.05, iterations)

    # record the optimization history
    spsa_loss_history = []  # SPSA: Simultaneous Perturbation Stochastic Approximation
    spsa_iteration = {"count": 0}
    def spsa_callback_1(num_evaluations: int, params: np.ndarray, function_value: float, stepsize: float, accepted: bool):
        if accepted:
            print(f"Iteration [{spsa_iteration['count']}] Energy: {function_value:.6f}")
            spsa_loss_history.append(function_value)
            spsa_iteration["count"] += 1

    # build optimizer
    optimizer = SPSA(
        maxiter=iterations,
        learning_rate=learning_rate_array,
        perturbation=perturbation_array,
        callback=spsa_callback_1
    )

    # run VQE
    vqe = VQE(
        estimator=estimator_1,
        ansatz=ansatz,
        optimizer=optimizer
    )
    result = vqe.compute_minimum_eigenvalue(operator=molecule_operator)
    print(f"Optimization Result of Ansatz: {result}")

    # decompose circuit and count gate num
    params = [result.optimal_parameters[p] for p in ansatz.parameters]
    translate_qiskit_to_qml_and_decompose(ansatz, params)
    return spsa_loss_history


# Plot Training Curve
def plot_training_curve(spsa_loss_history, ref_value):
    plt.figure(figsize=(15, 5))
    plt.plot(range(1, len(spsa_loss_history) + 1), spsa_loss_history, marker="o", linestyle="-", color="b", label=f"Optimization Value ({min(spsa_loss_history):.6f})")
    plt.axhline(y=ref_value, color='r', linestyle='--', label=f"Reference Value ({ref_value:.6f})")
    plt.text(0, ref_value, f'{ref_value:.6f}', color='r', verticalalignment='bottom', horizontalalignment='left')
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()
    plt.show()


# Baseline Test Function
def rotation_VQE_test():
    # Molecule operators
    molecule_operator_list = [H2_op, LiH_op, CO2_op]

    for molecule_operator in molecule_operator_list:
        print(f"Paili Operator: {molecule_operator}")

        # Print the optimal reference energy
        numpy_solver = NumPyMinimumEigensolver()
        result = numpy_solver.compute_minimum_eigenvalue(operator=molecule_operator)
        ref_value = result.eigenvalue.real
        print(f"Reference Energy Value: {ref_value:.5f}")

        # SPSA optimization for rotation VQE
        iterations = 150
        ansatz = EfficientSU2(molecule_operator.num_qubits)
        ansatz.decompose().draw(output="mpl", style="default", fold=20)
        plt.show()
        spsa_loss_history = rotation_VQE_training_SPSA(ansatz, molecule_operator, iterations)
        plot_training_curve(spsa_loss_history, ref_value)
        print('-' * 40)


if __name__ == '__main__':
    out_file = './results/Baseline_EfficientSU2_' + str(datetime.now()) + '.txt'

    with open(out_file, 'w') as f, redirect_stdout(f):
        start_time = datetime.now()
        print(start_time)
        rotation_VQE_test()
        end_time = datetime.now()
        print(end_time)
        print("Execution time (s): ", (end_time - start_time).total_seconds())