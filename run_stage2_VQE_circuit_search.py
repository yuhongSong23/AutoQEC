from run_baseline_EfficientSU2 import H2_op, LiH_op, CO2_op
from qiskit_algorithms import NumPyMinimumEigensolver
from a_circuits import CODE
import random
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
import numpy as np
from datetime import datetime
from contextlib import redirect_stdout


# todo: for each QEC code, you need to change the variable CODE in a_circuit.py file
# Get GATE_POOL and GATE_SET based on FT Gate Search Method
def get_GATE_POOL_SET(training_method):
    # FT gate set using noise-free method is empty
    if CODE.code_name == '7_1_3':
        if training_method == 'EB': # error-bounded training
            GATE_POOL = [
                ("h", lambda qc, q: qc.h(q)), # Hadamard
                ("s", lambda qc, q: qc.s(q)), # S gate
                ("hs", lambda qc, q: (qc.h(q), qc.s(q))),
                ("sh", lambda qc, q: (qc.s(q), qc.h(q))),
                ("hsdagger", lambda qc, q: (qc.h(q), qc.sdg(q))),
                ("cx", lambda qc, q1, q2: qc.cx(q1, q2))
            ] # Steane Code
        elif training_method == 'HN_B1': # IBM_Brussels [27, 104, 72, 41, 122, 119, 52]
            GATE_POOL = [
                ("h", lambda qc, q: qc.h(q)),  # Hadamard
                ("hs", lambda qc, q: (qc.h(q), qc.s(q))),
                ("hsdagger", lambda qc, q: (qc.h(q), qc.sdg(q))),
                ("cx", lambda qc, q1, q2: qc.cx(q1, q2))
            ]
        elif training_method == 'HN_B2': # IBM_Strasbourg [118, 109, 113, 110, 70, 60, 48]
            GATE_POOL = [
                ("cx", lambda qc, q1, q2: qc.cx(q1, q2))
            ]
    elif CODE.code_name == '5_1_3':
        if training_method == 'EB':
            GATE_POOL = [
                ("hs", lambda qc, q: (qc.h(q), qc.s(q))),
                ("sh", lambda qc, q: (qc.s(q), qc.h(q))),
                ("hsdagger", lambda qc, q: (qc.h(q), qc.sdg(q))),
            ] # [5, 1, 3] code
        elif training_method == 'HN_B1': # IBM_Brussels [41, 122, 119, 52, 117]
            GATE_POOL = [
                ("hs", lambda qc, q: (qc.h(q), qc.s(q)))
            ]
        elif training_method == 'HN_B2': # IBM_Strasbourg [113, 110, 70, 60, 48]
            GATE_POOL = []
    GATE_SET = [item[0] for item in GATE_POOL] # extract gate set
    print(f"GATE_SET: {GATE_SET}")
    return GATE_POOL, GATE_SET


# Random Generate Circuit Gene
def random_gene(num_qubits, GATE_SET, num_layers=3):
    # random generate individual gene
    gene = []
    for _ in range(num_layers):
        for qubit in range(num_qubits):
            for idx in range(len(GATE_SET)):
                gate_avail = random.choice([0, 1])
                if gate_avail:
                    gene.append((idx, qubit))
    return gene


# Build Gene Circuit
def build_gene_circuit(gene, num_qubits, GATE_POOL):
    # from gene to quantum circuit
    circuit = QuantumCircuit(num_qubits)
    for (gate_idx, qubit) in gene:
        gate_name, gate_func = GATE_POOL[gate_idx]
        if gate_name == 'cx':
            gate_func(circuit, qubit, (qubit+1)%num_qubits)
        else:
            gate_func(circuit, qubit)
    return circuit


# Compute Gene Fitness to Evaluate the Circuit Performance (both Energy and Circuit Depth)
def fitness(gene, num_qubits, molecule_operator, ref_energy, ref_gates, GATE_SET, GATE_POOL): # with VQE energy and gates counts
    # compute the score for genes
    original_circuit = build_gene_circuit(gene, num_qubits, GATE_POOL)
    try:
        circuit = transpile(original_circuit, basis_gates=GATE_SET, optimization_level=1)
    except Exception as e:
        # print(f"Transpile failed, using original circuit.")
        circuit = original_circuit
    state = Statevector.from_instruction(circuit)  # compute circuit state
    energy = np.real(state.expectation_value(molecule_operator))  # compute expectation ⟨ψ|H|ψ⟩

    gate_counts = circuit.count_ops()
    total_gate_num = sum(gate_counts.values())
    if total_gate_num == 0:
        fitness = 0
    else:
        fitness = 0.9 * (energy / ref_energy) + 0.1 * (1.0 - total_gate_num / ref_gates)
    return energy, total_gate_num, fitness, circuit


# --------------Random Circuit Search for VQE Applications---------------
def random_search_space_exploration(training_method, pop_size=15000):
    molecule_operator_list = [H2_op, LiH_op, CO2_op]
    num_layers = 5

    for molecule_operator in molecule_operator_list:
        search_history = []
        # Print the optimal reference energy
        numpy_solver = NumPyMinimumEigensolver()
        result = numpy_solver.compute_minimum_eigenvalue(operator=molecule_operator)
        ref_value = result.eigenvalue.real
        print(f"Reference Energy Value: {ref_value:.5f}")

        # Random search for the optimal circuit
        num_qubits = molecule_operator.num_qubits
        GATE_POOL, GATE_SET = get_GATE_POOL_SET(training_method)
        ref_gate_depth = len(GATE_POOL) * num_qubits * num_layers
        random_population = [random_gene(num_qubits, GATE_SET, num_layers) for _ in range(pop_size)]
        results = [fitness(gene, num_qubits, molecule_operator, ref_value, ref_gate_depth, GATE_SET, GATE_POOL) for gene in random_population]

        energys, gate_depths, fitness_scores, circuits = zip(*results)
        max_index = fitness_scores.index(max(fitness_scores))
        max_fitness, max_energy, max_depth, max_circuit = fitness_scores[max_index], energys[max_index], gate_depths[max_index], circuits[max_index]
        print(f"best fitness: {max_fitness}, best energy: {max_energy}, best_depth: {max_depth}")
        print(max_circuit.draw())
        for fit, energy, gate, circuit in zip(fitness_scores, energys, gate_depths, circuits):
            if (fit, energy, gate) not in search_history:
                search_history.append((fit, energy, gate))
        print(f"Search History: {search_history}")
        print('-' * 40)


# --------------Evolutionary Circuit Search for VQE Applications---------------
# Gene crossover
def crossover(gene1, gene2):
    cut = random.randint(1, min(len(gene1), len(gene2)))
    child = gene1[:cut] + gene2[cut:]
    return child

# Gene mutation
def mutation(gene, num_qubits, GATE_SET):
    mutation_type = random.choice(["replace", "insert", "delete"])
    mutate_pos = random.randint(0, len(gene) - 1)

    if mutation_type == "replace":
        gene[mutate_pos] = random.randint(0, len(GATE_SET) - 1), random.randint(0, num_qubits - 1)
    elif mutation_type == "insert":
       gene.insert(mutate_pos, (random.randint(0, len(GATE_SET) - 1), random.randint(0, num_qubits - 1)))
    elif mutation_type == "delete" and len(gene) > 1:
        del gene[mutate_pos]

    return gene

def evolutionary_search(molecule_operator, num_qubits, num_layers, iterations, ref_energy, GATE_SET, GATE_POOL):
    pop_size = 100
    cross_rate = 0.7
    mutate_rate = 0.3
    ref_gate_depth = len(GATE_POOL) * num_qubits * num_layers

    search_history = []
    population = [random_gene(num_qubits, GATE_SET, num_layers) for _ in range(pop_size)]
    for iter in range(iterations):
        print(f"Iterations [{iter}] ", end='')

        results = [fitness(gene, num_qubits, molecule_operator, ref_energy, ref_gate_depth, GATE_SET, GATE_POOL) for gene in population]
        energys, gate_depths, fitness_scores, circuits = zip(*results)
        best_index = fitness_scores.index(max(fitness_scores))
        max_fitness, max_energy, max_depth, max_circuit = fitness_scores[best_index], energys[best_index], gate_depths[best_index], circuits[best_index]
        print(f"best fitness: {max_fitness}, best energy: {max_energy}, best_depth: {max_depth}")
        print(max_circuit.draw())
        search_history.append((max_fitness, max_energy, max_depth))

        # selection
        sorted_population = [gene for _, gene in sorted(zip(fitness_scores, population), reverse=True)]
        selected_pop = sorted_population[:pop_size//2] # select the best half

        next_generation = []
        # crossover
        while len(next_generation) < pop_size:
            if random.random() < cross_rate:
                gene1, gene2 = random.sample(selected_pop, 2)
                new_gene = crossover(gene1, gene2)
                next_generation.append(new_gene)
        # mutation
        for i in range(len(next_generation)):
            if random.random() < mutate_rate:
                next_generation[i] = mutation(next_generation[i], num_qubits, GATE_SET)
        population = next_generation

    return search_history

def evolutionary_test(training_method, iterations=150):
    # Molecule operators
    molecule_operator_list = [H2_op, LiH_op, CO2_op]
    molecule_name_list = ['H2', 'LiH', 'CO2']
    for molecule_operator, molecule_name in zip(molecule_operator_list, molecule_name_list):
        print(f"molecular {molecule_name} Paili operator: {molecule_operator}")

        # Print the optimal reference energy
        numpy_solver = NumPyMinimumEigensolver()
        result = numpy_solver.compute_minimum_eigenvalue(operator=molecule_operator)
        ref_value = result.eigenvalue.real
        print(f"Reference Energy Value: {ref_value:.5f}")

        # Evolutionary Search
        num_layers = 5
        GATE_POOL, GATE_SET = get_GATE_POOL_SET(training_method)
        if len(GATE_SET) != 0:
            search_history = evolutionary_search(molecule_operator, molecule_operator.num_qubits, num_layers, iterations, ref_value, GATE_SET, GATE_POOL)
            print(f"Best Result: {search_history[-1]}")
            print(f"Energy History: {search_history}")
        else:
            print("GATE_SET is Empty!")


def circuit_search_test():
    training_method_list = ['HN_B1', 'HN_B2', 'EB']
    for training_method in training_method_list:
        start_time = datetime.now()
        random_search_space_exploration(training_method)
        print("Execution time (s): ", (datetime.now() - start_time).total_seconds())

        start_time = datetime.now()
        evolutionary_test(training_method)
        print("Execution time (s): ", (datetime.now() - start_time).total_seconds())


if __name__ == '__main__':
    out_file = './results/Stage2_VQE_Circuit_Search_' + str(datetime.now()) + '.txt'

    with open(out_file, 'w') as f, redirect_stdout(f):
        start_time = datetime.now()
        print(start_time)
        circuit_search_test()
        end_time = datetime.now()
        print(end_time)
        print("Execution time (s): ", (end_time - start_time).total_seconds())
