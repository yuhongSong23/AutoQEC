import pennylane as qml
from pennylane import numpy as pnp
import random
import numpy as np

def binary_to_pauli(row): # based on code generator matrix to generate the stabilizer
    n = len(row) // 2
    pauli_str = ""
    for i in range(n):
        if row[i] == 1 and row[i + n] == 0:
            pauli_str += "X"
        elif row[i] == 0 and row[i + n] == 1:
            pauli_str += "Z"
        else:
            pauli_str += "I"
    return pauli_str


class QEC_Code:
    def __init__(self, code_name, start_qubit_idx):
        self._init_by_code(code_name, start_qubit_idx)

    def _init_by_code(self, code_name, start_qubit_idx): # code information
        self.code_name = code_name
        self.start_qubit_idx = start_qubit_idx
        if self.code_name == '7_1_3': # Steane code
            self.phy_n_qubits = 7 # [0， 1， 2， 3， 4]
            self.log_n_qubits = 1
            self.distance = 3
            self.space_A = [self.start_qubit_idx]  # logical space
            self.generator_matrix = np.array([
                [0, 1, 1, 1, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0], # X
                [1, 0, 1, 1, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0], # X
                [1, 1, 0, 1, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0], # X
                [0, 0, 0, 0, 0, 0, 0,  0, 1, 1, 1, 1, 0, 0], # Z
                [0, 0, 0, 0, 0, 0, 0,  1, 0, 1, 1, 0, 1, 0], # Z
                [0, 0, 0, 0, 0, 0, 0,  1, 1, 0, 1, 0, 0, 1]]) # Z
            self.stabilizer = [binary_to_pauli(row) for row in self.generator_matrix] # {'IXXXXII', 'XIXXIXI', 'XXIXIIX', 'IZZZZII', 'ZIZZIZI', 'ZZIZIIZ'}
            self.syndrome_table = self.compute_syndrome_table()
            self.anc_n_qubits = self.phy_n_qubits - self.log_n_qubits  # n-k
            self.total_qubits = self.phy_n_qubits + self.anc_n_qubits
        elif self.code_name == '5_1_3': # 5,1,3 Perfect code
            self.phy_n_qubits = 5
            self.log_n_qubits = 1
            self.distance = 3
            self.space_A = [self.start_qubit_idx]
            self.generator_matrix = np.array([
                [0, 0, 1, 1, 0,  0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1,  1, 0, 1, 0, 0],
                [1, 0, 0, 0, 1,  0, 1, 0, 1, 0],
                [1, 1, 0, 0, 0,  0, 0, 1, 0, 1]])
            self.stabilizer = [binary_to_pauli(row) for row in self.generator_matrix]# {'IZXXZ', 'ZIZXX', 'XZIZX', 'XXZIZ'}
            self.syndrome_table = self.compute_syndrome_table()
            self.anc_n_qubits = self.phy_n_qubits - self.log_n_qubits  # n-k
            self.total_qubits = self.phy_n_qubits + self.anc_n_qubits
        else:
            raise ValueError("Unsupported QEC code!")

    def reset(self, code_name, start_qubit_idx):
        self._init_by_code(code_name, start_qubit_idx)

    def state_init(self, alpha, dev):
        if self.log_n_qubits == 1:
            theta = 2 * pnp.arccos(alpha) # alpha|0> + beta|1>
            if len(dev.wires) == self.log_n_qubits: # logical circuit
                qml.RY(theta, wires=0)
            elif len(dev.wires) % self.total_qubits == 0: # physical circuit, integeter multiple
                qml.RY(theta, wires=self.space_A)
        else:
            raise ValueError("Unsupported logical number of qubits")

    def encoder(self):
        if self.code_name == '7_1_3':
            # Using 7 physical qubits to represent 1 logical qubit
            qml.CNOT(wires=[self.start_qubit_idx, self.start_qubit_idx+1])
            qml.CNOT(wires=[self.start_qubit_idx, self.start_qubit_idx+2])
            qml.Hadamard(wires=self.start_qubit_idx+4)
            qml.Hadamard(wires=self.start_qubit_idx+5)
            qml.Hadamard(wires=self.start_qubit_idx+6)

            qml.CNOT(wires=[self.start_qubit_idx+6, self.start_qubit_idx+3])
            qml.CNOT(wires=[self.start_qubit_idx+6, self.start_qubit_idx+1])
            qml.CNOT(wires=[self.start_qubit_idx+6, self.start_qubit_idx+0])

            qml.CNOT(wires=[self.start_qubit_idx+5, self.start_qubit_idx+3])
            qml.CNOT(wires=[self.start_qubit_idx+5, self.start_qubit_idx+2])
            qml.CNOT(wires=[self.start_qubit_idx+5, self.start_qubit_idx])

            qml.CNOT(wires=[self.start_qubit_idx+4, self.start_qubit_idx+3])
            qml.CNOT(wires=[self.start_qubit_idx+4, self.start_qubit_idx+2])
            qml.CNOT(wires=[self.start_qubit_idx+4, self.start_qubit_idx+1])
        elif self.code_name == '5_1_3':
            # Figure 10.16, https://lmsspada.kemdikbud.go.id/pluginfile.php/743625/mod_resource/content/1/quantum%20Computing%20-%20Nakahara.pdf
            # Using 5 Physical qubits to represet 1 logical qubit
            qml.PauliZ(wires=self.start_qubit_idx)

            qml.Hadamard(wires=self.start_qubit_idx+1)
            qml.CNOT(wires=[self.start_qubit_idx+1, self.start_qubit_idx])
            qml.CZ(wires=[self.start_qubit_idx+1, self.start_qubit_idx+2])
            qml.CZ(wires=[self.start_qubit_idx+1, self.start_qubit_idx+4])

            qml.Hadamard(wires=self.start_qubit_idx+4)
            qml.CNOT(wires=[self.start_qubit_idx+4, self.start_qubit_idx])
            qml.CZ(wires=[self.start_qubit_idx+4, self.start_qubit_idx+1])
            qml.CZ(wires=[self.start_qubit_idx+4, self.start_qubit_idx+3])

            qml.Hadamard(wires=self.start_qubit_idx+3)
            qml.CNOT(wires=[self.start_qubit_idx+3, self.start_qubit_idx+4])
            qml.CZ(wires=[self.start_qubit_idx+3, self.start_qubit_idx])
            qml.CZ(wires=[self.start_qubit_idx+3, self.start_qubit_idx+2])

            qml.Hadamard(wires=self.start_qubit_idx+2)
            qml.CNOT(wires=[self.start_qubit_idx+2, self.start_qubit_idx+3])
            qml.CZ(wires=[self.start_qubit_idx+2, self.start_qubit_idx+1])
            qml.CZ(wires=[self.start_qubit_idx+2, self.start_qubit_idx+4])
        else:
            raise ValueError("Unsupported QEC code!")

    def decoder(self):
        return qml.adjoint(self.encoder)()

    def checker(self): # error detection
        # utilize stabilizers to build the checker
        # [5, 1, 3] code: Figure 10.17, https://lmsspada.kemdikbud.go.id/pluginfile.php/743625/mod_resource/content/1/quantum%20Computing%20-%20Nakahara.pdf
        # n-k ancillary qubits
        for i in range(self.start_qubit_idx+self.phy_n_qubits, self.start_qubit_idx+self.total_qubits):
            qml.Hadamard(wires=i)
        # n-k stabilizers for n-k ancillary qubits
        for i, S in enumerate(self.stabilizer):
            for j, op in enumerate(S):
                if op == 'I':
                    continue
                elif op == 'X':
                    qml.CNOT(wires=[self.start_qubit_idx+self.phy_n_qubits+i, self.start_qubit_idx+j])
                elif op == 'Z':
                    qml.CZ(wires=[self.start_qubit_idx+self.phy_n_qubits+i, self.start_qubit_idx+j])
        for i in range(self.start_qubit_idx+self.phy_n_qubits, self.start_qubit_idx+self.total_qubits):
            qml.Hadamard(wires=i)

    def compute_syndrome(self, error):
        swapped_vector = np.append(
            error[self.phy_n_qubits: 2 * self.phy_n_qubits],
            error[0: self.phy_n_qubits]
        )
        syndrome = self.generator_matrix @ swapped_vector % 2
        return syndrome

    def compute_syndrome_table(self):
        syndrome_table = {}
        error_type = ['X', 'Y', 'Z']
        for pauli_error in error_type:
            for i in range(self.phy_n_qubits):
                error = np.zeros(2 * self.phy_n_qubits)
                if pauli_error == 'X':
                    error[i], error[i + self.phy_n_qubits] = 1, 0
                elif pauli_error == 'Y':
                    error[i], error[i + self.phy_n_qubits] = 1, 1
                elif pauli_error == 'Z':
                    error[i], error[i + self.phy_n_qubits] = 0, 1
                syndrome = tuple(self.compute_syndrome(error).astype(bool))
                syndrome_table.setdefault(pauli_error, []).append(syndrome)
        return syndrome_table

    def corrector(self):
        # based on the syndrome table, control the data qubits
        if self.code_name == '7_1_3':
            for k, v in self.syndrome_table.items():
                for i, syndrome in enumerate(v):
                    if k == 'X': # lower 3 ancillary qubits
                        qml.ctrl(qml.PauliX, list(range(self.start_qubit_idx+self.phy_n_qubits+3, self.start_qubit_idx+self.total_qubits)),
                                 control_values=syndrome[-3:])(wires=self.start_qubit_idx+i)
                    elif k == 'Z': # upper 3 ancillary qubits
                        qml.ctrl(qml.PauliZ, list(range(self.start_qubit_idx+self.phy_n_qubits, self.start_qubit_idx+self.total_qubits-3)),
                                 control_values=syndrome[:3])(wires=self.start_qubit_idx+i)
        else:
            for k, v in self.syndrome_table.items():
                for i, syndrome in enumerate(v):
                    if k == 'X':
                        qml.ctrl(qml.PauliX, list(range(self.start_qubit_idx+self.phy_n_qubits, self.start_qubit_idx+self.total_qubits)), control_values=syndrome)(wires=self.start_qubit_idx+i)
                    elif k == 'Y':
                        qml.ctrl(qml.PauliY, list(range(self.start_qubit_idx+self.phy_n_qubits, self.start_qubit_idx+self.total_qubits)), control_values=syndrome)(wires=self.start_qubit_idx+i)
                    elif k == 'Z':
                        qml.ctrl(qml.PauliZ, list(range(self.start_qubit_idx+self.phy_n_qubits, self.start_qubit_idx+self.total_qubits)), control_values=syndrome)(wires=self.start_qubit_idx+i)

    def log_circuit(self, gate_name, angle=None): # Logical Circuit of Single Qubit Gate
        if self.log_n_qubits == 1:
            if gate_name == 'X':
                qml.PauliX(wires=0)
            elif gate_name == 'Y':
                qml.PauliZ(wires=0)
                qml.PauliX(wires=0)
            elif gate_name == 'Z':
                qml.PauliZ(wires=0)
            elif gate_name == 'S':
                qml.S(wires=0)
            elif gate_name == 'H':
                qml.Hadamard(wires=0)
            elif gate_name == 'T':
                qml.T(wires=0)
            elif gate_name == 'I':
                qml.Identity(wires=0)
            elif gate_name == 'SH':
                qml.S(wires=0)
                qml.Hadamard(wires=0)
            elif gate_name == 'HS':
                qml.Hadamard(wires=0)
                qml.S(wires=0)
            elif gate_name == 'SdaggerH':
                qml.adjoint(qml.S)(wires=0)
                qml.Hadamard(wires=0)
            elif gate_name == 'HSdagger':
                qml.Hadamard(wires=0)
                qml.adjoint(qml.S)(wires=0)
            elif gate_name == 'HT':
                qml.Hadamard(wires=0)
                qml.T(wires=0)
            elif gate_name == 'TH':
                qml.T(wires=0)
                qml.Hadamard(wires=0)
            elif gate_name == 'HTdagger':
                qml.Hadamard(wires=0)
                qml.adjoint(qml.T)(wires=0)
            elif gate_name == 'TdaggerH':
                qml.adjoint(qml.T)(wires=0)
                qml.Hadamard(wires=0)
            elif gate_name == 'ST':
                qml.S(wires=0)
                qml.T(wires=0)
            elif gate_name == 'SdaggerTdagger':
                qml.adjoint(qml.S)(wires=0)
                qml.adjoint(qml.T)(wires=0)
            elif gate_name == 'RZ' and angle is not None:
                qml.RZ(angle, wires=0)
            else:
                raise ValueError("Undefined logical circuit for this gate")
        else:
            raise ValueError("Unsupported logical number of qubits!")

    # Ansatz: noise-free ansatz
    def ansatz(self, weights):
        if self.log_n_qubits == 1:
            for layer in weights:
                for i in range(self.start_qubit_idx, self.start_qubit_idx+self.phy_n_qubits):
                    qml.RX(layer[i % self.total_qubits][0], wires=i)
                    qml.RY(layer[i % self.total_qubits][1], wires=i)
                    qml.RZ(layer[i % self.total_qubits][2], wires=i)
        else:
            raise ValueError("Unsupported logical number of qubits to build ansatz")

    # Ansatz: given each qubit error probability
    def ansatz_given_each_qubit_error_probability(self, weights, rotation_error_list):
        def noise_injection(error_prob, wire, rot_g_name):
            n_error_gate = 0
            if random.random() < error_prob:
                n_error_gate += 1
                noisy_gate = random.choice([qml.PauliX, qml.PauliY, qml.PauliZ])
                g = noisy_gate(wires=wire)
                print(f"Add a {g.name} gate on wire {wire} after {rot_g_name} gate", end='\t')
            return n_error_gate

        self.n_error_gate = 0
        if self.log_n_qubits == 1:
            for layer in weights:
                for i in range(self.start_qubit_idx, self.start_qubit_idx+self.phy_n_qubits):
                    rxy_error = rotation_error_list[i%self.phy_n_qubits]
                    qml.RX(layer[i % self.total_qubits][0], wires=i)
                    self.n_error_gate += noise_injection(rxy_error, i, "RX")
                    qml.RY(layer[i % self.total_qubits][1], wires=i)
                    self.n_error_gate += noise_injection(rxy_error, i, "RY")
                    qml.RZ(layer[i % self.total_qubits][2], wires=i)
            print(f"{self.n_error_gate} gates are added")
        else:
            raise ValueError("Unsupported logical number of qubits to build noisy ansatz")

    # Ansatz: with fixed number random noise
    def random_fixed_number_noisy_ansatz(self, weights, n_errors=1):
        def generate_errors(n_errors):
            error_type = [qml.PauliX, qml.PauliY, qml.PauliZ]
            total_layers = weights.shape[0]
            error_positions = []
            # random generate errors (error_layer, error_wire, after_rot)
            while len(error_positions) < n_errors:
                layer = random.randint(0, total_layers - 1)
                qubit = random.randint(self.start_qubit_idx, self.start_qubit_idx+self.phy_n_qubits - 1)
                rotation_gate = random.choice(["RX", "RY"])
                if (layer, qubit, rotation_gate) not in error_positions: # deduplication
                    error_positions.append((layer, qubit, rotation_gate))
            # errors type
            errors = [random.choice(error_type) for _ in range(n_errors)]
            return error_positions, errors

        def noise_injection(position, error_positions, errors):
            if position in error_positions:
                li, w, rot = position
                error_idx = error_positions.index(position)  # add error
                g = errors[error_idx](wires=w)
                print(f"Add a {g.name} gate on wire {w} after {rot} gate in layer {li}", end='\t')

        self.n_error_gate = 0
        if self.log_n_qubits == 1:
            error_positions, errors = generate_errors(n_errors)
            for l, layer in enumerate(weights):
                for i in range(self.start_qubit_idx, self.start_qubit_idx+self.phy_n_qubits):
                    qml.RX(layer[i % self.total_qubits][0], wires=i)
                    noise_injection((l, i, "RX"), error_positions, errors)
                    qml.RY(layer[i % self.total_qubits][1], wires=i)
                    noise_injection((l, i, "RY"), error_positions, errors)
                    qml.RZ(layer[i % self.total_qubits][2], wires=i)
            self.n_error_gate = n_errors
            print(f"{self.n_error_gate} gates are added")
        else:
            raise ValueError("Unsupported logical number of qubits to build noisy ansatz")