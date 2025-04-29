import pennylane as qml
import numpy as np
import math
from datetime import datetime
from contextlib import redirect_stdout


# --------------compute gate count for RZ gate decomposition--------------
@qml.qnode(qml.device("default.qubit"))
def circuit(angle):
    qml.RZ(angle, wires=0)
    return qml.expval(qml.Z(0))

def compute_gate_counts_for_gate_decomposition():
    angle_list = np.linspace(0, math.pi, 10)
    epsilon_list = [0.1]
    for epsilon in epsilon_list:
        print(f"precision: {epsilon}")
        for angle in angle_list:
            print(f"Angle: {angle}")
            decomposed_circuit = qml.transforms.clifford_t_decomposition(circuit, epsilon)
            specs_2 = qml.specs(decomposed_circuit)(angle)
            print("decomposed circuit depth: ", specs_2["resources"].depth)
            print(f"{specs_2['resources']}")
        print("-" * 20)


if __name__ == '__main__':
    out_file = './results/gate_decomposition_for_rotation_gate' + str(datetime.now()) + '.txt'

    with open(out_file, 'w') as f, redirect_stdout(f):
        start_time = datetime.now()
        print(start_time)
        compute_gate_counts_for_gate_decomposition()
        end_time = datetime.now()
        print(end_time)
        print("Execution time (s): ", (end_time - start_time).total_seconds())