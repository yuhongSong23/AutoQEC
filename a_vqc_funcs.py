import math
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


# Dataset
def dataset(logical_circuit, gate_name, data_num, angle=None, cx_flag=None):
    inputs = np.linspace(0, 1, data_num)
    np.random.seed(4)
    np.random.shuffle(inputs)
    if cx_flag is None:
        outputs = np.array([logical_circuit(alpha, gate_name, angle) for alpha in inputs]) # Logical Circuit for Single Qubit
    else:
        outputs = np.array([logical_circuit(alpha) for alpha in inputs]) # Logical_Circuit_CX

    ratio = 0.8
    num_train = math.floor(data_num * ratio)

    train_x, test_x = inputs[:num_train], inputs[num_train:]
    train_y, test_y = outputs[:num_train], outputs[num_train:]

    return pnp.array(train_x), pnp.array(train_y), pnp.array(test_x), pnp.array(test_y)


# Generate test dataset
def generate_test_dataset(logical_circuit, gate_name, data_num, angle=None, cx_flag=None):
    inputs = np.linspace(0, 1, data_num)
    np.random.seed(4)
    np.random.shuffle(inputs)
    if cx_flag is None:
        outputs = np.array([logical_circuit(alpha, gate_name, angle) for alpha in inputs]) # Logical Circuit for Single Qubit
    else:
        outputs = np.array([logical_circuit(alpha) for alpha in inputs]) # Logical_Circuit_CX

    return pnp.array(inputs), pnp.array(outputs)


# cost function
def cost_fn(circuit, weights, X_batch, Y_batch, noise_level=None):
    if noise_level is None:
        outputs = pnp.array([circuit(alpha, weights) for alpha in X_batch])
    else:
        outputs = pnp.array([circuit(alpha, weights, noise_level) for alpha in X_batch])
    fidelity = qml.math.fidelity(outputs, Y_batch)
    cost_list = 1.0 - fidelity
    return pnp.mean(cost_list)


# VQC-based FT gate training
def train(train_x, circuit, weights_init, train_y, optimizer, curve_name, epochs, batch_size=10, noise_level=None):
    print(f"Training for {len(train_x)} data......")
    weights = weights_init
    batch_size = min(batch_size, len(train_x))
    cost_cur = []

    for i in range(epochs): # epoch
        cost_val, batch_ite = 0.0, 0
        for j in range(0, len(train_x), batch_size): # batch
            batch_ite += 1
            # Update the weights by one optimizer step, using only a limited batch of data
            X_batch = train_x[j: j+batch_size] # a batch of train data
            Y_batch = train_y[j: j+batch_size, :] # a batch of train labels
            weights, cost_val_batch = optimizer.step_and_cost(lambda v:cost_fn(circuit, v, X_batch, Y_batch, noise_level), weights)
            print(f"\t batch: {batch_ite} | Cost: {cost_val_batch}")
            cost_val += cost_val_batch
        cost_epoch = cost_val/batch_ite
        cost_cur.append(cost_epoch)
        print(f"Epoch: {i + 1} | Cost: {cost_epoch}")
    # draw_curves_with_same_ylabel([cost_cur], ['Cost'], curve_name, "#Epoch", "Cost Values", range(1, epochs+1))
    return weights


# VQC-based FT gate inference
def test(test_x, circuit, weights, test_y, curve_name, noise_level=None):
    print(f"\nTask: {curve_name} for {len(test_x)} data......")
    if noise_level is None:
        outputs = pnp.array([circuit(alpha, weights) for alpha in test_x])
    else:
        outputs = pnp.array([circuit(alpha, weights, noise_level) for alpha in test_x])
    # print(f"Test y: {test_y}")
    # print(f"Density Matrix: {outputs}")
    fidelity = qml.math.fidelity(outputs, test_y)
    cost_list = 1.0 - fidelity
    print("(input alpha, cost) pair")
    for x, cost in zip(test_x, cost_list):
        print(x, ",", cost)
    # draw_curves_with_same_ylabel([cost_list], ['Cost'], curve_name, "Alpha", "Cost Values", test_x, line=False)
    print(f"Average Test Cost: {pnp.mean(cost_list)}")

    return cost_list

def train_and_test_circuit(phy_circuit, log_circuit, gate_name, task_name, vqc_shape, creterion, angle=None, epochs=30, data_num=200, lr=0.01, noise_level=None, cx_flag=None):
    train_x, train_y, test_x, test_y = dataset(log_circuit, gate_name, data_num, angle, cx_flag)
    # weight initialization for ansatz
    weights_init = 4 * pnp.pi * pnp.random.random(vqc_shape, requires_grad=True) - 2 * pnp.pi  # random weight # parameter initialize from 0 to 2pi
    # Traning and Test are the same model
    optimizer = creterion(stepsize=lr)
    trained_weights = train(train_x, phy_circuit, weights_init, train_y, optimizer, "Training Curve for " + task_name, epochs, batch_size=20, noise_level=noise_level)
    print("\nWeights after Training: ", trained_weights.shape)
    print(trained_weights)
    # test(test_x, phy_circuit, trained_weights, test_y, "Test Curve for " + task_name, noise_level=noise_level)
    # circuit information
    print("\nCircuit......")
    if noise_level is None:
        print(qml.draw(phy_circuit)(alpha=0.1, weights=trained_weights))
    else:
        print(qml.draw(phy_circuit)(alpha=0.1, weights=trained_weights, error_list=noise_level))
    return trained_weights