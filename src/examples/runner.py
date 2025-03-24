from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from grover import grover_circuit
# needed for multi-controlled gates (MCX)

def run_circuit(qc, shots=1024, draw=False):
    # runs the given circuit on AerSimulator
    # optionally draws the circuit and histogram

    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    print('measurement results:')
    print(counts)

    if draw:
        qc.draw('mpl')
        plot_histogram(counts)
        plt.show()

    return counts

if __name__ == '__main__':
    qc2 = grover_circuit(2)
    results_2 = run_circuit(qc2, shots=1024, draw=True)