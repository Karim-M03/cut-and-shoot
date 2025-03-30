from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from grover import grover_circuit
from simple_circuit import simple_circuit


def run_circuit(qc, shots=1024, draw=False):
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
    qc2 = simple_circuit()
    results_2 = run_circuit(qc2, shots=1024, draw=True)