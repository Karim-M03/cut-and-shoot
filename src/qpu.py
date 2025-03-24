from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider

class QPU:
    def __init__(self, qpu_type, index):
        self.qpu_type = qpu_type
        self.index = index
        self.backend = self._initialize_backend(qpu_type)

        if self.backend:
            self._set_backend_metrics()
        else:
            self.queue_time = float('inf')
            self.execution_time = 1
            self.simulator = True
            self.capacity = 0
            self.backend_name = 'unknown'

    def _initialize_backend(self, qpu_type):
        """Try to load either an AerSimulator or an IBM backend."""
        try:
            # Check if it's a local simulator name (like "aer_simulator")
            if qpu_type in [b.name() for b in AerSimulator.backends()]:
                return AerSimulator()
            
            provider = IBMProvider()
            if qpu_type in [b.name for b in provider.backends()]:
                return provider.get_backend(qpu_type)
        except Exception as e:
            print(f"[Warning] Failed to load backend '{qpu_type}': {e}")

        # fallback
        print(f"[Fallback] Using default AerSimulator for '{qpu_type}'")
        return AerSimulator()

    def _set_backend_metrics(self):
        """Set execution time, queue time, and capacity based on backend."""
        config = self.backend.configuration()

        self.simulator = config.simulator
        self.capacity = config.n_qubits
        self.backend_name = config.backend_name

        if self.simulator:
            self.queue_time = 0
            self.execution_time = 0
        else:
            status = self.backend.status()
            self.queue_time = status.pending_jobs if status.operational else float('inf')
            self.execution_time = getattr(config, 'default_rep_delay', 1)

    def update_metrics(self, execution_time, queue_time, capacity):
        """replace properties in case of bad data"""
        self.execution_time = execution_time
        self.queue_time = queue_time
        self.capacity = capacity
        self.backend = AerSimulator()
        

    def __repr__(self):
        return (
            f"QPU(index={self.index}, name={self.backend_name}, "
            f"simulator={self.simulator}, "
            f"capacity={self.capacity}, "
            f"exec_time={self.execution_time}, "
            f"queue={self.queue_time})"
        )
