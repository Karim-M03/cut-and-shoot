

from qiskit_ibm_provider import IBMProvider
"""
this module saves IBM Quantum account credential.
Replace IBM_TOKEN with your actual IBM Quantum API token
"""

IBMProvider.save_account(token='IBM_TOKEN', overwrite=True)