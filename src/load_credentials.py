

from qiskit_ibm_provider import IBMProvider
"""
Module: load_credentials.py
This module provides functionality for saving IBM Quantum account credential
Usage:
    Replace IBM_TOKEN with your actual IBM Quantum Experience API token
"""

IBMProvider.save_account(token='IBM_TOKEN', overwrite=True)
