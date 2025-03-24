from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def ripple_carry_adder(n):
    a_reg = QuantumRegister(n, name="a")      
    b_reg = QuantumRegister(n, name="b")       
    carry_reg = QuantumRegister(n + 1, name="c")  
    classical_reg = ClassicalRegister(2 * n + 1, name="output")
    qc = QuantumCircuit(a_reg, b_reg, carry_reg, classical_reg)
    for i in range(n):
        qc.ccx(a_reg[i], b_reg[i], carry_reg[i + 1])  
        qc.cx(a_reg[i], b_reg[i])                     
        qc.ccx(b_reg[i], carry_reg[i], carry_reg[i + 1])  
    qc.cx(carry_reg[n - 1], carry_reg[n])
    qc.measure(a_reg, classical_reg[:n]) 
    qc.measure(b_reg, classical_reg[n:2 * n])  
    qc.measure(carry_reg[n], classical_reg[2 * n])

    return qc
