# Tesi Karim Mahmoud
Estensione del Cut&Shoot
---

## Step 0
Implementare su un solver il modello MILP CutQC.

---

## Step 1
- **Più QPUs tra le quali scegliere**:  
  Ogni QPU (ci vengono dati):  
  - Stima del tempo di esecuzione per quel circuito.  
  - Tempo di coda.  

- **Selezioniamo solo una**  
  Minimizza (tempo di coda + tempo di esecuzione + tempo post-processing).

---

## Step 2
- **Più QPUs tra le quali scegliere**:  
  Ogni QPU (ci vengono dati):  
  - Stima del tempo di esecuzione per quel circuito.  
  - Tempo di coda.

- **Shot-wise uniforme** (numero di shots/QPUs selezionate).

- **Selezioniamo insieme QPUs**  
  Minimizza max (tempo di coda + tempo di esecuzione\#QPU + tempo post-processing

---

## Step 3
- **Equivalente Step 2 + vincoli QoS**  
  (ad esempio, ogni QPU ha associato un modello prezzo e/o una stima dell’affidabilità dei risultati).

- **Selezioniamo insieme QPUs**  
  Minimizza la combinazione lineare max (tempo di coda + tempo di esecuzione\#QPU + tempo post-processing) + gli altri requisiti.

- **Vincoli di tipo predicato**  
  (esempio: GDPR che ci scarta macchine fuori dall’Europa).

---

## Step 4
- **Step 3 ma senza shot-wise uniforme**  
  (il modello sceglie il modo migliore per distribuire gli shots).

---

## Step Bonus
- **Stimiamo noi tempi di esecuzione e coda.**

---
