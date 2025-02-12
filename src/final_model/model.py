import pulp

class ModelloCutAndShoot:
    def __init__(
        self,
        edges,
        vertex_weights,
        qpus,
        num_shots_per_subcircuit,
        num_subcircuits,
        alpha=0.5,
        beta=0.5
    ):
        """
        inizializza il modello con:
        
        :param edges: lista di (u, v) che rappresentano gli archi del circuito
        :param vertex_weights: dizionario {vertex: weight}, peso (qubit/gate) di ciascun vertice
        :param qpus: lista di  QPU
        :param num_shots_per_subcircuit: Numero di shot da assegnare a ciascun sottocircuito
        :param alpha: peso per il numero di tagli nella funzione obiettivo
        :param beta: peso per il makespan nella funzione obiettivo
        """
        self.edges = edges
        self.vertices = list(vertex_weights.keys())
        self.vertex_weights = vertex_weights
        self.qpus = qpus
        self.num_shots_per_subcircuit = num_shots_per_subcircuit

        if alpha < 0:
            raise Exception("Alpha deve essere maggiore o uguale a 0")
        elif beta < 0:
            raise Exception("Beta deve essere maggiore o uguale a 0")
        elif abs(alpha + beta) != 1.0:
            raise Exception("Alpha + Beta deve essere uguale di 1")

        self.alpha = alpha
        self.beta = beta

        self.num_subcircuits = num_subcircuits
        self.num_qpus = len(qpus)

        self.qpus_index = range(self.num_qpus)
        self.subcircuits = range(self.num_subcircuits)

        self.problem = pulp.LpProblem("CircuitCutter_WithQPU", pulp.LpMinimize)

        # variabili
        self.aggiungi_variabili()

        # vincoli
        self.aggiungi_vincoli()

        # funzione obiettivo
        self.costruisci_funzione_obiettivo()

    def aggiungi_variabili(self):
        """
        definizione dellel variabili del modello
        """
        # y[v,c]: se il gate v appartiene al sottocircuito c
        self.y = pulp.LpVariable.dicts(
            "y",
            [(v, c) for v in self.vertices for c in self.subcircuits],
            cat=pulp.LpBinary
        )

        # x[e,c]: se l'arco e è tagliato dal sottocircuito c
        self.x = pulp.LpVariable.dicts(
            "x",
            [(e, c) for e in self.edges for c in self.subcircuits],
            cat=pulp.LpBinary
        )

        # a[c]: numero di gate nel sottocircuito c
        # p[c]: qubit di init aggiuntivi
        # o[c]: qubit misurati in uscita
        # f[c]: qubit totali visti = a[c] + p[c] - o[c]
        # d[c]: qubit in ingresso = a[c] + p[c]
        self.a = pulp.LpVariable.dicts("a", self.subcircuits, cat=pulp.LpInteger)
        self.p = pulp.LpVariable.dicts("p", self.subcircuits, cat=pulp.LpInteger)
        self.o = pulp.LpVariable.dicts("o", self.subcircuits, cat=pulp.LpInteger)
        self.f = pulp.LpVariable.dicts("f", self.subcircuits, cat=pulp.LpInteger)
        self.d = pulp.LpVariable.dicts("d", self.subcircuits, cat=pulp.LpInteger)

        # z_p[e,c] = x[e,c] * y[e[1], c]
        # z_o[e,c] = x[e,c] * y[e[0], c]
        self.z_p = pulp.LpVariable.dicts(
            "z_p",
            [(e, c) for e in self.edges for c in self.subcircuits],
            cat=pulp.LpBinary
        )
        self.z_o = pulp.LpVariable.dicts(
            "z_o",
            [(e, c) for e in self.edges for c in self.subcircuits],
            cat=pulp.LpBinary
        )

        # shots_assign[c,q] = numero di shot del sottocircuito c assegnati alla QPU q
        self.shots_assign = pulp.LpVariable.dicts(
            "shots_assign",
            [(c, q) for c in self.subcircuits for q in self.qpus_index],
            lowBound=0,
            upBound=self.num_shots_per_subcircuit,
            cat=pulp.LpInteger
        )

        # use_q[q] = 1 se la QPU q è utilizzata da almeno un sottocircuito
        self.use_q = pulp.LpVariable.dicts(
            "use_q",
            self.qpus_index,
            cat=pulp.LpBinary
        )

        # T_q[q] = tempo totale di utilizzo della QPU q
        self.T_q = pulp.LpVariable.dicts(
            "T_q",
            self.qpus_index,
            lowBound=0,
            cat=pulp.LpContinuous
        )

        # makespan (massimo tra i T_q)
        self.T = pulp.LpVariable("Makespan", lowBound=0, cat=pulp.LpContinuous)

        # abilita[c,q]: 1 se la QPU q è abilitata ad eseguire il sottocircuito c
        self.abilita = pulp.LpVariable.dicts(
            'abilita',
            [(c, q) for c in self.subcircuits for q in self.qpus_index],
            cat=pulp.LpBinary
        )

    def aggiungi_vincoli(self):
        """
        aggiunge i vincoli al modello
        """
        # Per comodità
        edges = self.edges
        vertices = self.vertices
        subcircuits = self.subcircuits
        qpus_index = self.qpus_index

        # vincoli su a[c], p[c], o[c], f[c], d[c]
        for c in subcircuits:
            # a[c] = somma del peso dei vertici (gate) assegnati al sottocircuito c
            self.problem += (
                self.a[c] == pulp.lpSum(self.vertex_weights[v]*self.y[v,c] for v in vertices),
                f"A_{c}"
            )
            # p[c] = somma di z_p[e,c]
            self.problem += (
                self.p[c] == pulp.lpSum(self.z_p[(e,c)] for e in edges),
                f"P_{c}"
            )
            # o[c] = somma di z_o[e,c]
            self.problem += (
                self.o[c] == pulp.lpSum(self.z_o[(e,c)] for e in edges),
                f"O_{c}"
            )
            # f[c] = a[c] + p[c] - o[c]
            self.problem += (
                self.f[c] == self.a[c] + self.p[c] - self.o[c],
                f"F_{c}"
            )
            # d[c] = a[c] + p[c]
            self.problem += (
                self.d[c] == self.a[c] + self.p[c],
                f"D_{c}"
            )

        # linearizzazione per z_p e z_o
        for e in edges:
            for c in subcircuits:
                # z_p = x[e,c] * y[e[1], c]
                self.problem += (
                    self.z_p[(e, c)] <= self.x[(e, c)],
                    f"zp_1_{e}_{c}"
                )
                self.problem += (
                    self.z_p[(e, c)] <= self.y[e[1], c],
                    f"zp_2_{e}_{c}"
                )
                self.problem += (
                    self.z_p[(e, c)] >= self.x[(e, c)] + self.y[e[1], c] - 1,
                    f"zp_3_{e}_{c}"
                )

                # z_o = x[e,c] * y[e[0], c]
                self.problem += (
                    self.z_o[(e, c)] <= self.x[(e, c)],
                    f"zo_1_{e}_{c}"
                )
                self.problem += (
                    self.z_o[(e, c)] <= self.y[e[0], c],
                    f"zo_2_{e}_{c}"
                )
                self.problem += (
                    self.z_o[(e, c)] >= self.x[(e, c)] + self.y[e[0], c] - 1,
                    f"zo_3_{e}_{c}"
                )

        # pgni vertice deve stare esattamente in un sottocircuito
        for v in vertices:
            self.problem += (
                pulp.lpSum(self.y[v,c] for c in subcircuits) == 1,
                f"Vertice_{v}_unico"
            )

        # vincoli sugli archi: non si può tagliare se e[0] e e[1] sono nello stesso sottocircuito
        for c in subcircuits:
            for e in edges:
                self.problem += (
                    self.x[e,c] <= self.y[e[0],c] + self.y[e[1],c],
                    f"x_1_{e}_{c}"
                )
                self.problem += (
                    self.x[e,c] >= self.y[e[0],c] - self.y[e[1],c],
                    f"x_2_{e}_{c}"
                )
                self.problem += (
                    self.x[e,c] >= self.y[e[1],c] - self.y[e[0],c],
                    f"x_3_{e}_{c}"
                )
                self.problem += (
                    self.x[e,c] <= 2 - self.y[e[0],c] - self.y[e[1],c],
                    f"x_4_{e}_{c}"
                )

        #vincolo di ordine
        for k in range(self.num_subcircuits):
            self.problem += (
                pulp.lpSum(self.y[(k, j)] for j in range(k+1, self.num_subcircuits)) == 0,
                f"Ordine_subc_{k}"
            )

        # vincoli sul numero di shot
        for c in subcircuits:
            self.problem += (
                pulp.lpSum(self.shots_assign[(c, q)] for q in qpus_index) == self.num_shots_per_subcircuit,
                f"ShotsTotali_sottoc_{c}"
            )

        # attivazione QPU: se la QPU q ha shot > 0, use_q[q] = 1
        M_shots = self.num_subcircuits * self.num_shots_per_subcircuit
        for q in qpus_index:
            self.problem += (
                pulp.lpSum(self.shots_assign[(c, q)] for c in subcircuits) <= M_shots * self.use_q[q],
                f"UseQ_{q}_1"
            )

        # vincoli di capacità QPU (abilita[c,q])
        BigM_d = len(self.vertices) + len(self.edges)  # un big M per d[c]
        for c in subcircuits:
            for q in qpus_index:
                # d[c] <= capacita[q] + BigM_d*(1 - abilita[c,q])
                self.problem += (
                    self.d[c] <= self.qpus[q].capacita + BigM_d*(1 - self.abilita[(c,q)]),
                    f"cap_{c}_{q}"
                )
                # Se abilita[c,q] = 0 => shots_assign[c,q] = 0
                self.problem += (
                    self.shots_assign[(c, q)] <= self.num_shots_per_subcircuit * self.abilita[(c,q)],
                    f"enable_shots_{c}_{q}"
                )

        # vincoli per il tempo di utilizzo QPU
        for q in qpus_index:
            # T_q[q] >= tempo_di_coda * use_q[q]
            self.problem += (
                self.T_q[q] >= self.qpus[q].tempo_di_coda * self.use_q[q],
                f"TempoCodaMin_q{q}"
            )

            # T_q[q] >= somma shot_assign * tempo_di_esecuzione
            self.problem += (
                self.T_q[q] >= pulp.lpSum(
                    self.shots_assign[(c, q)] * self.qpus[q].tempo_di_esecuzione
                    for c in subcircuits
                ),
                f"TempoEsecuzioneMin_q{q}"
            )

            # T_q[q] <= (tempo_di_coda * use_q[q]) + (somma shot_assign * tempo_di_esecuzione)
            self.problem += (
                self.T_q[q] <= self.qpus[q].tempo_di_coda * self.use_q[q] + pulp.lpSum(
                    self.shots_assign[(c, q)] * self.qpus[q].tempo_di_esecuzione
                    for c in subcircuits
                ),
                f"TempoMax_q{q}"
            )

            # makespan T >= T_q[q]
            self.problem += (
                self.T >= self.T_q[q],
                f"Makespan_{q}"
            )

    def costruisci_funzione_obiettivo(self):
        """
        funzione obiettivo: alpha*(K_norm) + beta*(T_norm)
        K = sum(x[e,c]) / 2
        T = makespan
        """
        K = pulp.lpSum(self.x[e, c] for c in self.subcircuits for e in self.edges) / 2.0
        K_max = len(self.edges) / 2.0  # K massimo se tutti gli archi venissero tagliati in ogni subcircuito

        T_max = max(
            qpu.tempo_di_coda + self.num_subcircuits * self.num_shots_per_subcircuit * qpu.tempo_di_esecuzione
            for qpu in self.qpus
        )

        # normalizzazioni
        K_norm = K / K_max if K_max > 0 else 0
        T_norm = self.T * (1/T_max) if T_max > 0 else 0

        # funzione obiettivo
        """ self.problem += (
            self.alpha * K_norm + self.beta * T_norm,
            "Minimizza_Tagli_e_Makespan_Normalizzati"
        ) """

        self.problem += (
            self.alpha * K_norm + self.beta * T_norm,
            "Minimizza_Tagli"
        )

    def solve_model(self, solver=None):
        """
        :param solver: risolutore specifico (none se si vuole usare quello di defualt)
        :return: lo status e il valore di ottimo della funzione obiettivo
        """
        if solver is not None:
            self.problem.solve(solver)
        else:
            self.problem.solve( )

        return pulp.LpStatus[self.problem.status], pulp.value(self.problem.objective)

    def stampa_e_restituisci_risultato(self):
        """
        stampa informazioni di sintesi.
        restituisce un dizionario sottocircuiti con la struttura:
        {
        sub_id: {
            "vertices": [...],
            "shots": {...}
            },
        ...
        }
        """
        status = pulp.LpStatus[self.problem.status]
        obj_value = pulp.value(self.problem.objective) if self.problem.objective is not None else None
        print(f"Status: {status}")
        print(f"Valore obiettivo: {obj_value:.4f}\n")

        # struttura che conterrà il risultato
        sottocircuiti = {}

        print("Sottocircuiti (archi assegnati):")
        for c in self.subcircuits:
            vertici_sottocircuito = [v for v in self.vertices if pulp.value(self.y[v, c]) == 1]
            print(f"  Sottocircuito {c}: {vertici_sottocircuito}")
            
        
            sottocircuiti[c] = {
                "vertices": vertici_sottocircuito,
                "shots": {}
            }

        for c in self.subcircuits:
            assegnazione = {}
            for q in self.qpus_index:
                val = pulp.value(self.shots_assign[(c, q)])
                if val > 0:
                    assegnazione[self.qpus[q].index] = (int(val))
            print(f"Sottocircuito {c} -> Shots assegnati: {assegnazione}")
            
            sottocircuiti[c]["shots"] = assegnazione

        print("\n-- Utilizzo QPU --")
        for q in self.qpus_index:
            t_q_val = pulp.value(self.T_q[q])
            if t_q_val > 0:
                shots_total = sum(pulp.value(self.shots_assign[(c, q)]) for c in self.subcircuits)
                print(f"QPU {self.qpus[q].index}: T_q = {t_q_val:.2f}, shots totali = {shots_total}")

        T_val = pulp.value(self.T)
        print(f"\nMakespan T = {T_val:.2f}")

        return sottocircuiti
