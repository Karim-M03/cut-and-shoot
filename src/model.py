import pulp

class CutAndShootModel:
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
        initializes the model with:
        
        :param edges: a list of (u, v) representing the circuit edges
        :param vertex_weights: a dictionary {vertex: weight} for each vertex's weight (qubit/gate)
        :param qpus: a list of QPU objects
        :param num_shots_per_subcircuit: number of shots to assign to each subcircuit
        :param num_subcircuits: total number of subcircuits
        :param alpha: weight for the number of cuts in the objective function
        :param beta: weight for the time makespan in the objective function
        """
        self.edges = edges
        self.vertices = list(vertex_weights.keys())
        self.vertex_weights = vertex_weights
        self.qpus = qpus
        self.num_shots_per_subcircuit = num_shots_per_subcircuit

        if alpha < 0:
            raise Exception("alpha must be greater than or equal to 0")
        elif beta < 0:
            raise Exception("beta must be greater than or equal to 0")
        elif abs(alpha + beta) != 1.0:
            raise Exception("alpha + meta must equal 1")

        self.alpha = alpha
        self.beta = beta

        self.num_subcircuits = num_subcircuits
        self.num_qpus = len(qpus)

        self.qpus_index = range(self.num_qpus)
        self.subcircuits = range(self.num_subcircuits)

        self.problem = pulp.LpProblem("Cut&Shoot", pulp.LpMinimize)

        # variables
        self.add_variables()

        # constraints
        self.add_constraints()

        # objective function
        self.build_objective_function()

    def add_variables(self):
        """
        definition of the model variables
        """
        # y[v,c]: 1 if gate v belongs to subcircuit c, 0 otherwise
        self.y = pulp.LpVariable.dicts(
            "y",
            [(v, c) for v in self.vertices for c in self.subcircuits],
            cat=pulp.LpBinary
        )

        # x[e,c]: 1 if the edge e is cut by subcircuit c, 0 otherwise
        self.x = pulp.LpVariable.dicts(
            "x",
            [(e, c) for e in self.edges for c in self.subcircuits],
            cat=pulp.LpBinary
        )

        # a[c]: number of qubits entering subcircuit c
        # p[c]: additional initialization qubits for subcircuit c
        # o[c]: qubits measured at the output of subcircuit c
        # f[c]: total qubits 'seen' = a[c] + p[c] - o[c]
        # d[c]: total input qubits = a[c] + p[c]
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

        # u[c] = 1 if subcircuit c is actually used (has qubits), 0 otherwise
        self.u = pulp.LpVariable.dicts("u", self.subcircuits, cat=pulp.LpBinary)

        # shots_assign[c,q] = number of shots of subcircuit c assigned to QPU q
        self.shots_assign = pulp.LpVariable.dicts(
            "shots_assign",
            [(c, q) for c in self.subcircuits for q in self.qpus_index],
            lowBound=0,
            upBound=self.num_shots_per_subcircuit,
            cat=pulp.LpInteger
        )

        # use_q[q] = 1 if QPU q is used by at least one subcircuit, 0 otherwise
        self.use_q = pulp.LpVariable.dicts(
            "use_q",
            self.qpus_index,
            cat=pulp.LpBinary
        )

        # T_q[q] = total usage time of QPU q
        self.T_q = pulp.LpVariable.dicts(
            "T_q",
            self.qpus_index,
            lowBound=0,
            cat=pulp.LpInteger
        )

        # makespan (the maximum among T_q)
        self.T = pulp.LpVariable("Makespan", lowBound=0, cat=pulp.LpInteger)

        # abilita[c,q]: 1 if QPU q can run subcircuit c
        self.abilita = pulp.LpVariable.dicts(
            'abilita',
            [(c, q) for c in self.subcircuits for q in self.qpus_index],
            cat=pulp.LpBinary
        )

    def add_constraints(self):
        """
        adds constraints to the model
        """
        edges = self.edges
        vertices = self.vertices
        subcircuits = self.subcircuits
        qpus_index = self.qpus_index

        # constraints on a[c], p[c], o[c], f[c], d[c]
        for c in subcircuits:
            # a[c] = sum of the weights of vertices assigned to subcircuit c
            self.problem += (
                self.a[c] == pulp.lpSum(self.vertex_weights[v]*self.y[v, c] for v in vertices),
                f"A_{c}"
            )
            # p[c] = sum of z_p[e,c]
            self.problem += (
                self.p[c] == pulp.lpSum(self.z_p[(e, c)] for e in edges),
                f"P_{c}"
            )
            # o[c] = sum of z_o[e,c]
            self.problem += (
                self.o[c] == pulp.lpSum(self.z_o[(e, c)] for e in edges),
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

        # lnearization constraints for z_p and z_o
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

        # every vertex must be in exactly one subcircuit
        for v in vertices:
            self.problem += (
                pulp.lpSum(self.y[v, c] for c in subcircuits) == 1,
                f"Unique_vertex_{v}"
            )

        # wdge constraints: cannot cut an edge if both endpoints are in the same subcircuit
        for c in subcircuits:
            for e in edges:
                self.problem += (
                    self.x[e, c] <= self.y[e[0], c] + self.y[e[1], c],
                    f"x_1_{e}_{c}"
                )
                self.problem += (
                    self.x[e, c] >= self.y[e[0], c] - self.y[e[1], c],
                    f"x_2_{e}_{c}"
                )
                self.problem += (
                    self.x[e, c] >= self.y[e[1], c] - self.y[e[0], c],
                    f"x_3_{e}_{c}"
                )
                self.problem += (
                    self.x[e, c] <= 2 - self.y[e[0], c] - self.y[e[1], c],
                    f"x_4_{e}_{c}"
                )

        # ordering constraint (as provided)
        for k in range(self.num_subcircuits):
            self.problem += (
                pulp.lpSum(self.y[(k, j)] for j in range(k+1, self.num_subcircuits)) == 0,
                f"Ordering_subc_{k}"
            )

        # big M for constraints
        BigM_d = sum(self.vertex_weights.values())  # A large value based on the total number of gates

        for c in self.subcircuits:
            # if subcircuit c has any qubits in input, then u[c] = 1, otherwise 0
            self.problem += (
                self.d[c] <= BigM_d * self.u[c],
                f"Active_subcircuit_{c}"
            )

        # constraints on the number of shots
        for c in subcircuits:
            # we do not assign shots to an empty subcircuit
            self.problem += (
                pulp.lpSum(self.shots_assign[(c, q)] for q in qpus_index)
                == self.num_shots_per_subcircuit * self.u[c],
                f"Total_shots_subc_{c}"
            )

        # QPU activation: if QPU q has any assigned shots > 0, use_q[q] = 1
        M_shots = self.num_subcircuits * self.num_shots_per_subcircuit
        for q in qpus_index:
            self.problem += (
                pulp.lpSum(self.shots_assign[(c, q)] for c in subcircuits) <= M_shots * self.use_q[q],
                f"UseQ_{q}_1"
            )

        # QPU capacity constraints (abilita[c,q])
        BigM_d = len(self.vertices) + len(self.edges)  # Another big M for d[c]
        for c in subcircuits:
            for q in qpus_index:
                # d[c] <= qpus[q].capacity + BigM_d*(1 - abilita[c,q])
                self.problem += (
                    self.d[c] <= self.qpus[q].capacity + BigM_d*(1 - self.abilita[(c, q)]),
                    f"cap_{c}_{q}"
                )
                # If abilita[c,q] = 0 => shots_assign[c,q] = 0
                self.problem += (
                    self.shots_assign[(c, q)] <= self.num_shots_per_subcircuit * self.abilita[(c, q)],
                    f"enable_shots_{c}_{q}"
                )

        # QPU usage time constraints
        for q in qpus_index:
            # T_q[q] >= queue_time * use_q[q]
            self.problem += (
                self.T_q[q] >= self.qpus[q].queue_time * self.use_q[q],
                f"QueueTimeMin_q{q}"
            )

            # T_q[q] >= sum of (shots_assign * execution_time)
            self.problem += (
                self.T_q[q] >= pulp.lpSum(
                    self.shots_assign[(c, q)] * self.qpus[q].execution_time
                    for c in subcircuits
                ),
                f"ExecutionTimeMin_q{q}"
            )

            # T_q[q] <= (queue_time * use_q[q]) + sum(shots_assign * execution_time)
            self.problem += (
                self.T_q[q] <= self.qpus[q].queue_time * self.use_q[q] + pulp.lpSum(
                    self.shots_assign[(c, q)] * self.qpus[q].execution_time
                    for c in subcircuits
                ),
                f"MaxTime_q{q}"
            )

            # makespan T >= T_q[q]
            self.problem += (
                self.T >= self.T_q[q],
                f"Makespan_{q}"
            )

    def build_objective_function(self):
        """
        objective function: alpha * (K_norm) + beta * (T_norm)
        where:
          K = sum(x[e,c]) / 2
          T = makespan
        """
        # number of cuts K
        self.K = pulp.lpSum(self.x[e, c] for c in self.subcircuits for e in self.edges) / 2.0
        K_max = len(self.edges) / 2.0  # Max possible K if all edges were cut in every subcircuit

        # max T if all subcircuits + shots go to a single QPU with the worst time
        T_max = max(
            qpu.queue_time + self.num_subcircuits * self.num_shots_per_subcircuit * qpu.execution_time
            for qpu in self.qpus
        )

        # normalizations
        K_norm = self.K / K_max if K_max > 0 else 0
        T_norm = self.T * (1/T_max) if T_max > 0 else 0

        # objective function
        self.problem += (
            self.alpha * K_norm + self.beta * T_norm,
            "Minimize_Cuts_and_Makespan_Normalized"
        )

    def solve_model(self, solver=None):
        """
        solves the model using the specified solver (or the default if none)
        :param solver: specific solver (None if default is to be used)
        :return: the solver status and the optimal objective value
        """
        if solver is not None:
            self.problem.solve(solver)
        else:
            self.problem.solve(pulp.PULP_CBC_CMD(msg=False))
        return pulp.LpStatus[self.problem.status], pulp.value(self.problem.objective)

    def print_and_return_solution(self):
        """
        prints the solution status and objective value, then returns a structured
        dictionary containing only the subcircuits that include at least one vertex
        """
        status = pulp.LpStatus[self.problem.status]
        obj_value = pulp.value(self.problem.objective) if self.problem.objective is not None else None
        print(f"Status: {status}")
        print(f"Objective value: {obj_value:.4f}\n")

        if status == "Infeasible":
            return None

        # result dictionary (only subcircuits with at least one vertex)
        subcircuits_data = {}

        print("Number of cuts:", pulp.value(self.K))

        print("Subcircuits (assigned vertices):")
        for c in self.subcircuits:
            # get vertices assigned to subcircuit c
            subcircuit_vertices = [v for v in self.vertices if pulp.value(self.y[v, c]) == 1]

            if not subcircuit_vertices:
                continue

            print(f"  Subcircuit {c}: {subcircuit_vertices}")

            subcircuits_data[c] = {
                "vertices": subcircuit_vertices,
                "shots": {}
            }

        # oopulate shot assignment info (only for valid subcircuits)
        for c in self.subcircuits:
            if c not in subcircuits_data:
                continue

            assignment = {}
            for q in self.qpus_index:
                val = pulp.value(self.shots_assign[(c, q)])
                if val > 0:
                    assignment[self.qpus[q].index] = int(val)

            print(f"Subcircuit {c} -> Assigned shots: {assignment}")
            subcircuits_data[c]["shots"] = assignment

        print("\n-- QPU Usage --")
        for q in self.qpus_index:
            t_q_val = pulp.value(self.T_q[q])
            if t_q_val > 0:
                total_shots = sum(pulp.value(self.shots_assign[(c, q)]) for c in self.subcircuits)
                print(f"QPU {self.qpus[q].index}: T_q = {t_q_val:.2f}, total shots = {total_shots}")

        T_val = pulp.value(self.T)
        print(f"\nMakespan T = {T_val:.2f}")

        # add subcircuit details
        for c in self.subcircuits:
            if c not in subcircuits_data:
                continue

            subcircuits_data[c]['capacity'] = pulp.value(self.d[c])
            subcircuits_data[c]['input_qubits'] = pulp.value(self.a[c])
            subcircuits_data[c]['init_qubits'] = pulp.value(self.p[c])
            subcircuits_data[c]['measured_qubits'] = pulp.value(self.o[c])
            subcircuits_data[c]['contributing_qubits'] = pulp.value(self.f[c])

            subcircuits_data[c].update({"cuts": {"out": [], "in": []}})
            for e in self.edges:
                if pulp.value(self.x[(e, c)]) > 0:
                    # if the edge is cut and the source belongs to subcircuit c, it's an out cut
                    if pulp.value(self.y[e[0], c]) == 1:
                        subcircuits_data[c]["cuts"]["out"].append(e[0])
                    # otherwise if the target belongs to subcircuit c, it's an in cut
                    else:
                        subcircuits_data[c]["cuts"]["in"].append(e[1])

        print("\nCuts per subcircuit:")
        for c, data in subcircuits_data.items():
            print(f"Subcircuit {c} - In: {data['cuts']['in']}, Out: {data['cuts']['out']}")

        return subcircuits_data
