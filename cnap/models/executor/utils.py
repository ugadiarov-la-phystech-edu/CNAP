import networkx as nx
import numpy as np
import torch
import math


class GenerateGraph:
    def __init__(self, graph_type, size, degree=None):
        self.graph_type = graph_type
        self.size = size
        self.degree = degree

    def generate_graph(self):
        if self.graph_type == "erdos-renyi":
            return self._erdos_renyi()
        elif self.graph_type == "barabasi-albert":
            return self._barabasi_albert()
        elif self.graph_type == "star":
            return self._star()
        elif self.graph_type == "caveman":
            return self._caveman()
        elif self.graph_type == "caterpillar":
            return self._caterpillar()
        elif self.graph_type == "lobster":
            return self._lobster()
        elif self.graph_type == "tree":
            return self._tree()
        elif self.graph_type == "grid":
            return self._grid()
        elif self.graph_type == "ladder":
            return self._ladder()
        elif self.graph_type == "line":
            return self._line()
        else:
            print("Undefined graph type.")

    def _erdos_renyi(self):
        return nx.fast_gnp_random_graph(self.size, self.degree / self.size, directed=False)

    def _barabasi_albert(self):
        return nx.barabasi_albert_graph(self.size, self.degree)

    def _star(self):
        return nx.star_graph(self.size - 1)

    def _caveman(self):
        # num_clique be as close as possible to clique_size
        for i in range(int(math.sqrt(self.size)), 0, -1):
            if self.size % i == 0:
                num_clique = i
                clique_size = self.size // i
                break
        return nx.caveman_graph(num_clique, clique_size)

    def _caterpillar(self):
        graph = nx.empty_graph(self.size)
        backbone = np.random.randint(low=1, high=self.size)
        # Generate the backbone stalk
        for i in range(1, backbone):
            graph.add_edge(i - 1, i)
        # Add edges between backbone stalk to rest of nodes
        for i in range(backbone, self.size):
            graph.add_edge(i, np.random.randint(backbone))
        return graph

    def _lobster(self):
        graph = nx.empty_graph(self.size)
        backbone = np.random.randint(low=1, high=self.size)
        branches = np.random.randint(low=backbone + 1, high=self.size + 1)
        # Generate the backbone stalk
        for i in range(1, backbone):
            graph.add_edge(i - 1, i)
        # Add branches to the backbone
        for i in range(backbone, branches):
            graph.add_edge(i, np.random.randint(backbone))
        # Attach rest of nodes to the branches
        for i in range(branches, self.size):
            graph.add_edge(i, np.random.randint(low=backbone, high=branches))
        return graph

    def _tree(self):
        return nx.random_powerlaw_tree(self.size, tries=10000)

    def _grid(self):
        # width be as close as possible to height
        for i in range(int(math.sqrt(self.size)), 0, -1):
            if self.size % i == 0:
                width = i
                height = self.size // i
                break
        return nx.grid_2d_graph(width, height)

    def _ladder(self):
        graph = nx.ladder_graph(self.size // 2)
        if not self.size & 1:
            graph.add_node(self.size - 1)
            graph.add_edge(0, self.size - 1)
        return graph

    def _line(self):
        return nx.path_graph(self.size)


def cartpole_graph(depth=10, delta=0.1, accel=0.05, thresh=0.5, initial_position=0):
    x = [initial_position]
    is_terminal = [False]
    links = {0: []}
    last_chd = list(x)
    last_ind = list([0])
    tail_ind = 1
    for d in range(depth - 1):
        nxt_x = []
        nxt_ind = []
        for i in range(len(last_chd)):
            nxt_pos = last_chd[i] + last_chd[i] * accel
            nxt_pos_1 = nxt_pos + delta
            x.append(nxt_pos_1)
            links[last_ind[i]].append(tail_ind)
            links[tail_ind] = []
            if nxt_pos_1 > thresh or nxt_pos_1 < -thresh:
                is_terminal.append(True)
            else:
                is_terminal.append(False)
                nxt_x.append(nxt_pos_1)
                nxt_ind.append(tail_ind)
            tail_ind += 1
            nxt_pos_2 = nxt_pos - delta
            x.append(nxt_pos_2)
            links[last_ind[i]].append(tail_ind)
            links[tail_ind] = []
            if nxt_pos_2 > thresh or nxt_pos_2 < -thresh:
                is_terminal.append(True)
            else:
                is_terminal.append(False)
                nxt_x.append(nxt_pos_2)
                nxt_ind.append(tail_ind)
            tail_ind += 1
        last_chd = list(nxt_x)
        last_ind = list(nxt_ind)

    for i in range(len(last_chd)):
        nxt_pos = last_chd[i] + last_chd[i] * accel
        nxt_pos_1 = nxt_pos + delta
        x.append(nxt_pos_1)
        is_terminal.append(True)
        links[last_ind[i]].append(tail_ind)
        links[tail_ind] = []
        tail_ind += 1
        nxt_pos_2 = nxt_pos + delta
        x.append(nxt_pos_2)
        is_terminal.append(True)
        links[last_ind[i]].append(tail_ind)
        links[tail_ind] = []
        tail_ind += 1

    for i in range(len(is_terminal)):
        if is_terminal[i]:
            assert len(links[i]) == 0

    P = torch.zeros((2, tail_ind, tail_ind))
    R = torch.zeros((tail_ind, 2))

    for i in range(len(x)):
        if is_terminal[i]:
            for j in range(2):
                P[j][i][i] = 1.0
                R[i][j] = 0.0
        else:
            for j in range(len(links[i])):
                P[j][i][links[i][j]] = 1.0
                if is_terminal[links[i][j]]:
                    R[i][j] = 0.0
                else:
                    R[i][j] = 1.0

    return P, R


def find_policy(p, r, discount, v):
    max_a, argmax_a = torch.max(r + discount * torch.einsum('ijk,k->ji', p, v), dim=1)
    return argmax_a


def deterministic_k_mdp(nb_states, nb_actions):
    P = torch.zeros(nb_actions, nb_states, nb_states)
    R = torch.randn(nb_states, nb_actions)
    for s in range(nb_states):
        for act in range(nb_actions):
            s_prime = np.random.choice(nb_states)
            P[act][s][s_prime] = 1.0
    return P, R