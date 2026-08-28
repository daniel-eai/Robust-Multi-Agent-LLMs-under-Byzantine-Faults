import math
import os
import logging
import networkx as nx
import numpy as np
from random import Random
from .base_topology import BaseTopology

logger = logging.getLogger(__name__)


class RobustRandomTopology(BaseTopology):
    """r-robust random graph via Erdos-Renyi + MILP verification loop."""

    def __init__(self, num_nodes: int, seed: int = 42, **kwargs):
        super().__init__(num_nodes, "robust_random", **kwargs)
        _env_r = os.environ.get("TARGET_R")
        self.r = int(_env_r) if _env_r else math.ceil(num_nodes / 2)
        self.seed = seed
        self.build_topology()

    def build_topology(self):
        from r_robust import directed_milp_r_robustness

        n = self.num_nodes
        r = self.r
        nodes = self.get_all_nodes()

        prob = (np.log(n) + (r - 1) * np.log(np.log(n))) / n

        rng_seed = self.seed
        robust = False
        G = None
        while not robust:
            G = nx.erdos_renyi_graph(n, prob, seed=rng_seed, directed=False)
            robust = directed_milp_r_robustness(list(G.edges()), n) >= r
            rng_seed += 1

        for (i, j) in G.edges():
            if i < len(nodes) and j < len(nodes):
                self.graph.add_edge(nodes[i], nodes[j])

        logger.info(
            f"Robust-random topology built: {n} nodes, "
            f"{self.graph.number_of_edges()} edges, r={r}"
        )
        return self.graph

    def get_topology_metrics(self):
        connections = self.get_all_connections()
        total_edges = sum(len(v) for v in connections.values()) // 2
        n = self.num_nodes
        return {
            "topology_type": "robust_random",
            "node_count": n,
            "edge_count": total_edges,
            "r": self.r,
            "average_degree": 2 * total_edges / n if n > 0 else 0,
        }
