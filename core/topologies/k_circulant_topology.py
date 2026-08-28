import math
import os
import logging
from .base_topology import BaseTopology

logger = logging.getLogger(__name__)


class KCirkulantTopology(BaseTopology):
    """k-circulant graph topology. r = ceil(k/2), k = 2r-1."""

    def __init__(self, num_nodes: int, seed: int = 42, **kwargs):
        super().__init__(num_nodes, "k_circulant", **kwargs)
        _env_r = os.environ.get("TARGET_R")
        self.r = int(_env_r) if _env_r else math.ceil(num_nodes / 2)
        self.build_topology()

    def build_topology(self):
        nodes = self.get_all_nodes()
        n = self.num_nodes
        r = self.r
        k = 2 * r - 1

        edges = []
        for i in range(n):
            for j in range(1, k + 1):
                src, dst = i, (i + j) % n
                if (dst, src) not in edges:
                    edges.append((src, dst))

        for (i, j) in edges:
            if i < len(nodes) and j < len(nodes):
                self.graph.add_edge(nodes[i], nodes[j])

        logger.info(f"k-circulant topology built: {n} nodes, {self.graph.number_of_edges()} edges, k={k}")
        return self.graph

    def get_topology_metrics(self):
        connections = self.get_all_connections()
        total_edges = sum(len(v) for v in connections.values()) // 2
        n = self.num_nodes
        return {
            "topology_type": "k_circulant",
            "node_count": n,
            "edge_count": total_edges,
            "k": 2 * self.r - 1,
            "average_degree": 2 * total_edges / n if n > 0 else 0,
        }
