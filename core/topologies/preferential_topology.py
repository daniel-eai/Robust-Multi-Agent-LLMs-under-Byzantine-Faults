import math
import os
import logging
from .base_topology import BaseTopology

logger = logging.getLogger(__name__)


class PreferentialTopology(BaseTopology):
    """Preferential attachment r-robust graph.
    First (2r-1) nodes form a clique; remaining nodes each connect to r hub nodes."""

    def __init__(self, num_nodes: int, seed: int = 42, **kwargs):
        super().__init__(num_nodes, "preferential", **kwargs)
        _env_r = os.environ.get("TARGET_R")
        self.r = int(_env_r) if _env_r else math.ceil(num_nodes / 2)
        self.build_topology()

    def build_topology(self):
        nodes = self.get_all_nodes()
        n = self.num_nodes
        r = self.r

        edges = []
        hub = 2 * r - 1
        for i in range(hub):
            for j in range(i + 1, hub):
                edges.append((i, j))
        for i in range(hub, n):
            for j in range(r):
                edges.append((i, j))

        for (i, j) in edges:
            if i < len(nodes) and j < len(nodes):
                self.graph.add_edge(nodes[i], nodes[j])

        logger.info(f"Preferential topology built: {n} nodes, {self.graph.number_of_edges()} edges, r={r}")
        return self.graph

    def get_topology_metrics(self):
        connections = self.get_all_connections()
        total_edges = sum(len(v) for v in connections.values()) // 2
        n = self.num_nodes
        return {
            "topology_type": "preferential",
            "node_count": n,
            "edge_count": total_edges,
            "r": self.r,
            "average_degree": 2 * total_edges / n if n > 0 else 0,
        }
