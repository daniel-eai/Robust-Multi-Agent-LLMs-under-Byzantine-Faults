#!/usr/bin/env python3
"""
gamma-MERG (Maximum Edge Robust Graph) topology.

Constructs a ceil(n/2)-robust graph with the minimum number of edges,
based on Lee & Panagou (2026b): "Minimal construction of graphs with
maximum robustness."

For SAC (Proposition 4.4), the graph must be (F+1)-robust.
gamma-MERG gives gamma = ceil(n/2)-robustness, so it supports
F up to gamma-1 = ceil(n/2)-1 Byzantine agents under the F-local model.

Recommended setup: n >= 4F+1 (satisfies the stronger total-model condition).
"""

import math
import itertools
import logging
from random import Random
from .base_topology import BaseTopology

logger = logging.getLogger(__name__)


def _build_merg_edges(n: int, seed: int = 42):
    """Return edge list for gamma-MERG with n nodes."""
    rng = Random(seed)

    if n % 2 == 1:
        return _merg_odd(n, rng)
    else:
        return _merg_even(n, rng)


def _merg_odd(n: int, rng: Random):
    gamma = math.ceil(n / 2)
    edges = []

    # (gamma+1)-clique on nodes 0..gamma
    for i in range(gamma + 1):
        for j in range(i + 1, gamma + 1):
            edges.append((i, j))

    # Remaining nodes each connect to gamma distinct nodes in the clique
    clique_nodes = list(range(gamma + 1))
    for i in range(gamma + 1, n):
        chosen = rng.sample(clique_nodes, gamma)
        for x in chosen:
            edges.append((i, x))

    return edges


def _merg_even(n: int, rng: Random):
    gamma = n // 2
    edges = []
    delta = gamma - 2 if gamma % 2 != 0 else gamma - 3

    # Connect gamma nodes in X to all others, skipping delta/2 edges
    for i in range(gamma):
        for j in range(i + 1, 2 * gamma):
            if i % 2 == 0 and j == i + 1 and i <= delta:
                continue
            edges.append((i, j))

    # Add ceil(gamma/2) node-disjoint non-edges
    extra_pairs = math.ceil(gamma / 2)
    existing = set(edges)
    used_nodes = set()
    added = 0

    all_pairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(all_pairs)
    available = [p for p in all_pairs if p not in existing]

    for (i, j) in available:
        if i not in used_nodes and j not in used_nodes:
            edges.append((i, j))
            used_nodes.add(i)
            used_nodes.add(j)
            added += 1
            if added == extra_pairs:
                break

    return edges


class MERGTopology(BaseTopology):
    """gamma-MERG topology (maximum robustness with minimum edges)."""

    def __init__(self, num_nodes: int, seed: int = 42, **kwargs):
        super().__init__(num_nodes, "merg", **kwargs)
        self.seed = seed
        self.gamma = math.ceil(num_nodes / 2)
        self.build_topology()

    def build_topology(self):
        nodes = self.get_all_nodes()
        node_list = list(range(self.num_nodes))

        edges = _build_merg_edges(self.num_nodes, self.seed)

        for (i, j) in edges:
            if i < len(nodes) and j < len(nodes):
                self.graph.add_edge(nodes[i], nodes[j])

        logger.info(
            f"gamma-MERG topology built: {self.num_nodes} nodes, "
            f"{self.graph.number_of_edges()} edges, gamma={self.gamma}"
        )
        return self.graph

    def get_topology_metrics(self):
        import networkx as nx
        connections = self.get_all_connections()
        total_edges = sum(len(v) for v in connections.values()) // 2
        n = self.num_nodes
        return {
            "topology_type": "merg",
            "node_count": n,
            "edge_count": total_edges,
            "gamma_robustness": self.gamma,
            "max_byzantine_F_local": self.gamma - 1,
            "average_degree": 2 * total_edges / n if n > 0 else 0,
        }
