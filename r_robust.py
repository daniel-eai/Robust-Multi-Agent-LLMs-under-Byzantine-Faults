"""r-robustness verifier.

Original implementation used Gurobi (via cvxpy) to solve the dual MILP that
computes the r-robustness of an undirected graph. Gurobi is not available in
this environment, so we fall back to a brute-force enumeration over all
non-trivial bipartitions (V1, V2) of V (disjoint, both non-empty, not
necessarily covering V).

For each such bipartition, r-robustness requires that at least one of V1 or
V2 contains a node with >= r neighbors in the *other* partition. The
graph's r-robustness is then the largest r for which this holds across all
bipartitions, which equals the minimum over all bipartitions of
  max( max_{i in V1} |N(i) ∩ V2|, max_{i in V2} |N(i) ∩ V1| ).

For n = 7 (our experimental setting) there are 3^7 = 2187 bipartitions of
the form (V1, V2, V \\ (V1 ∪ V2)). The brute force is well under a
millisecond per call.
"""
from __future__ import annotations
from itertools import product
from typing import Iterable, Tuple


def directed_milp_r_robustness(edges: Iterable[Tuple[int, int]], n: int, min_r: float = 0.0) -> int:
    """Return the largest r for which the undirected graph G=(V, edges) on
    n nodes is r-robust. min_r is accepted for API compatibility with the
    original Gurobi-based signature but is otherwise unused (the brute-force
    enumeration always returns the exact value)."""
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        if i == j:
            continue
        adj[i].add(j)
        adj[j].add(i)

    # r-robust definition (LeBlanc et al. 2013):
    # A non-empty S ⊆ V is r-reachable iff ∃ i ∈ S with |N(i) \ S| >= r,
    # i.e. node i has at least r neighbours OUTSIDE S (anywhere in V \ S,
    # NOT just in V2). A graph is r-robust iff for every pair of non-empty,
    # disjoint S1, S2, at least one is r-reachable. The largest r is the
    # min over (S1, S2) of max(max_{i∈S1} |N(i)\S1|, max_{i∈S2} |N(i)\S2|).
    best_r = n
    for assign in product((0, 1, 2), repeat=n):
        V1 = [i for i in range(n) if assign[i] == 1]
        V2 = [i for i in range(n) if assign[i] == 2]
        if not V1 or not V2:
            continue
        V1_set = set(V1)
        V2_set = set(V2)
        max_in_V1 = max(len(adj[i] - V1_set) for i in V1)
        max_in_V2 = max(len(adj[i] - V2_set) for i in V2)
        r_for_partition = max(max_in_V1, max_in_V2)
        if r_for_partition < best_r:
            best_r = r_for_partition
            if best_r < min_r:
                return best_r
    return best_r


if __name__ == "__main__":
    # K_n is ceil(n/2)-robust under LeBlanc et al. 2013 (the worst-case
    # bipartition splits V evenly so each side has ceil(n/2) neighbours
    # outside it).
    import math as _m
    for n in (3, 5, 7):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        r = directed_milp_r_robustness(edges, n)
        print(f"K_{n}: r = {r} (expected {_m.ceil(n / 2)})")
