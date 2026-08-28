
import networkx as nx
import numpy as np
from .base_topology import BaseTopology

class CompleteGraphTopology(BaseTopology):

    def __init__(self, num_nodes: int, **kwargs):
        super().__init__(num_nodes, "complete", **kwargs)
        self.build_topology()

    def build_topology(self) -> nx.Graph:

        nodes = self.get_all_nodes()

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i < j:            
                    self.graph.add_edge(node1, node2)

        self.node_positions = self._calculate_circular_positions()

        self.logger.info(f"Complete graph topology built: {self.num_nodes} nodes, {self.graph.number_of_edges()} edges")
        return self.graph

    def _calculate_circular_positions(self) -> dict:

        positions = {}
        nodes = self.get_all_nodes()

        if self.num_nodes == 1:
            positions[nodes[0]] = (0, 0)
        else:
            angle_step = 2 * np.pi / self.num_nodes
            radius = 1.0

            for i, node in enumerate(nodes):
                angle = i * angle_step
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions[node] = (x, y)

        return positions

    def get_topology_properties(self) -> dict:

        properties = self.analyze_network_properties()

        expected_edges = self.num_nodes * (self.num_nodes - 1) // 2

        complete_graph_properties = {
            "is_complete": self.graph.number_of_edges() == expected_edges,
            "expected_edges": expected_edges,
            "actual_edges": self.graph.number_of_edges(),
            "node_degree": self.num_nodes - 1,           
            "redundancy": "maximum",         
            "fault_tolerance_level": "optimal"          
        }

        properties.update(complete_graph_properties)
        return properties

    def calculate_byzantine_resilience(self) -> dict:

        max_faults = (self.num_nodes - 1) // 3

        return {
            "topology_type": "complete",
            "max_byzantine_faults": max_faults,
            "resilience_ratio": max_faults / self.num_nodes if self.num_nodes > 0 else 0,
            "communication_complexity": "O(n²)",
            "consensus_rounds": 1,              
            "advantages": [
                "Highest fault tolerance",
                "Fastest consensus speed",
                "Strongest network resilience",
                "No single point of failure"
            ],
            "disadvantages": [
                "Highest communication overhead",
                "Poor scalability",
                "High resource consumption"
            ]
        }

    def simulate_consensus_process(self, malicious_ratio: float = 0.2) -> dict:

        malicious_nodes = self.set_malicious_nodes(malicious_ratio)

        consensus_simulation = {
            "topology": "complete",
            "total_nodes": self.num_nodes,
            "malicious_nodes": len(malicious_nodes),
            "communication_rounds": 1,
            "messages_per_round": self.num_nodes * (self.num_nodes - 1),
            "consensus_possible": len(malicious_nodes) <= (self.num_nodes - 1) // 3,
            "isolation_resistance": "maximum",            
            "partition_resistance": "maximum"           
        }

        return consensus_simulation 