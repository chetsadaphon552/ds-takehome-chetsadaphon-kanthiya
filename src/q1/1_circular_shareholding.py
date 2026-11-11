"""
Problem 1 - Discrete Mathematics: Shareholding Networks
Part 2: Circular Shareholding Detection using Strongly Connected Components (SCC)

Requirement:
- Custom implementation without graph libraries (NetworkX used only for validation).
- Time Complexity: O(|V| + |E|).
- Benchmarked on random directed graphs with 100,000 nodes.
"""

import sys
import time
import random
from collections import defaultdict

# Increase recursion depth if using recursive DFS, or use an explicit iterative stack
sys.setrecursionlimit(300000)


class ShareholdingGraph:
    """
    เก็บกราฟแบบมีทิศทางด้วย adjacency list เพื่อเข้าถึงเส้นเชื่อมของแต่ละบริษัทได้เร็ว
    """
    def __init__(self, num_nodes: int = 0):
        self.num_nodes = num_nodes
        self.adj = [[] for _ in range(num_nodes)]  # รายชื่อเพื่อนบ้านของแต่ละโหนด

    def add_edge(self, u: int, v: int) -> None:
        """Add directed edge u -> v (Company u directly owns shares in Company v)."""
        self.adj[u].append(v)

    def find_circular_shareholding_groups(self):
        """
        Tarjan's Strongly Connected Components (SCC) Algorithm.
        Time Complexity: O(|V| + |E|)
        Space Complexity: O(|V|)
        
        Returns:
            List[List[int]]: Groups of companies involved in circular shareholding
                            (SCCs with size > 1, or size == 1 with self-loops).
        """
        n = self.num_nodes
        index = 0
        indices = [-1] * n
        lowlink = [-1] * n
        on_stack = [False] * n
        stack = []
        sccs = []

        # ใช้ stack จำลองการเรียกซ้ำ เพื่อรองรับกราฟใหญ่โดยไม่ติด recursion limit
        for root in range(n):
            if indices[root] != -1:
                continue

            # call_stack เก็บตำแหน่งเพื่อนบ้านถัดไปที่ DFS ต้องตรวจของแต่ละโหนด
            call_stack = [(root, 0)]  # (node, next_neighbor_index)
            indices[root] = lowlink[root] = index
            index += 1
            stack.append(root)
            on_stack[root] = True

            while call_stack:
                u, edge_idx = call_stack[-1]
                neighbors = self.adj[u]

                if edge_idx < len(neighbors):
                    # Advance edge pointer
                    call_stack[-1] = (u, edge_idx + 1)
                    v = neighbors[edge_idx]

                    if indices[v] == -1:
                        # Successor v has not yet been visited; recurse on it
                        indices[v] = lowlink[v] = index
                        index += 1
                        stack.append(v)
                        on_stack[v] = True
                        call_stack.append((v, 0))
                    elif on_stack[v]:
                        # Successor v is in stack and hence in the current SCC
                        lowlink[u] = min(lowlink[u], indices[v])
                else:
                    # Finished searching all neighbors of u
                    call_stack.pop()
                    if call_stack:
                        parent, _ = call_stack[-1]
                        lowlink[parent] = min(lowlink[parent], lowlink[u])

                    # เมื่อ lowlink เท่ากับ index แปลว่า u เป็นรากของ SCC ให้ดึงสมาชิกออกจาก stack
                    if lowlink[u] == indices[u]:
                        current_scc = []
                        while True:
                            w = stack.pop()
                            on_stack[w] = False
                            current_scc.append(w)
                            if w == u:
                                break

                        # Filter: Circular shareholding means either:
                        # 1. Component has > 1 node (A -> B -> ... -> A)
                        # 2. Component has 1 node with a self-loop (A -> A)
                        if len(current_scc) > 1:
                            sccs.append(current_scc)
                        elif len(current_scc) == 1:
                            node = current_scc[0]
                            if node in self.adj[node]:
                                sccs.append(current_scc)

        return sccs


def generate_benchmark_graph(num_nodes: int = 100_000, num_edges: int = 250_000, seed: int = 42):
    """
    สร้างกราฟสุ่มแบบทำซ้ำได้ตาม seed เพื่อใช้ทดสอบอัลกอริทึมกับข้อมูลขนาดใหญ่
    """
    random.seed(seed)
    graph = ShareholdingGraph(num_nodes)

    # 1. Create a sparse random directed graph
    edges_set = set()
    while len(edges_set) < num_edges:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in edges_set:
            edges_set.add((u, v))
            graph.add_edge(u, v)

    return graph


if __name__ == "__main__":
    # รันทดสอบกับกราฟ 100,000 โหนดตามข้อกำหนด พร้อมรายงานเวลาและจำนวนกลุ่มที่พบ
    print("=" * 70)
    print("Problem 1 - Part 2: Circular Shareholding Detection (Tarjan's SCC)")
    print("=" * 70)

    N = 100_000
    M = 250_000
    print(f"Generating benchmark graph with |V| = {N:,} nodes and |E| = {M:,} edges...")
    t0 = time.time()
    g = generate_benchmark_graph(num_nodes=N, num_edges=M, seed=42)
    print(f"Graph generated in {time.time() - t0:.2f} seconds.")

    print("\nRunning custom Tarjan's SCC algorithm...")
    t1 = time.time()
    circular_groups = g.find_circular_shareholding_groups()
    duration = time.time() - t1

    print(f"Completed in {duration:.4f} seconds.")
    print(f"Total circular shareholding groups detected: {len(circular_groups)}")

    if circular_groups:
        sample_group = circular_groups[0]
        print(f"Sample circular group size: {len(sample_group)}")
        print(f"Sample members: {sample_group[:5]}...")

    print("\nTime Complexity Analysis: O(|V| + |E|)")
    print("Space Complexity Analysis: O(|V|)")
    print("=" * 70)
