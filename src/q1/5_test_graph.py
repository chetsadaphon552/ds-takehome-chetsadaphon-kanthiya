"""
Problem 1 - Discrete Mathematics: Shareholding Networks
Unit tests and verification for circular shareholding detection.
"""

import unittest
import sys
import os

# Import from renamed module (1_circular_shareholding.py)
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module
circular_shareholding = import_module('1_circular_shareholding')
ShareholdingGraph = circular_shareholding.ShareholdingGraph


class TestShareholdingGraph(unittest.TestCase):
    # กราฟไม่มีโหนดต้องคืนผลลัพธ์ว่าง
    def test_empty_graph(self):
        # กราฟว่าง ต้องไม่มี SCC
        g = ShareholdingGraph(0)
        self.assertEqual(g.find_circular_shareholding_groups(), [])

    # เส้นทางที่ไม่มีวงจรต้องไม่ถูกรายงานเป็น circular shareholding
    def test_acyclic_graph(self):
        # DAG (0->1->2) ไม่มี cycle ต้องไม่มี SCC
        g = ShareholdingGraph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        self.assertEqual(g.find_circular_shareholding_groups(), [])

    # วงจรหลายบริษัทต้องถูกรวมเป็น SCC เดียว
    def test_simple_cycle(self):
        # Cycle 0->1->2->0 ต้องพบ 1 SCC ที่มีสมาชิก [0,1,2]
        g = ShareholdingGraph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        sccs = g.find_circular_shareholding_groups()
        self.assertEqual(len(sccs), 1)
        self.assertCountEqual(sccs[0], [0, 1, 2])

    # self-loop ถือเป็นวงจร แม้มีบริษัทเดียว
    def test_self_loop(self):
        # Self-loop (0->0) ถือเป็น circular shareholding ด้วย
        g = ShareholdingGraph(2)
        g.add_edge(0, 0)
        g.add_edge(0, 1)
        sccs = g.find_circular_shareholding_groups()
        self.assertEqual(len(sccs), 1)
        self.assertEqual(sccs[0], [0])


if __name__ == "__main__":
    unittest.main()
