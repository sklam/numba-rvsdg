
from unittest import main
from textwrap import dedent
from numba_rvsdg.core.datastructures.block_map import BlockMap

from numba_rvsdg.tests.test_utils import MapComparator


class TestBlockMapConversion(MapComparator):

    def test_yaml_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = ["""
            "0":
                jt: ["1", "2"]
            "1":
                jt: ["3"]
            "2":
                jt: ["4"]
            "3":
                jt: ["4"]
            "4":
                jt: []""",
        # Case # 2: Cyclic graph, no back edges
            """
            "0":
                jt: ["1", "2"]
            "1":
                jt: ["5"]
            "2":
                jt: ["1", "5"]
            "3":
                jt: ["0"]
            "4":
                jt: []
            "5":
                jt: ["3", "4"]""",
        # Case # 3: Graph with backedges
            """
            "0":
                jt: ["1"]
            "1":
                jt: ["2", "3"]
            "2":
                jt: ["4"]
            "3":
                jt: []
            "4":
                jt: ["2", "3"]
                be: ["2"]"""]

        for case in cases:
            case = dedent(case)
            block_map = BlockMap.from_yaml(case)
            self.assertEqual(case, block_map.to_yaml())
    
    def test_dict_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [{
    "0":
        {"jt": ["1", "2"]},
    "1":
        {"jt": ["3"]},
    "2":
        {"jt": ["4"]},
    "3":
        {"jt": ["4"]},
    "4":
        {"jt": []}},
        # Case # 2: Cyclic graph, no back edges
        {
    "0":
        {"jt": ["1", "2"]},
    "1":
        {"jt": ["5"]},
    "2":
        {"jt": ["1", "5"]},
    "3":
        {"jt": ["0"]},
    "4":
        {"jt": []},
    "5":
        {"jt": ["3", "4"]}},
        # Case # 3: Graph with backedges
        {
    "0":
        {"jt": ["1"]},
    "1":
        {"jt": ["2", "3"]},
    "2":
        {"jt": ["4"]},
    "3":
        {"jt": []},
    "4":
        {"jt": ["2", "3"],
        "be": ["2"]}}]

        for case in cases:
            block_map = BlockMap.from_dict(case)
            self.assertEqual(case, block_map.to_dict())

if __name__ == "__main__":
    main()
           
