from dataclasses import dataclass
from enum import IntEnum
from pprint import pprint
from io import StringIO
from typing import IO
import random
import textwrap

from mock_asm import ProgramGen, parse, VM, Inst, GotoOperands, BrCtrOperands


def test_mock_asm():
    asm = textwrap.dedent("""
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr A B
        label B
            print B
    """)

    instlist = parse(asm)
    assert instlist[0].operands.text == "Start"
    assert instlist[1].operands.jump_target == 2
    assert instlist[2].operands.text == "A"
    assert instlist[3].operands.counter == 10
    assert instlist[4].operands.true_target == 2
    assert instlist[4].operands.false_target == 5
    assert instlist[5].operands.text == "B"

    with StringIO() as buf:
        VM(buf).run(instlist)
        got = buf.getvalue().split()

    expected = ["Start", *(["A"] * 10), "B"]
    assert got == expected


def test_double_exchange_loop():
    asm = textwrap.dedent("""
            print Start
       label A
            print A
            ctr 4
            brctr B Exit
        label B
            print B
            ctr 5
            brctr A Exit
        label Exit
            print Exit
    """)
    instlist = parse(asm)
    with StringIO() as buf:
        VM(buf).run(instlist)
        got = buf.getvalue().split()

    expected = ["Start", *(["A", "B"] * 3), "A", "Exit"]
    assert got == expected


def test_program_gen():
    rng = random.Random(123)
    pg = ProgramGen(rng)
    ct_term = 0
    total = 10000
    for i in range(total):
        print(str(i).center(80, "="))
        asm = pg.generate_program()


        instlist = parse(asm)
        with StringIO() as buf:
            terminated = VM(buf).run(instlist,
                                     max_step=1000)
            got = buf.getvalue().split()
            if terminated:
                print(asm)
                print(got)
                ct_term += 1
    print("terminated", ct_term, "total", total)

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.labels import Label
from numba_rvsdg.core.datastructures.basic_block import BasicBlock, block_types
from numba_rvsdg.core.transformations import (
    restructure_loop,
    restructure_branch,
    join_returns,
)


@dataclass(frozen=True, order=True)
class MockAsmLabel(Label):
    pass

@dataclass(frozen=True)
class MockAsmBasicBlock(BasicBlock):
    bbinstlist: list[Inst]
    bboffset: int


# NOTE: odd that this needs to be registered
block_types["mockasm"] = MockAsmBasicBlock

from numba_rvsdg.core.datastructures.labels import (
    ControlLabel,
)
from numba_rvsdg.core.datastructures.basic_block import (
    ControlVariableBlock,
    BranchBlock,
)
# NOTE: modified Renderer to be more general
class Renderer(object):
    def __init__(self, scfg: SCFG):
        from graphviz import Digraph

        self.g = Digraph()
        self.scfg = scfg

        self.rendered_blocks = set()
        self.render_region(self.g, None)
        self.render_edges()

    def render_basic_block(self, graph, block_name: str):
        body = str(block_name)
        graph.node(str(block_name), shape="rect", label=body)

    def render_branching_block(self, graph, block_name: str):
        block = self.scfg[block_name]

        if isinstance(block.label, ControlLabel):

            def find_index(v):
                if hasattr(v, "offset"):
                    return v.offset
                if hasattr(v, "index"):
                    return v.index

            body = str(block_name) + "\l"
        else:
            raise Exception("Unknown label type: " + block.label)
        graph.node(str(block_name), shape="rect", label=body)

    def render_region(self, graph, region_name):
        # If region name is none, we're in the 'root' region
        # that is the graph itself.
        if region_name is None:
            region_name = self.scfg.meta_region
            region = self.scfg.regions[region_name]
        else:
            region = self.scfg.regions[region_name]

        with graph.subgraph(name=f"cluster_{region_name}") as subg:
            kind = region.kind
            if kind == "loop":
                color = "blue"
            elif kind == "head":
                color = "red"
            elif kind == "branch":
                color = "green"
            elif kind == "tail":
                color = "purple"
            else:
                color = "black"
            subg.attr(color=color, label=str(region.region_name))

            for sub_region in self.scfg.region_tree[region_name]:
                self.render_region(subg, sub_region)

            # If there are no further subregions then we render the blocks
            all_blocks, _ = self.scfg.block_view(region_name)
            for block_name in all_blocks:
                self.render_block(subg, block_name)

    def render_block(self, graph, block_name):
        if block_name in self.rendered_blocks:
            return

        block = self.scfg[block_name]
        if type(block) == ControlVariableBlock:
            self.render_control_variable_block(graph, block_name)
        elif type(block) == BranchBlock:
            self.render_branching_block(graph, block_name)
        elif isinstance(block, BasicBlock):
            self.render_basic_block(graph, block_name)
        else:
            raise Exception("unreachable")
        self.rendered_blocks.add(block_name)

    def render_edges(self):
        for block_name, out_edges in self.scfg.out_edges.items():
            for out_edge in out_edges:
                if (block_name, out_edge) in self.scfg.back_edges:
                    self.g.edge(
                        str(block_name),
                        str(out_edge),
                        style="dashed",
                        color="grey",
                        constraint="0",
                    )
                else:
                    self.g.edge(str(block_name), str(out_edge))

    def view(self, *args):
        self.g.view(*args)



class MockAsmRenderer(Renderer):
    def render_basic_block(self, graph, block_name: str):
        block = self.scfg.blocks[block_name]
        if isinstance(block, MockAsmBasicBlock):
            end = r"\l"
            lines = [
                f"offset: {block.bboffset} | {block_name} ",
                *[str(inst) for inst in block.bbinstlist],
            ]
            body = ''.join([ln + end for ln in lines])
            graph.node(str(block_name), shape="rect", label=body)
        else:
            super().render_basic_block(graph, block_name)


def to_scfg(instlist: list[Inst]) -> SCFG:
    labels = set([0, len(instlist)])
    for inst in instlist:
        if isinstance(inst.operands, GotoOperands):
            labels.add(inst.operands.jump_target)
        elif isinstance(inst.operands, BrCtrOperands):
            labels.add(inst.operands.true_target)
            labels.add(inst.operands.false_target)

    scfg = SCFG()
    bb_offsets =sorted(labels)
    labelmap = {}
    bbmap = {}
    edgemap = {}
    for begin, end in zip(bb_offsets, bb_offsets[1:]):
        bb = instlist[begin:end]
        labelmap[begin] = label = MockAsmLabel()
        name = scfg.add_block(
            block_type="mockasm",
            block_label=label,
            bbinstlist=bb,
            bboffset=begin,
        )
        edgemap[begin] = name
        bbmap[name] = bb

    for name, bb in bbmap.items():
        inst = bb[-1]  # terminator
        if isinstance(inst.operands, GotoOperands):
            targets = [inst.operands.jump_target]
        elif isinstance(inst.operands, BrCtrOperands):
            targets = [inst.operands.true_target,
                       inst.operands.false_target]
        else:
            targets = None
        if targets:
            scfg.add_connections(name, [edgemap[tgt] for tgt in targets])
    scfg.check_graph()

    MockAsmRenderer(scfg).view()
    join_returns(scfg)
    restructure_loop(scfg)
    restructure_branch(scfg)
    return scfg


def test_mock_scfg_loop():
    asm = textwrap.dedent("""
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr A B
        label B
            print B
    """)

    instlist = parse(asm)
    scfg = to_scfg(instlist)


def test_mock_scfg_basic():
    asm = textwrap.dedent("""
        label S
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr S B
        label B
            print B
    """)

    instlist = parse(asm)
    scfg = to_scfg(instlist)

def test_mock_scfg_diamond():
    asm = textwrap.dedent("""
            print Start
            ctr 1
            brctr A B
        label A
            print A
            goto C
        label B
            print B
            goto C
        label C
            print C
    """)

    instlist = parse(asm)
    scfg = to_scfg(instlist)