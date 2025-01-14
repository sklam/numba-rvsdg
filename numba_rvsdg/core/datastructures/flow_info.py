import dis

from typing import Set, Tuple, Dict, Sequence
from dataclasses import dataclass, field

from numba_rvsdg.core.datastructures.basic_block import PythonBytecodeBlock
from numba_rvsdg.core.datastructures.block_map import BlockMap
from numba_rvsdg.core.datastructures.labels import (
    Label,
    ControlLabelGenerator,
    PythonBytecodeLabel,
)
from numba_rvsdg.core.utils import (
    is_conditional_jump,
    _next_inst_offset,
    is_unconditional_jump,
    is_exiting,
    _prev_inst_offset,
)


@dataclass()
class FlowInfo:
    """FlowInfo converts Bytecode into a ByteFlow object (CFG)."""

    block_offsets: Set[int] = field(default_factory=set)
    """Marks starting offset of basic-block
    """

    jump_insts: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    """Contains jump instructions and their target offsets.
    """

    last_offset: int = field(default=0)
    """Offset of the last bytecode instruction.
    """

    clg: ControlLabelGenerator = field(
        default_factory=ControlLabelGenerator, compare=False
    )

    def _add_jump_inst(self, offset: int, targets: Sequence[int]):
        """Add jump instruction to FlowInfo."""
        for off in targets:
            assert isinstance(off, int)
            self.block_offsets.add(off)
        self.jump_insts[offset] = tuple(targets)

    @staticmethod
    def from_bytecode(bc: dis.Bytecode) -> "FlowInfo":
        """
        Build control-flow information that marks start of basic-blocks and
        jump instructions.
        """
        flowinfo = FlowInfo()

        for inst in bc:
            # Handle jump-target instruction
            if inst.offset == 0 or inst.is_jump_target:
                flowinfo.block_offsets.add(inst.offset)
            # Handle by op
            if is_conditional_jump(inst.opname):
                flowinfo._add_jump_inst(
                    inst.offset, (_next_inst_offset(inst.offset), inst.argval)
                )
            elif is_unconditional_jump(inst.opname):
                flowinfo._add_jump_inst(inst.offset, (inst.argval,))
            elif is_exiting(inst.opname):
                flowinfo._add_jump_inst(inst.offset, ())

        flowinfo.last_offset = inst.offset
        return flowinfo

    def build_basicblocks(self: "FlowInfo", end_offset=None) -> "BlockMap":
        """
        Build a graph of basic-blocks
        """
        offsets = sorted(self.block_offsets)
        # enumerate labels
        labels = dict(
            (offset, PythonBytecodeLabel(self.clg.new_index())) for offset in offsets
        )
        if end_offset is None:
            end_offset = _next_inst_offset(self.last_offset)
        bbmap = BlockMap(graph={}, clg=self.clg)
        for begin, end in zip(offsets, [*offsets[1:], end_offset]):
            label = labels[begin]
            targets: Tuple[Label, ...]
            term_offset = _prev_inst_offset(end)
            if term_offset not in self.jump_insts:
                # implicit jump
                targets = (labels[end],)
            else:
                targets = tuple(labels[o] for o in self.jump_insts[term_offset])
            block = PythonBytecodeBlock(
                label=label,
                begin=begin,
                end=end,
                _jump_targets=targets,
                backedges=(),
            )
            bbmap.add_block(block)
        return bbmap
