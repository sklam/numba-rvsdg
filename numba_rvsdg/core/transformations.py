from collections import defaultdict
from typing import Set, Dict, List

from numba_rvsdg.core.datastructures.labels import (
    Label,
    SyntheticForIter,
    SyntheticBranch,
    SyntheticHead,
    SyntheticExitingLatch,
    SyntheticExit,
    SynthenticAssignment,
    PythonBytecodeLabel,
)
from numba_rvsdg.core.datastructures.block_map import BlockMap
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    ControlVariableBlock,
    BranchBlock,
    RegionBlock,
)

from numba_rvsdg.core.utils import _logger


def loop_restructure_helper(bbmap: BlockMap, loop: Set[Label]):
    """Loop Restructuring

    Applies the algorithm LOOP RESTRUCTURING from section 4.1 of Bahmann2015.

    Note that this will modify both the `bbmap` and the `loop` in-place.

    Parameters
    ----------
    bbmap: BlockMap
        The BlockMap containing the loop
    loop: Set[Label]
        The loop (strongly connected components) that is to be restructured

    """

    headers, entries = bbmap.find_headers_and_entries(loop)
    exiting_blocks, exit_blocks = bbmap.find_exiting_and_exits(loop)
    #assert len(entries) == 1
    headers_were_unified = False

    # If there are multiple headers, insert assignment and control blocks,
    # such that only a single loop header remains.
    if len(headers) > 1:
        headers_were_unified = True
        solo_head_label = SyntheticHead(str(bbmap.clg.new_index()))
        bbmap.insert_block_and_control_blocks(solo_head_label, entries, headers)
        loop.add(solo_head_label)
        loop_head: Label = solo_head_label
    else:
        loop_head: Label = next(iter(headers))
    # If there is only a single exiting latch (an exiting block that also has a
    # backedge to the loop header) we can exit early, since the condition for
    # SCFG is fullfilled.
    backedge_blocks = [
        block for block in loop if headers.intersection(bbmap[block].jump_targets)
    ]
    if (len(backedge_blocks) == 1 and len(exiting_blocks) == 1
        and backedge_blocks[0] == next(iter(exiting_blocks))):
        bbmap.add_block(bbmap.graph.pop(backedge_blocks[0]).replace_backedge(loop_head))
        return

    # The synthetic exiting latch and synthetic exit need to be created
    # based on the state of the cfg. If there are multiple exits, we need a
    # SyntheticExit, otherwise we only need a SyntheticExitingLatch
    synth_exiting_latch = SyntheticExitingLatch(str(bbmap.clg.new_index()))
    # Set a flag, this will determine the variable assignment and block
    # insertion later on
    needs_synth_exit = len(exit_blocks) > 1
    if needs_synth_exit:
        synth_exit = SyntheticExit(str(bbmap.clg.new_index()))

    # This sets up the various control variables.
    # If there were multiple headers, we must re-use the variable that was used
    # for looping as the exit variable
    if headers_were_unified:
        exit_variable = bbmap[solo_head_label].variable
    else:
        exit_variable = bbmap.clg.new_variable()
    # This variable denotes the backedge
    backedge_variable = bbmap.clg.new_variable()
    # Now we setup the lookup tables for the various control variables,
    # depending on the state of the CFG and what is needed
    exit_value_table = dict(((i, j) for i, j in enumerate(exit_blocks)))
    if needs_synth_exit:
        backedge_value_table = dict((i, j) for i, j in enumerate((loop_head, synth_exit)))
    else:
        backedge_value_table = dict((i, j) for i, j in enumerate((loop_head, next(iter(exit_blocks)))))
    if headers_were_unified:
        header_value_table = bbmap[solo_head_label].branch_value_table
    else:
        header_value_table = {}

    # This does a dictionary reverse lookup, to determine the key for a given
    # value.
    def reverse_lookup(d, value):
        for k, v in d.items():
            if v == value:
                return k
        else:
            return "UNUSED"

    # Now that everything is in place, we can start to insert blocks, depending
    # on what is needed
    # All new blocks are recorded for later insertion into the loop set
    new_blocks = set()
    doms = _doms(bbmap)
    # For every block in the loop:
    for label in sorted(loop, key=lambda x: x.index):
        # If the block is an exiting block or a backedge block
        if label in exiting_blocks or label in backedge_blocks:
            # Copy the jump targets, these will be modified
            new_jt = list(bbmap[label].jump_targets)
            # For each jump_target in the blockj
            for jt in bbmap[label].jump_targets:
                # If the target is an exit block
                if jt in exit_blocks:
                    # Create a new assignment label and record it
                    synth_assign = SynthenticAssignment(str(bbmap.clg.new_index()))
                    new_blocks.add(synth_assign)
                    # Setup the table for the variable assignment
                    variable_assignment = {}
                    # Setup the variables in the assignment table to point to
                    # the correct blocks
                    if needs_synth_exit:
                        variable_assignment[exit_variable]  = reverse_lookup(exit_value_table, jt)
                    variable_assignment[backedge_variable]  = reverse_lookup(backedge_value_table,
                        synth_exit if needs_synth_exit else next(iter(exit_blocks)))
                    # Create the actual control variable block
                    synth_assign_block = ControlVariableBlock(
                        label=synth_assign,
                        _jump_targets=(synth_exiting_latch,),
                        backedges=(),
                        variable_assignment=variable_assignment,
                    )
                    # Insert the assignment to the block map
                    bbmap.add_block(synth_assign_block)
                    # Insert the new block into the new jump_targets making
                    # sure, that it replaces the correct jump_target, order
                    # matters in this case.
                    new_jt[new_jt.index(jt)] = synth_assign
                # If the target is the loop_head
                elif jt in headers and label not in doms[jt]:
                    # Create the assignment and record it
                    synth_assign = SynthenticAssignment(str(bbmap.clg.new_index()))
                    new_blocks.add(synth_assign)
                    # Setup the variables in the assignment table to point to
                    # the correct blocks
                    variable_assignment = {}
                    variable_assignment[backedge_variable] = reverse_lookup(backedge_value_table, loop_head)
                    if needs_synth_exit:
                        variable_assignment[exit_variable] = reverse_lookup(header_value_table, jt)
                    # Update the backedge block - remove any existing backedges
                    # that point to the headers, no need to add a backedge,
                    # since it will be contained in the SyntheticExitingLatch
                    # later on.
                    block = bbmap.graph.pop(label)
                    jts = list(block.jump_targets)
                    for h in headers:
                        if h in jts:
                            jts.remove(h)
                    bbmap.add_block(block.replace_jump_targets(jump_targets=tuple(jts)))
                    # Setup the assignment block and initialize it with the
                    # correct jump_targets and variable assignment.
                    synth_assign_block = ControlVariableBlock(
                        label=synth_assign,
                        _jump_targets=(synth_exiting_latch,),
                        backedges=(),
                        variable_assignment=variable_assignment,
                    )
                    # Add the new block to the BlockMap
                    bbmap.add_block(synth_assign_block)
                    # Update the jump targets again, order matters
                    new_jt[new_jt.index(jt)] = synth_assign
            # finally, replace the jump_targets for this block with the new ones
            bbmap.add_block(
                bbmap.graph.pop(label).replace_jump_targets(jump_targets=tuple(new_jt))
            )
    # Add any new blocks to the loop.
    loop.update(new_blocks)

    # Insert the exiting latch, add it to the loop and to the graph.
    synth_exiting_latch_block = BranchBlock(
        label=synth_exiting_latch,
        _jump_targets=(synth_exit if needs_synth_exit else next(iter(exit_blocks)), loop_head),
        backedges=(loop_head,),
        variable=backedge_variable,
        branch_value_table=backedge_value_table,
    )
    loop.add(synth_exiting_latch)
    bbmap.add_block(synth_exiting_latch_block)
    # If an exit is to be created, we do so too, but only add it to the bbmap,
    # since it isn't part of the loop
    if needs_synth_exit:
        synth_exit_block = BranchBlock(
            label=synth_exit,
            _jump_targets=tuple(exit_blocks),
            backedges=(),
            variable=exit_variable,
            branch_value_table=exit_value_table,
        )
        bbmap.add_block(synth_exit_block)


def restructure_loop(bbmap: BlockMap):
    """Inplace restructuring of the given graph to extract loops using
    strongly-connected components
    """
    # obtain a List of Sets of Labels, where all labels in each set are strongly
    # connected, i.e. all reachable from one another by traversing the subset
    scc: List[Set[Label]] = bbmap.compute_scc()
    # loops are defined as strongly connected subsets who have more than a
    # single label and single label loops that point back to to themselves.
    loops: List[Set[Label]] = [
        nodes
        for nodes in scc
        if len(nodes) > 1 or next(iter(nodes)) in bbmap[next(iter(nodes))].jump_targets
    ]

    _logger.debug(
        "restructure_loop found %d loops in %s", len(loops), bbmap.graph.keys()
    )
    # rotate and extract loop
    for loop in loops:
        loop_restructure_helper(bbmap, loop)
        extract_region(bbmap, loop, "loop")


def find_head_blocks(bbmap: BlockMap, begin: Label) -> Set[Label]:
    head = bbmap.find_head()
    head_region_blocks = set()
    current_block = head
    # Start at the head block and traverse the graph linearly until
    # reaching the begin block.
    while True:
        head_region_blocks.add(current_block)
        if current_block == begin:
            break
        else:
            jt = bbmap.graph[current_block].jump_targets
            assert len(jt) == 1
            current_block = next(iter(jt))
    return head_region_blocks


def find_branch_regions(bbmap: BlockMap, begin: Label, end: Label) -> Set[Label]:
    # identify branch regions
    doms = _doms(bbmap)
    postdoms = _post_doms(bbmap)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    branch_regions = []
    jump_targets = bbmap.graph[begin].jump_targets
    for bra_start in jump_targets:
        for jt in jump_targets:
            if jt != bra_start and bbmap.is_reachable_dfs(jt, bra_start):
                branch_regions.append(tuple())
                break
        else:
            sub_keys: Set[PythonBytecodeLabel] = set()
            branch_regions.append((bra_start, sub_keys))
            # a node is part of the branch if
            # - the start of the branch is a dominator; and,
            # - the tail of the branch is not a dominator
            for k, kdom in doms.items():
                if bra_start in kdom and end not in kdom:
                    sub_keys.add(k)
    return branch_regions


def _find_branch_regions(bbmap: BlockMap, begin: Label, end: Label) -> Set[Label]:
    # identify branch regions
    branch_regions = []
    for bra_start in bbmap[begin].jump_targets:
        region = []
        region.append(bra_start)
    return branch_regions


def find_tail_blocks(
    bbmap: BlockMap, begin: Set[Label], head_region_blocks, branch_regions
):
    tail_subregion = set((b for b in bbmap.graph.keys()))
    tail_subregion.difference_update(head_region_blocks)
    for reg in branch_regions:
        if not reg:
            continue
        b, sub = reg
        tail_subregion.discard(b)
        for s in sub:
            tail_subregion.discard(s)
    # exclude parents
    tail_subregion.discard(begin)
    return tail_subregion


def extract_region(bbmap, region_blocks, region_kind):
    headers, entries = bbmap.find_headers_and_entries(region_blocks)
    exiting_blocks, exit_blocks = bbmap.find_exiting_and_exits(region_blocks)
    assert len(headers) == 1
    assert len(exiting_blocks) == 1
    region_header = next(iter(headers))
    region_exiting = next(iter(exiting_blocks))

    head_subgraph = BlockMap(
        {label: bbmap.graph[label] for label in region_blocks}, clg=bbmap.clg
    )

    if isinstance(bbmap[region_exiting], RegionBlock):
        region_exit = bbmap[region_exiting].exit
    else:
        region_exit = region_exiting

    subregion = RegionBlock(
        label=region_header,
        _jump_targets=bbmap[region_exiting].jump_targets,
        backedges=(),
        kind=region_kind,
        headers=headers,
        subregion=head_subgraph,
        exit=region_exit,
    )
    bbmap.remove_blocks(region_blocks)
    bbmap.graph[region_header] = subregion


def restructure_branch(bbmap: BlockMap):
    print("restructure_branch", bbmap.graph)
    doms = _doms(bbmap)
    postdoms = _post_doms(bbmap)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    regions = [r for r in _iter_branch_regions(bbmap, immdoms, postimmdoms)]

    # Early exit when no branching regions are found.
    # TODO: the whole graph should become a linear mono head
    if not regions:
        return

    # Compute initial regions.
    begin, end = regions[0]
    head_region_blocks = find_head_blocks(bbmap, begin)
    branch_regions = find_branch_regions(bbmap, begin, end)
    tail_region_blocks = find_tail_blocks(
        bbmap, begin, head_region_blocks, branch_regions
    )

    # Unify headers of tail subregion if need be.
    headers, entries = bbmap.find_headers_and_entries(tail_region_blocks)
    if len(headers) > 1:
        end = SyntheticHead(bbmap.clg.new_index())
        bbmap.insert_block_and_control_blocks(end, entries, headers)

    # Recompute regions.
    head_region_blocks = find_head_blocks(bbmap, begin)
    branch_regions = find_branch_regions(bbmap, begin, end)
    tail_region_blocks = find_tail_blocks(
        bbmap, begin, head_region_blocks, branch_regions
    )

    # Branch region processing:
    # Close any open branch regions by inserting a SyntheticTail.
    # Populate any empty branch regions by inserting a SyntheticBranch.
    for region in branch_regions:
        if region:
            bra_start, inner_nodes = region
            if inner_nodes:
                # Insert SyntheticTail
                exiting_blocks, _ = bbmap.find_exiting_and_exits(inner_nodes)
                tail_headers, _ = bbmap.find_headers_and_entries(tail_region_blocks)
                _, _ = bbmap.join_tails_and_exits(exiting_blocks, tail_headers)

        else:
            # Insert SyntheticBranch
            tail_headers, _ = bbmap.find_headers_and_entries(tail_region_blocks)
            synthetic_branch_block_label = SyntheticBranch(str(bbmap.clg.new_index()))
            bbmap.insert_block(synthetic_branch_block_label, (begin,), tail_headers)

    # Recompute regions.
    head_region_blocks = find_head_blocks(bbmap, begin)
    branch_regions = find_branch_regions(bbmap, begin, end)
    tail_region_blocks = find_tail_blocks(
        bbmap, begin, head_region_blocks, branch_regions
    )

    # extract subregions
    extract_region(bbmap, head_region_blocks, "head")
    for region in branch_regions:
        if region:
            bra_start, inner_nodes = region
            if inner_nodes:
                extract_region(bbmap, inner_nodes, "branch")
    extract_region(bbmap, tail_region_blocks, "tail")


def _iter_branch_regions(
    bbmap: BlockMap, immdoms: Dict[Label, Label], postimmdoms: Dict[Label, Label]
):
    for begin, node in [i for i in bbmap.graph.items()]:
        if len(node.jump_targets) > 1:
            # found branch
            if begin in postimmdoms:
                end = postimmdoms[begin]
                if immdoms[end] == begin:
                    yield begin, end


def _imm_doms(doms: Dict[Label, Set[Label]]) -> Dict[Label, Label]:
    idoms = {k: v - {k} for k, v in doms.items()}
    changed = True
    while changed:
        changed = False
        for k, vs in idoms.items():
            nstart = len(vs)
            for v in list(vs):
                vs -= idoms[v]
            if len(vs) < nstart:
                changed = True
    # fix output
    out = {}
    for k, vs in idoms.items():
        if vs:
            [v] = vs
            out[k] = v
    return out


def _doms(bbmap: BlockMap):
    # compute dom
    entries = set()
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    node: BasicBlock
    for src, node in bbmap.graph.items():
        for dst in node.jump_targets:
            # check dst is in subgraph
            if dst in bbmap.graph:
                preds_table[dst].add(src)
                succs_table[src].add(dst)

    for k in bbmap.graph:
        if not preds_table[k]:
            entries.add(k)
    return _find_dominators_internal(
        entries, list(bbmap.graph.keys()), preds_table, succs_table
    )


def _post_doms(bbmap: BlockMap):
    # compute post dom
    entries = set()
    for k, v in bbmap.graph.items():
        targets = set(v.jump_targets) & set(bbmap.graph)
        if not targets:
            entries.add(k)
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    node: BasicBlock
    for src, node in bbmap.graph.items():
        for dst in node.jump_targets:
            # check dst is in subgraph
            if dst in bbmap.graph:
                preds_table[src].add(dst)
                succs_table[dst].add(src)

    return _find_dominators_internal(
        entries, list(bbmap.graph.keys()), preds_table, succs_table
    )


def _find_dominators_internal(entries, nodes, preds_table, succs_table):
    # From NUMBA
    # See theoretical description in
    # http://en.wikipedia.org/wiki/Dominator_%28graph_theory%29
    # The algorithm implemented here uses a todo-list as described
    # in http://pages.cs.wisc.edu/~fischer/cs701.f08/finding.loops.html

    # if post:
    #     entries = set(self._exit_points)
    #     preds_table = self._succs
    #     succs_table = self._preds
    # else:
    #     entries = set([self._entry_point])
    #     preds_table = self._preds
    #     succs_table = self._succs

    import functools

    if not entries:
        raise RuntimeError("no entry points: dominator algorithm " "cannot be seeded")

    doms = {}
    for e in entries:
        doms[e] = set([e])

    todo = []
    for n in nodes:
        if n not in entries:
            doms[n] = set(nodes)
            todo.append(n)

    while todo:
        n = todo.pop()
        if n in entries:
            continue
        new_doms = set([n])
        preds = preds_table[n]
        if preds:
            new_doms |= functools.reduce(set.intersection, [doms[p] for p in preds])
        if new_doms != doms[n]:
            assert len(new_doms) < len(doms[n])
            doms[n] = new_doms
            todo.extend(succs_table[n])
    return doms
