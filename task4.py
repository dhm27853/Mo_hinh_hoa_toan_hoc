#!/usr/bin/env python3
import sys, time
import xml.etree.ElementTree as ET
from collections import deque
from dd import autoref as _bdd
import pulp

# -------------------------
# Data structures for Petri net
# -------------------------
class PetriNet:
    def __init__(self):
        self.places = []           # list of place ids (strings)
        self.transitions = []      # list of transition ids (strings)
        self.input_arcs = {}       # transition_id -> set(place_id)
        self.output_arcs = {}      # transition_id -> set(place_id)
        self.initial_marking = {}  # place_id -> 0/1
        self.place_index = {}      # place_id -> index

    def finalize(self):
        # ensure deterministic ordering of places
        self.place_index = {p: i for i, p in enumerate(self.places)}
        for t in self.transitions:
            self.input_arcs.setdefault(t, set())
            self.output_arcs.setdefault(t, set())
        # ensure initial for all places
        for p in self.places:
            self.initial_marking.setdefault(p, 0)

# -------------------------
# PNML parser (robust for common PNML flavors)
# -------------------------
def parse_pnml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    # collect all places, transitions, arcs (regardless of xml namespace)
    places = []
    transitions = []
    arcs = []

    # Helper to check local tag name without ns
    def local_name(tag):
        return tag.split('}')[-1] if '}' in tag else tag

    for elem in root.iter():
        ln = local_name(elem.tag)
        if ln == 'place':
            pid = elem.attrib.get('id')
            places.append((pid, elem))
        elif ln == 'transition':
            tid = elem.attrib.get('id')
            transitions.append((tid, elem))
        elif ln == 'arc':
            src = elem.attrib.get('source')
            tgt = elem.attrib.get('target')
            arcs.append((src, tgt, elem))

    net = PetriNet()
    # places + initial markings
    for pid, p_elem in places:
        net.places.append(pid)
        # Try common patterns for initial marking inside the place element
        # look for <initialMarking><text>n</text></initialMarking> or <marking><text>n</text></marking>
        init = 0
        for child in p_elem:
            ctag = local_name(child.tag).lower()
            if 'initialmark' in ctag or 'mark' == ctag or 'marking' in ctag:
                # try to find text node under it
                txt = None
                for g in child.iter():
                    if local_name(g.tag).lower() == 'text' and g.text:
                        txt = g.text.strip()
                        break
                if txt and txt.isdigit():
                    init = int(txt)
                    break
        net.initial_marking[pid] = 1 if init > 0 else 0

    # transitions
    for tid, _ in transitions:
        net.transitions.append(tid)

    # arcs mapping
    place_ids = set(net.places)
    trans_ids = set(net.transitions)
    for src, tgt, _ in arcs:
        if src in place_ids and tgt in trans_ids:
            # place -> transition = input arc
            net.input_arcs.setdefault(tgt, set()).add(src)
        elif src in trans_ids and tgt in place_ids:
            # transition -> place = output arc
            net.output_arcs.setdefault(src, set()).add(tgt)
        else:
            # ignore arcs that don't match (some PNML variants use nested refs)
            pass

    net.finalize()
    return net

# -------------------------
# Reachability (explicit BFS)
# -------------------------
def marking_to_tuple(mark, places):
    return tuple(mark.get(p, 0) for p in places)

def tuple_to_marking(tup, places):
    return {places[i]: tup[i] for i in range(len(places))}

def enabled_transitions(marking, net: PetriNet):
    enabled = []
    for t in net.transitions:
        inputs = net.input_arcs.get(t, set())
        if len(inputs) == 0:
            # convention: transitions with no input are always enabled
            enabled.append(t)
            continue
        ok = True
        for p in inputs:
            if marking.get(p, 0) == 0:
                ok = False
                break
        if ok:
            enabled.append(t)
    return enabled

def fire_transition(marking, t, net: PetriNet):
    new = dict(marking)
    for p in net.input_arcs.get(t, set()):
        new[p] = 0
    for p in net.output_arcs.get(t, set()):
        new[p] = 1
    return new

def explicit_bfs(net: PetriNet, max_states=200_000):
    start = net.initial_marking.copy()
    start_t = marking_to_tuple(start, net.places)
    q = deque([start_t])
    seen = {start_t}
    markings = [start_t]
    while q:
        if len(seen) > max_states:
            print("Warning: exceeded max_states ({}) — stopping BFS".format(max_states))
            break
        cur_t = q.popleft()
        cur = tuple_to_marking(cur_t, net.places)
        for t in enabled_transitions(cur, net):
            nxt = fire_transition(cur, t, net)
            nxt_t = marking_to_tuple(nxt, net.places)
            if nxt_t not in seen:
                seen.add(nxt_t)
                q.append(nxt_t)
                markings.append(nxt_t)
    return markings

# -------------------------
# BDD: build from explicit markings
# -------------------------
def build_bdd_from_markings(marking_tuples, places):
    bdd = _bdd.BDD()
    var_names = [f"p_{i}" for i in range(len(places))]
    for v in var_names:
        bdd.add_var(v)

    def cube_from_tuple(tup):
        node = bdd.true
        for i,bit in enumerate(tup):
            v = bdd.var(var_names[i])
            lit = v if bit==1 else ~v
            node = node & lit
        return node

    reach = bdd.false
    for t in marking_tuples:
        reach = reach | cube_from_tuple(t)
    return bdd, var_names, reach

def bdd_contains(bdd, var_names, reach_bdd, marking_tuple):
    node = bdd.true
    for i,bit in enumerate(marking_tuple):
        v = bdd.var(var_names[i])
        lit = v if bit==1 else ~v
        node = node & lit
    return (reach_bdd & node) != bdd.false

# -------------------------
# ILP: find dead marking (not necessarily reachable)
# -------------------------
def solve_deadmark_ilp(net: PetriNet, bdd=None, var_names=None, reach_bdd=None, time_limit_seconds=30):
    """
    Return (marking_tuple, elapsed_seconds, iterations) if found reachable deadlock
           (None, elapsed_seconds, iterations) if none (no reachable deadlock)
    """
    n = len(net.places)
    # Create ILP model (feasibility)
    prob = pulp.LpProblem("dead_marking", pulp.LpStatusOptimal)
    # binary variables M_i for each place index
    M = [pulp.LpVariable(f"M_{i}", lowBound=0, upBound=1, cat='Integer') for i in range(n)]

    # dead constraints: for each transition t, sum(inputs) <= |inputs|-1
    for t in net.transitions:
        inputs = list(net.input_arcs.get(t, []))
        if len(inputs) == 0:
            # transition with no inputs is always enabled -> makes dead marking impossible
            # Add an impossible constraint to make model infeasible
            prob += 0 <= -1, f"trans_{t}_no_input_prevents_dead"
        else:
            prob += pulp.lpSum([ M[net.place_index[p]] for p in inputs ]) <= len(inputs) - 1, f"dead_{t}"

    # dummy objective (we only need feasibility)
    prob += 0, "obj"

    excluded = 0
    start_time = time.time()
    iteration = 0
    while True:
        iteration += 1
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
        res = prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if status.lower() not in ('optimal', 'feasible'):
            elapsed = time.time() - start_time
            return None, elapsed, iteration-1
        # retrieve solution marking tuple
        marking_tuple = tuple(int(round(pulp.value(M[i]))) for i in range(n))
        # If BDD provided: check reachability
        if bdd is not None and reach_bdd is not None and var_names is not None:
            if bdd_contains(bdd, var_names, reach_bdd, marking_tuple):
                elapsed = time.time() - start_time
                return marking_tuple, elapsed, iteration
            # else: exclude this marking and continue
        else:
            # No BDD: return any dead marking found (but Task4 requires reachable check)
            elapsed = time.time() - start_time
            return marking_tuple, elapsed, iteration

        # Exclude this exact marking and continue
        excluded += 1
        expr = []
        for i, bit in enumerate(marking_tuple):
            if bit == 1:
                expr.append(M[i])
            else:
                expr.append(1 - M[i])
        prob += pulp.lpSum(expr) <= n - 1, f"exclude_{excluded}"
        # loop continues

# -------------------------
# CLI main
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python task4_deadlock_ilp_bdd.py <model.pnml>")
        sys.exit(1)
    pnml = sys.argv[1]
    print("Parsing PNML:", pnml)
    net = parse_pnml(pnml)
    print(f"Found {len(net.places)} places and {len(net.transitions)} transitions.")
    print("Initial marking (places with 1):", [p for p in net.places if net.initial_marking.get(p,0)==1])

    print("\n[*] Computing explicit reachable markings (BFS)...")
    t0 = time.time()
    markings = explicit_bfs(net)
    t1 = time.time()
    print(f" - Reachable markings: {len(markings)} (time {t1-t0:.3f}s)")

    print("[*] Building BDD from reachable markings...")
    bdd, var_names, reach_bdd = build_bdd_from_markings(markings, net.places)
    print("[*] Running ILP to find reachable deadlock...")
    dm, elapsed, iterations = solve_deadmark_ilp(net, bdd=bdd, var_names=var_names, reach_bdd=reach_bdd)
    if dm is None:
        print(f"No reachable deadlock found. Iterations: {iterations}. Time: {elapsed:.3f}s")
    else:
        print(f"Reachable deadlock found (iterations: {iterations}, time: {elapsed:.3f}s):")
        for i,p in enumerate(net.places):
            print(f"  {p} = {dm[i]}")
    print("Done.")

if __name__ == "__main__":
    main()
