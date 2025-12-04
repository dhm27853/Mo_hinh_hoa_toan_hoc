#!/usr/bin/env python3
import sys
import time
import argparse
from dd.autoref import BDD
import pulp
from parser import PNMLParser
from explicit import ExplicitReachabilityAnalyzer

try:
    from task3 import symbolic_reachability_bdd
    HAS_TASK3 = True
except Exception:
    HAS_TASK3 = False

def build_bdd_from_markings(marking_tuples, place_order):
    bdd = BDD()
    var_names = [f"p_{i}" for i in range(len(place_order))]
    bdd.declare(*var_names)
    def cube(tup):
        node = bdd.true
        for i,bit in enumerate(tup):
            v = bdd.var(var_names[i])
            lit = v if bit else ~v
            node = node & lit
        return node
    reach = bdd.false
    for t in marking_tuples:
        reach = reach | cube(t)
    return bdd, var_names, reach

def bdd_contains_marking(bdd, var_names, reach_bdd, marking_tuple):
    node = bdd.true
    for i,bit in enumerate(marking_tuple):
        v = bdd.var(var_names[i])
        lit = v if bit else ~v
        node = node & lit
    return (reach_bdd & node) != bdd.false

def solve_deadlock_ilp(pn, place_order, bdd=None, var_names=None, reach_bdd=None, time_limit=30):
    n = len(place_order)
    idx = {p:i for i,p in enumerate(place_order)}
    prob = pulp.LpProblem("dead_marking", pulp.LpStatusOptimal)
    M = [pulp.LpVariable(f"M_{i}", lowBound=0, upBound=1, cat='Integer') for i in range(n)]
    for t in pn.transitions:
        inputs = [p for p,w in pn.input_places.get(t, [])]
        if len(inputs) == 0:
            prob += 0 <= -1
        else:
            prob += pulp.lpSum([ M[idx[p]] for p in inputs ]) <= len(inputs) - 1
    prob += 0
    excluded = 0
    start = time.time()
    it = 0
    while True:
        it += 1
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        status = prob.solve(solver)
        status_str = pulp.LpStatus[prob.status].lower()
        if status_str not in ('optimal','feasible'):
            return None, time.time()-start, it-1
        marking = tuple(int(round(pulp.value(M[i]))) for i in range(n))
        if bdd is not None:
            if bdd_contains_marking(bdd, var_names, reach_bdd, marking):
                return marking, time.time()-start, it
        else:
            return marking, time.time()-start, it
        excluded += 1
        expr = []
        for i, bit in enumerate(marking):
            expr.append(M[i] if bit else (1 - M[i]))
        prob += pulp.lpSum(expr) <= n - 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pnml")
    ap.add_argument("--no-symbolic", action="store_true")
    ap.add_argument("--time-limit", type=int, default=30)
    args = ap.parse_args()

    parser = PNMLParser()
    pn = parser.parse(args.pnml)
    place_order = sorted(list(pn.places.keys()))

    bdd = var_names = reach_bdd = None
    used_symbolic = False

    if (not args.no_symbolic) and HAS_TASK3:
        try:
            count, duration, steps = symbolic_reachability_bdd(pn)
            used_symbolic = True
        except Exception:
            used_symbolic = False

    if not used_symbolic:
        analyzer = ExplicitReachabilityAnalyzer(pn)
        reachable = analyzer.compute_reachable_bfs()
        reachable_list = sorted(reachable)
        bdd, var_names, reach_bdd = build_bdd_from_markings(reachable_list, place_order)

    dm, elapsed, it = solve_deadlock_ilp(
        pn, place_order, bdd=bdd, var_names=var_names,
        reach_bdd=reach_bdd, time_limit=args.time_limit
    )

    if dm is None:
        print(f"No reachable deadlock found. Iterations: {it}. Time: {elapsed:.3f}s")
    else:
        print(f"Reachable deadlock found (iterations: {it}, time: {elapsed:.3f}s):")
        for i,p in enumerate(place_order):
            print(f"{p} = {dm[i]}")

if __name__ == "__main__":
    main()
