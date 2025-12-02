#!/usr/bin/env python3
"""
Integration Test for Task 5: Optimization over Reachable Markings
==================================================================

This script demonstrates how to:
1. Parse a Petri net (using task4.py)
2. Compute reachable markings
3. Build a BDD from reachable markings
4. Run optimization to find the maximum-weight marking
5. Verify the results
"""

import sys
import time
from task4 import parse_pnml, explicit_bfs, build_bdd_from_markings
from task5 import (
    optimize_reachable_markings,
    generate_random_weights,
    generate_custom_weights,
    verify_marking_in_reachable,
    compute_score
)


def main():
    """Main test function."""
    
    if len(sys.argv) < 2:
        print("Usage: python test_task5.py <model.pnml> [seed]")
        print("\nExample:")
        print("  python test_task5.py pnml_reachability.pnml 42")
        sys.exit(1)
    
    pnml_file = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    
    print("\n" + "="*70)
    print("TASK 5 INTEGRATION TEST")
    print("="*70)
    
    try:
        # ====================================================================
        # STEP 1: Parse Petri Net
        # ====================================================================
        print("\n[STEP 1] Parsing Petri Net...")
        print("-"*70)
        
        net = parse_pnml(pnml_file)
        print(f"✓ Parsed successfully")
        print(f"  - Places: {len(net.places)}")
        print(f"  - Transitions: {len(net.transitions)}")
        print(f"  - Initial marking: {[p for p in net.places if net.initial_marking.get(p, 0) == 1]}")
        
        # ====================================================================
        # STEP 2: Compute Reachable Markings (Explicit BFS)
        # ====================================================================
        print("\n[STEP 2] Computing Reachable Markings (BFS)...")
        print("-"*70)
        
        t0 = time.time()
        markings = explicit_bfs(net, max_states=200_000)
        t1 = time.time()
        
        print(f"✓ Computed {len(markings)} reachable markings in {t1-t0:.3f}s")
        
        # ====================================================================
        # STEP 3: Build BDD from Reachable Markings
        # ====================================================================
        print("\n[STEP 3] Building BDD from Reachable Markings...")
        print("-"*70)
        
        t0 = time.time()
        bdd, var_names, reach_bdd = build_bdd_from_markings(markings, net.places)
        t1 = time.time()
        
        print(f"✓ Built BDD in {t1-t0:.3f}s")
        print(f"  - Variables: {len(var_names)}")
        print(f"  - Variable names: {var_names}")
        
        # ====================================================================
        # STEP 4: Generate Weights
        # ====================================================================
        print("\n[STEP 4] Generating Weights...")
        print("-"*70)
        
        weights = generate_random_weights(len(net.places), seed=seed)
        print(f"✓ Generated random weights (seed={seed})")
        for i, (place, weight) in enumerate(zip(net.places, weights.values())):
            print(f"  - {place}: {weight}")
        
        # ====================================================================
        # STEP 5: Run Optimization
        # ====================================================================
        print("\n[STEP 5] Running Optimization...")
        print("-"*70)
        
        results = optimize_reachable_markings(
            petri_net=net,
            bdd=bdd,
            var_names=var_names,
            reachable_bdd=reach_bdd,
            weights=weights,
            verbose=True
        )
        
        # ====================================================================
        # STEP 6: Verify Results
        # ====================================================================
        print("\n[STEP 6] Verifying Results...")
        print("-"*70)
        
        optimal_marking = results['optimal_marking']
        max_score = results['max_score']
        
        # Check if marking is reachable
        is_reachable = verify_marking_in_reachable(
            bdd, var_names, reach_bdd, optimal_marking
        )
        print(f"✓ Optimal marking is reachable: {is_reachable}")
        
        # Verify score computation
        computed_score = compute_score(optimal_marking, weights, var_names)
        print(f"✓ Score verification: {computed_score} == {max_score}: {abs(computed_score - max_score) < 1e-6}")
        
        # ====================================================================
        # STEP 7: Summary
        # ====================================================================
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Reachable markings: {len(markings)}")
        print(f"Maximum score: {max_score}")
        print(f"Optimal marking: {optimal_marking}")
        print(f"Optimization time: {results['execution_time']:.6f}s")
        print(f"BDD nodes memoized: {results['num_memoized_nodes']}")
        print("="*70)
        
        return 0
        
    except FileNotFoundError:
        print(f"❌ Error: File '{pnml_file}' not found")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

