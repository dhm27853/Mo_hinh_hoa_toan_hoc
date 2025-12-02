#!/usr/bin/env python3
"""
Task 5: Optimization over Reachable Markings using BDD
========================================================

This module implements an optimization algorithm to find the marking in a set of
reachable markings that maximizes an objective function:

    Score = Σ (weight_i × token_i)

where token_i is 1 if place i has a token, 0 otherwise.

Key Features:
- DFS traversal on BDD nodes with memoization (no enumeration)
- Recursive algorithm that follows high/low branches of BDD
- Backtracking to reconstruct the optimal marking
- Execution time measurement
"""

import sys
import time
import random
from typing import Dict, Tuple, List, Optional
from dd.autoref import BDD

# ============================================================================
# PART 1: Weight Generation
# ============================================================================

def generate_random_weights(num_places: int, seed: Optional[int] = None, 
                           min_weight: int = 1, max_weight: int = 100) -> Dict[int, int]:
    """
    Generate random integer weights for each place.
    
    Args:
        num_places: Number of places in the Petri net
        seed: Random seed for reproducibility (None = no seed)
        min_weight: Minimum weight value
        max_weight: Maximum weight value
    
    Returns:
        Dictionary mapping place index -> weight
    """
    if seed is not None:
        random.seed(seed)
    
    weights = {}
    for i in range(num_places):
        weights[i] = random.randint(min_weight, max_weight)
    
    return weights


def generate_custom_weights(num_places: int, weights_list: List[int]) -> Dict[int, int]:
    """
    Create weights from a custom list.
    
    Args:
        num_places: Number of places
        weights_list: List of weights (must match num_places)
    
    Returns:
        Dictionary mapping place index -> weight
    """
    if len(weights_list) != num_places:
        raise ValueError(f"Expected {num_places} weights, got {len(weights_list)}")
    
    return {i: w for i, w in enumerate(weights_list)}


# ============================================================================
# PART 2: BDD Optimization with DFS + Memoization
# ============================================================================

class BDDOptimizer:
    """
    Optimizer for finding the maximum-weight marking in a reachable set.
    
    Uses recursive DFS traversal on BDD nodes with memoization to avoid
    enumerating all markings.
    """
    
    def __init__(self, bdd: BDD, var_names: List[str], reachable_bdd, 
                 weights: Dict[int, int]):
        """
        Initialize the optimizer.
        
        Args:
            bdd: BDD manager object
            var_names: List of variable names (e.g., ["p_0", "p_1", ...])
            reachable_bdd: BDD node representing reachable markings
            weights: Dictionary mapping place index -> weight
        """
        self.bdd = bdd
        self.var_names = var_names
        self.reachable_bdd = reachable_bdd
        self.weights = weights
        self.num_places = len(var_names)
        
        # Memoization cache: node_id -> (max_score, path_to_max)
        self.memo = {}
        
        # Track the path that achieves maximum score
        self.best_path = None
        self.best_score = float('-inf')
    
    def _dfs_max_weight(self, u, depth: int = 0, swap_children: bool = False) -> Tuple[float, Optional[List[int]]]:
        """
        Recursive DFS to find maximum weight path in BDD.

        A BDD node represents a decision for a place:
        - High branch (x_i = 1): Add weight of place i
        - Low branch (x_i = 0): Add 0

        Args:
            u: Current BDD node reference (integer in dd library)
            depth: Current depth in recursion (for debugging)

        Returns:
            Tuple of (max_score, path_to_max) where:
            - max_score: Maximum achievable score from this node
            - path_to_max: List of place indices set to 1 in optimal path
        """
        # Base cases (dd library: True=1, False=0 or -1)
        if u == self.bdd.false:
            # Dead end: no valid marking from here
            return float('-inf'), None

        if u == self.bdd.true:
            # Leaf node: valid marking found (all remaining places = 0)
            return 0.0, []

        # Check memoization (u is integer, use directly as key)
        if u in self.memo:
            return self.memo[u]

        # Get variable index at this node
        # In dd library: bdd.var(u) returns the variable level/index
        # In dd, get variable level; then get var name and determine cofactors by substitution
        try:
            succ_info = self.bdd.succ(u)
        except Exception:
            return float('-inf'), None
        if not (isinstance(succ_info, tuple) and len(succ_info) >= 1):
            return float('-inf'), None
        var_level = succ_info[0]

        # Map level -> var name -> index
        if var_level is None or var_level < 0 or var_level >= self.num_places:
            return float('-inf'), None
        try:
            var_name = self.bdd.var_at_level(var_level)
        except Exception:
            return float('-inf'), None
        if var_name not in self.var_names:
            return float('-inf'), None
        idx = self.var_names.index(var_name)
        place_weight = self.weights.get(idx, 0)

        # Determine cofactors by substitution to avoid ambiguity of succ child order
        try:
            hi_node = self.bdd.let({var_name: self.bdd.true}, u)
            lo_node = self.bdd.let({var_name: self.bdd.false}, u)
        except Exception:
            return float('-inf'), None

        # Recursively evaluate both branches
        high_score, high_path = self._dfs_max_weight(hi_node, depth + 1)
        if high_path is not None:
            high_score += place_weight
            high_path = [idx] + high_path

        low_score, low_path = self._dfs_max_weight(lo_node, depth + 1)

        # Choose the branch with higher score
        # Handle cases where one branch is invalid (-inf)
        if high_path is None and low_path is None:
            result = (float('-inf'), None)
        elif high_path is None:
            result = (low_score, low_path)
        elif low_path is None:
            result = (high_score, high_path)
        else:
            # Both branches valid, compare scores
            if high_score >= low_score:
                result = (high_score, high_path)
            else:
                result = (low_score, low_path)

        # Memoize (u is integer, use directly as key)
        self.memo[u] = result
        return result
    
    def optimize(self) -> Tuple[float, List[str]]:
        """
        Find the marking that maximizes the objective function.
        
        Returns:
            Tuple of (max_score, optimal_marking) where:
            - max_score: Maximum objective value
            - optimal_marking: List of place names with tokens
        """
        max_score, path = self._dfs_max_weight(self.reachable_bdd)
        
        if path is None:
            # No valid marking found
            return float('-inf'), []
        
        # Convert path (list of indices) to place names
        optimal_marking = [self.var_names[idx] for idx in sorted(path)]
        
        return max_score, optimal_marking


# ============================================================================
# PART 3: Main Optimization Function
# ============================================================================

def optimize_reachable_markings(petri_net, bdd, var_names: List[str], 
                               reachable_bdd, weights: Optional[Dict[int, int]] = None,
                               verbose: bool = True) -> Dict:
    """
    Find the marking in reachable_bdd that maximizes the weighted objective.
    
    This is the main entry point for Task 5.
    
    Args:
        petri_net: PetriNet object (for reference)
        bdd: BDD manager object
        var_names: List of variable names (e.g., ["p_0", "p_1", ...])
        reachable_bdd: BDD node representing reachable markings
        weights: Dictionary mapping place index -> weight
                If None, generates random weights
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with keys:
        - 'max_score': Maximum objective value found
        - 'optimal_marking': List of place names with tokens
        - 'execution_time': Time taken for optimization (seconds)
        - 'weights': Dictionary of weights used
        - 'num_places': Number of places
    """
    
    # Generate weights if not provided
    if weights is None:
        num_places = len(var_names)
        weights = generate_random_weights(num_places, seed=42)
        if verbose:
            print("Generated random weights (seed=42)")
    
    if verbose:
        print("\n" + "="*70)
        print("TASK 5: OPTIMIZATION OVER REACHABLE MARKINGS")
        print("="*70)
        print(f"Number of places: {len(var_names)}")
        print(f"Weights: {weights}")
        print("-"*70)
    
    # Start optimization
    start_time = time.time()
    
    # Create optimizer
    optimizer = BDDOptimizer(bdd, var_names, reachable_bdd, weights)
    
    # Run optimization
    max_score, optimal_marking = optimizer.optimize()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Format results
    results = {
        'max_score': max_score,
        'optimal_marking': optimal_marking,
        'execution_time': execution_time,
        'weights': weights,
        'num_places': len(var_names),
        'num_memoized_nodes': len(optimizer.memo)
    }
    
    # Print results
    if verbose:
        print("\nRESULTS:")
        print("-"*70)
        print(f"Maximum Objective Value: {max_score}")
        print(f"Optimal Marking (places with tokens): {optimal_marking}")
        print(f"Execution Time: {execution_time:.6f} seconds")
        print(f"BDD Nodes Memoized: {len(optimizer.memo)}")
        print("="*70)
    
    return results


# ============================================================================
# PART 4: Utility Functions for Integration
# ============================================================================

def verify_marking_in_reachable(bdd, var_names: List[str], reachable_bdd, 
                               marking: List[str]) -> bool:
    """
    Verify that a marking is in the reachable set.
    
    Args:
        bdd: BDD manager
        var_names: List of variable names
        reachable_bdd: BDD node for reachable markings
        marking: List of place names with tokens
    
    Returns:
        True if marking is reachable, False otherwise
    """
    # Create a cube for the marking
    marking_set = set(marking)
    node = bdd.true
    
    for i, var_name in enumerate(var_names):
        v = bdd.var(var_name)
        if var_name in marking_set:
            node = node & v
        else:
            node = node & ~v
    
    # Check if this marking is in reachable_bdd
    return (reachable_bdd & node) != bdd.false


def compute_score(marking: List[str], weights: Dict[int, int], 
                 var_names: List[str]) -> float:
    """
    Compute the objective score for a given marking.
    
    Args:
        marking: List of place names with tokens
        weights: Dictionary mapping place index -> weight
        var_names: List of variable names
    
    Returns:
        Total score
    """
    score = 0.0
    marking_set = set(marking)
    
    for i, var_name in enumerate(var_names):
        if var_name in marking_set:
            score += weights.get(i, 0)
    
    return score


# ============================================================================
# PART 5: Example / Testing
# ============================================================================

def example_usage():
    """
    Example of how to use the optimizer.
    
    This assumes you have already:
    1. Parsed a Petri net
    2. Computed reachable markings (explicit or BDD)
    3. Built a BDD from those markings
    """
    print("Example: Optimization over Reachable Markings")
    print("=" * 70)
    print("""
    Usage:
    ------
    
    # Step 1: Parse Petri net and compute reachable markings
    from task4 import parse_pnml, explicit_bfs, build_bdd_from_markings
    
    net = parse_pnml("model.pnml")
    markings = explicit_bfs(net)
    bdd, var_names, reach_bdd = build_bdd_from_markings(markings, net.places)
    
    # Step 2: Define weights
    weights = {0: 10, 1: 20, 2: 15}  # or use generate_random_weights()
    
    # Step 3: Run optimization
    results = optimize_reachable_markings(
        petri_net=net,
        bdd=bdd,
        var_names=var_names,
        reachable_bdd=reach_bdd,
        weights=weights,
        verbose=True
    )
    
    # Step 4: Access results
    print(f"Max Score: {results['max_score']}")
    print(f"Optimal Marking: {results['optimal_marking']}")
    print(f"Time: {results['execution_time']:.6f}s")
    """)


if __name__ == "__main__":
    # If run directly, show example usage
    example_usage()

