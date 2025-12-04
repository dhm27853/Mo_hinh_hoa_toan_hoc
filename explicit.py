import time
import tracemalloc
from collections import deque
from typing import Dict, List, Set, Tuple
from parser import PetriNet, PNMLParser, print_petri_net_summary


class ExplicitReachabilityAnalyzer:
    """Explicit reachability computation using BFS/DFS"""

    def __init__(self, petri_net: PetriNet):
        self.petri_net = petri_net
        self.reachable_markings: Set[Tuple[int, ...]] = set()
        self.execution_time = 0.0
        self.memory_usage = 0.0

    def marking_to_tuple(self, marking: Dict[str, int]) -> Tuple[int, ...]:
        """Convert marking dictionary to tuple for hashing"""
        # Sort by place_id to ensure consistent ordering
        sorted_places = sorted(self.petri_net.places.keys())
        return tuple(marking.get(place_id, 0) for place_id in sorted_places)

    def tuple_to_marking(self, marking_tuple: Tuple[int, ...]) -> Dict[str, int]:
        """Convert marking tuple back to dictionary"""
        sorted_places = sorted(self.petri_net.places.keys())
        return {place_id: marking_tuple[i]
                for i, place_id in enumerate(sorted_places)}

    def compute_reachable_bfs(self) -> Set[Tuple[int, ...]]:
        """
        Compute all reachable markings using Breadth-First Search (BFS)

        Returns:
            Set of reachable markings as tuples
        """
        print("\n" + "=" * 60)
        print("EXPLICIT REACHABILITY COMPUTATION (BFS)")
        print("=" * 60)

        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()

        # Get initial marking
        initial_marking = self.petri_net.get_initial_marking()
        initial_tuple = self.marking_to_tuple(initial_marking)

        # Initialize BFS
        queue = deque([initial_tuple])
        visited = {initial_tuple}
        self.reachable_markings = {initial_tuple}

        transitions_fired = 0

        print(f"Initial marking: {initial_tuple}")
        print(f"Starting BFS exploration...\n")

        # BFS main loop
        while queue:
            current_tuple = queue.popleft()
            current_marking = self.tuple_to_marking(current_tuple)

            # Try to fire each transition
            for trans_id in self.petri_net.transitions:
                if self.petri_net.is_transition_enabled(trans_id, current_marking):
                    # Fire transition and get new marking
                    new_marking = self.petri_net.fire_transition(trans_id, current_marking)
                    new_tuple = self.marking_to_tuple(new_marking)

                    transitions_fired += 1

                    # Add to reachable set if not visited
                    if new_tuple not in visited:
                        visited.add(new_tuple)
                        self.reachable_markings.add(new_tuple)
                        queue.append(new_tuple)

        # Stop timing and memory tracking
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.execution_time = end_time - start_time
        self.memory_usage = peak / 1024  # Convert to KB

        # Print results
        print(f"✓ BFS exploration completed!")
        print(f"Number of reachable markings: {len(self.reachable_markings)}")
        print(f"Transitions fired (total): {transitions_fired}")
        print(f"Execution time: {self.execution_time:.6f} seconds")
        print(f"Peak memory usage: {self.memory_usage:.2f} KB")
        print("=" * 60 + "\n")

        return self.reachable_markings

    def compute_reachable_dfs(self) -> Set[Tuple[int, ...]]:
        """
        Compute all reachable markings using Depth-First Search (DFS)

        Returns:
            Set of reachable markings as tuples
        """
        print("\n" + "=" * 60)
        print("EXPLICIT REACHABILITY COMPUTATION (DFS)")
        print("=" * 60)

        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()

        # Get initial marking
        initial_marking = self.petri_net.get_initial_marking()
        initial_tuple = self.marking_to_tuple(initial_marking)

        # Initialize DFS
        stack = [initial_tuple]
        visited = {initial_tuple}
        self.reachable_markings = {initial_tuple}

        transitions_fired = 0

        print(f"Initial marking: {initial_tuple}")
        print(f"Starting DFS exploration...\n")

        # DFS main loop
        while stack:
            current_tuple = stack.pop()
            current_marking = self.tuple_to_marking(current_tuple)

            # Try to fire each transition
            for trans_id in self.petri_net.transitions:
                if self.petri_net.is_transition_enabled(trans_id, current_marking):
                    # Fire transition and get new marking
                    new_marking = self.petri_net.fire_transition(trans_id, current_marking)
                    new_tuple = self.marking_to_tuple(new_marking)

                    transitions_fired += 1

                    # Add to reachable set if not visited
                    if new_tuple not in visited:
                        visited.add(new_tuple)
                        self.reachable_markings.add(new_tuple)
                        stack.append(new_tuple)

        # Stop timing and memory tracking
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.execution_time = end_time - start_time
        self.memory_usage = peak / 1024  # Convert to KB

        # Print results
        print(f"✓ DFS exploration completed!")
        print(f"Number of reachable markings: {len(self.reachable_markings)}")
        print(f"Transitions fired (total): {transitions_fired}")
        print(f"Execution time: {self.execution_time:.6f} seconds")
        print(f"Peak memory usage: {self.memory_usage:.2f} KB")
        print("=" * 60 + "\n")

        return self.reachable_markings

    def print_reachable_markings(self, limit: int = 20):
        """Print reachable markings (limited to first 'limit' markings)"""
        if not self.reachable_markings:
            print("No reachable markings computed yet.")
            return

        print("\n" + "=" * 60)
        print(f"REACHABLE MARKINGS (showing first {min(limit, len(self.reachable_markings))})")
        print("=" * 60)

        sorted_places = sorted(self.petri_net.places.keys())

        # Print header
        header = "   " + "  ".join(f"{pid:>6}" for pid in sorted_places)
        print(header)
        print("-" * len(header))

        # Print markings
        for idx, marking_tuple in enumerate(sorted(self.reachable_markings)):
            if idx >= limit:
                print(f"... and {len(self.reachable_markings) - limit} more markings")
                break

            row = f"{idx + 1:2d} " + "  ".join(f"{token:>6}" for token in marking_tuple)
            print(row)

        print("=" * 60 + "\n")

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the reachability analysis"""
        return {
            'num_reachable_markings': len(self.reachable_markings),
            'execution_time': self.execution_time,
            'memory_usage_kb': self.memory_usage,
            'num_places': len(self.petri_net.places),
            'num_transitions': len(self.petri_net.transitions)
        }

    def verify_1_safe_property(self) -> bool:
        """Verify that all reachable markings satisfy 1-safe property"""
        print("\n" + "=" * 60)
        print("VERIFYING 1-SAFE PROPERTY")
        print("=" * 60)

        violations = []
        for marking_tuple in self.reachable_markings:
            for i, tokens in enumerate(marking_tuple):
                if tokens > 1:
                    violations.append((marking_tuple, i, tokens))

        if violations:
            print(f"  WARNING: Found {len(violations)} 1-safe violations!")
            for marking, place_idx, tokens in violations[:5]:  # Show first 5
                sorted_places = sorted(self.petri_net.places.keys())
                place_id = sorted_places[place_idx]
                print(f"  Place '{place_id}' has {tokens} tokens in marking {marking}")
            if len(violations) > 5:
                print(f"  ... and {len(violations) - 5} more violations")
            print("=" * 60 + "\n")
            return False
        else:
            print("✓ All reachable markings satisfy 1-safe property")
            print("  (each place has at most 1 token)")
            print("=" * 60 + "\n")
            return True


def compare_bfs_dfs(petri_net: PetriNet):
    """Compare BFS and DFS approaches"""
    print("\n" + "=" * 60)
    print("COMPARING BFS vs DFS")
    print("=" * 60)

    # Run BFS
    analyzer_bfs = ExplicitReachabilityAnalyzer(petri_net)
    reachable_bfs = analyzer_bfs.compute_reachable_bfs()
    stats_bfs = analyzer_bfs.get_statistics()

    # Run DFS
    analyzer_dfs = ExplicitReachabilityAnalyzer(petri_net)
    reachable_dfs = analyzer_dfs.compute_reachable_dfs()
    stats_dfs = analyzer_dfs.get_statistics()

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'BFS':>12} {'DFS':>12}")
    print("-" * 60)
    print(
        f"{'Reachable markings':<30} {stats_bfs['num_reachable_markings']:>12} {stats_dfs['num_reachable_markings']:>12}")
    print(f"{'Execution time (s)':<30} {stats_bfs['execution_time']:>12.6f} {stats_dfs['execution_time']:>12.6f}")
    print(f"{'Memory usage (KB)':<30} {stats_bfs['memory_usage_kb']:>12.2f} {stats_dfs['memory_usage_kb']:>12.2f}")

    # Verify they found the same markings
    if reachable_bfs == reachable_dfs:
        print("\n✓ BFS and DFS found identical reachable sets")
    else:
        print("\n  WARNING: BFS and DFS found different reachable sets!")
        print(f"  BFS only: {len(reachable_bfs - reachable_dfs)} markings")
        print(f"  DFS only: {len(reachable_dfs - reachable_bfs)} markings")

    print("=" * 60 + "\n")


def main():
    """Main function for Task 2"""
    import sys
    import glob

    if len(sys.argv) > 1:
        pnml_files = [sys.argv[1]]
    else:
        pnml_files = glob.glob("*.pnml")

        if not pnml_files:
            print("No .pnml files found in current directory")
            print("\nUsage:")
            print("  python explicit.py [filename.pnml]")
            print("  python explicit.py              # Process all .pnml files")
            print("\nExamples:")
            print("  python explicit.py exam.pnml    # Process only exam.pnml")
            print("  python explicit.py              # Process all .pnml files")
            sys.exit(1)

    print("=" * 60)
    print("TASK 2: EXPLICIT REACHABILITY COMPUTATION")
    print("=" * 60)
    print(f"\nProcessing {len(pnml_files)} PNML file(s):")
    for i, filename in enumerate(pnml_files, 1):
        print(f"  {i}. {filename}")
    print()

    # Parse each file
    parser = PNMLParser()

    for filename in pnml_files:
        print(f"\n{'#' * 60}")
        print(f"# Processing: {filename}")
        print(f"{'#' * 60}")

        try:
            # Task 1: Parse PNML
            petri_net = parser.parse(filename)
            print_petri_net_summary(petri_net)

            # Task 2: Compute reachability (BFS)
            analyzer = ExplicitReachabilityAnalyzer(petri_net)
            reachable = analyzer.compute_reachable_bfs()

            # Print results
            analyzer.print_reachable_markings(limit=20)

            # Verify 1-safe property
            analyzer.verify_1_safe_property()

            # Optional: Compare BFS vs DFS
            # Uncomment the line below to compare both approaches
            # compare_bfs_dfs(petri_net)

        except FileNotFoundError:
            print(f"\n Error: File '{filename}' not found!")
            print(f"   Please make sure the file exists in the current directory.")
        except Exception as e:
            print(f"\n Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ALL FILES PROCESSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
