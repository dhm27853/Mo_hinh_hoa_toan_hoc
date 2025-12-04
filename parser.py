import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class Place:
    """Represents a place in the Petri net"""
    id: str
    name: str = ""
    initial_tokens: int = 0

    def __repr__(self):
        return f"Place({self.id}, tokens={self.initial_tokens})"


@dataclass
class Transition:
    """Represents a transition in the Petri net"""
    id: str
    name: str = ""

    def __repr__(self):
        return f"Transition({self.id})"


@dataclass
class Arc:
    """Represents an arc in the Petri net"""
    id: str
    source: str
    target: str
    weight: int = 1

    def __repr__(self):
        return f"Arc({self.source} -> {self.target}, weight={self.weight})"


@dataclass
class PetriNet:
    """Main Petri net structure"""
    places: Dict[str, Place] = field(default_factory=dict)
    transitions: Dict[str, Transition] = field(default_factory=dict)
    arcs: List[Arc] = field(default_factory=list)

    # Flow relations: transition -> input/output places
    input_places: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)  # transition -> [(place, weight)]
    output_places: Dict[str, List[Tuple[str, int]]] = field(default_factory=dict)  # transition -> [(place, weight)]

    def add_place(self, place: Place):
        """Add a place to the Petri net"""
        self.places[place.id] = place

    def add_transition(self, transition: Transition):
        """Add a transition to the Petri net"""
        self.transitions[transition.id] = transition
        self.input_places[transition.id] = []
        self.output_places[transition.id] = []

    def add_arc(self, arc: Arc):
        """Add an arc and update flow relations"""
        self.arcs.append(arc)

        # Determine if arc is from place to transition or vice versa
        if arc.source in self.places and arc.target in self.transitions:
            # Place -> Transition (input place)
            self.input_places[arc.target].append((arc.source, arc.weight))
        elif arc.source in self.transitions and arc.target in self.places:
            # Transition -> Place (output place)
            self.output_places[arc.source].append((arc.target, arc.weight))
        else:
            raise ValueError(f"Invalid arc: {arc.source} -> {arc.target}")

    def get_initial_marking(self) -> Dict[str, int]:
        """Get the initial marking (tokens in each place)"""
        return {place_id: place.initial_tokens
                for place_id, place in self.places.items()}

    def is_transition_enabled(self, transition_id: str, marking: Dict[str, int]) -> bool:
        """Check if a transition is enabled in the given marking"""
        if transition_id not in self.transitions:
            return False

        # A transition is enabled if all input places have enough tokens
        for place_id, weight in self.input_places[transition_id]:
            if marking.get(place_id, 0) < weight:
                return False
        return True

    def fire_transition(self, transition_id: str, marking: Dict[str, int]) -> Dict[str, int]:
        """Fire a transition and return the new marking"""
        if not self.is_transition_enabled(transition_id, marking):
            raise ValueError(f"Transition {transition_id} is not enabled")

        new_marking = marking.copy()

        # Remove tokens from input places
        for place_id, weight in self.input_places[transition_id]:
            new_marking[place_id] -= weight

        # Add tokens to output places
        for place_id, weight in self.output_places[transition_id]:
            new_marking[place_id] = new_marking.get(place_id, 0) + weight

        return new_marking

    def verify_consistency(self) -> List[str]:
        """Verify the consistency of the Petri net"""
        errors = []

        # Check 1: All arcs reference valid places/transitions
        for arc in self.arcs:
            source_exists = arc.source in self.places or arc.source in self.transitions
            target_exists = arc.target in self.places or arc.target in self.transitions

            if not source_exists:
                errors.append(f"Arc {arc.id}: source node '{arc.source}' not found")
            if not target_exists:
                errors.append(f"Arc {arc.id}: target node '{arc.target}' not found")

        # Check 2: All arcs connect place-transition or transition-place (not place-place or transition-transition)
        for arc in self.arcs:
            source_is_place = arc.source in self.places
            source_is_transition = arc.source in self.transitions
            target_is_place = arc.target in self.places
            target_is_transition = arc.target in self.transitions

            valid_connection = (source_is_place and target_is_transition) or \
                               (source_is_transition and target_is_place)

            if not valid_connection:
                if source_is_place and target_is_place:
                    errors.append(
                        f"Arc {arc.id}: invalid connection from place to place ({arc.source} -> {arc.target})")
                elif source_is_transition and target_is_transition:
                    errors.append(
                        f"Arc {arc.id}: invalid connection from transition to transition ({arc.source} -> {arc.target})")

        # Check 3: All transitions have at least one input place
        for trans_id in self.transitions:
            if not self.input_places[trans_id]:
                errors.append(f"Transition {trans_id} has no input places")

        # Check 4: All transitions have at least one output place
        for trans_id in self.transitions:
            if not self.output_places[trans_id]:
                errors.append(f"Transition {trans_id} has no output places")

        # Check 5: Verify no duplicate arc IDs
        arc_ids = [arc.id for arc in self.arcs]
        if len(arc_ids) != len(set(arc_ids)):
            duplicates = [aid for aid in arc_ids if arc_ids.count(aid) > 1]
            errors.append(f"Duplicate arc IDs found: {set(duplicates)}")

        # Check 6: Verify no duplicate place/transition IDs
        place_ids = list(self.places.keys())
        trans_ids = list(self.transitions.keys())
        all_ids = place_ids + trans_ids
        if len(all_ids) != len(set(all_ids)):
            errors.append("Duplicate node IDs found between places and transitions")

        return errors

    def __repr__(self):
        return (f"PetriNet(places={len(self.places)}, "
                f"transitions={len(self.transitions)}, "
                f"arcs={len(self.arcs)})")


class PNMLParser:
    """Parser for PNML files"""

    def __init__(self):
        self.namespace = {'pnml': 'http://www.pnml.org/version-2009/grammar/pnml'}

    def parse(self, filename: str) -> PetriNet:
        """Parse a PNML file and return a PetriNet object"""
        try:
            tree = ET.parse(filename)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format in PNML file: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"PNML file not found: {filename}")

        root = tree.getroot()

        petri_net = PetriNet()

        # Find the net element
        net = root.find('.//pnml:net', self.namespace)
        if net is None:
            # Try without namespace
            net = root.find('.//net')

        if net is None:
            raise ValueError("No net element found in PNML file")

        # Parse places
        self._parse_places(net, petri_net)

        # Parse transitions
        self._parse_transitions(net, petri_net)

        # Parse arcs
        self._parse_arcs(net, petri_net)

        # Verify consistency
        errors = petri_net.verify_consistency()
        if errors:
            print("\n  Consistency Check Warnings:")
            for error in errors:
                print(f"  - {error}")
            print()

        return petri_net

    def _parse_places(self, net, petri_net: PetriNet):
        """Parse all places from the net element"""
        places = net.findall('.//pnml:place', self.namespace)
        if not places:
            places = net.findall('.//place')

        for place_elem in places:
            place_id = place_elem.get('id')

            # Get name
            name_elem = place_elem.find('.//pnml:name/pnml:text', self.namespace)
            if name_elem is None:
                name_elem = place_elem.find('.//name/text')
            name = name_elem.text if name_elem is not None else ""

            # Get initial marking
            marking_elem = place_elem.find('.//pnml:initialMarking/pnml:text', self.namespace)
            if marking_elem is None:
                marking_elem = place_elem.find('.//initialMarking/text')
            tokens = int(marking_elem.text) if marking_elem is not None else 0

            place = Place(id=place_id, name=name, initial_tokens=tokens)
            petri_net.add_place(place)

    def _parse_transitions(self, net, petri_net: PetriNet):
        """Parse all transitions from the net element"""
        transitions = net.findall('.//pnml:transition', self.namespace)
        if not transitions:
            transitions = net.findall('.//transition')

        for trans_elem in transitions:
            trans_id = trans_elem.get('id')

            # Get name
            name_elem = trans_elem.find('.//pnml:name/pnml:text', self.namespace)
            if name_elem is None:
                name_elem = trans_elem.find('.//name/text')
            name = name_elem.text if name_elem is not None else ""

            transition = Transition(id=trans_id, name=name)
            petri_net.add_transition(transition)

    def _parse_arcs(self, net, petri_net: PetriNet):
        """Parse all arcs from the net element"""
        arcs = net.findall('.//pnml:arc', self.namespace)
        if not arcs:
            arcs = net.findall('.//arc')

        for arc_elem in arcs:
            arc_id = arc_elem.get('id')
            source = arc_elem.get('source')
            target = arc_elem.get('target')

            # Get weight (inscription)
            weight_elem = arc_elem.find('.//pnml:inscription/pnml:text', self.namespace)
            if weight_elem is None:
                weight_elem = arc_elem.find('.//inscription/text')
            weight = int(weight_elem.text) if weight_elem is not None else 1

            arc = Arc(id=arc_id, source=source, target=target, weight=weight)
            petri_net.add_arc(arc)


def print_petri_net_summary(petri_net: PetriNet):
    """Print a summary of the parsed Petri net"""
    print("=" * 60)
    print("PETRI NET PARSING SUMMARY")
    print("=" * 60)
    print(f"Places:      {len(petri_net.places)}")
    print(f"Transitions: {len(petri_net.transitions)}")
    print(f"Arcs:        {len(petri_net.arcs)}")

    # Initial marking
    initial_marking = petri_net.get_initial_marking()
    tokens_count = sum(initial_marking.values())
    print(f"Total initial tokens: {tokens_count}")

    # List places with tokens
    places_with_tokens = [(pid, tokens) for pid, tokens in initial_marking.items() if tokens > 0]
    if places_with_tokens:
        print(f"Places with tokens: {', '.join([f'{p}({t})' for p, t in places_with_tokens])}")

    print("=" * 60)
    print()


def print_petri_net_info(petri_net: PetriNet):
    """Print detailed information about the Petri net"""
    print("=" * 60)
    print("PETRI NET INFORMATION")
    print("=" * 60)

    print(f"\nPlaces ({len(petri_net.places)}):")
    for place_id, place in petri_net.places.items():
        print(f"  {place_id}: {place.name or 'unnamed'} (tokens={place.initial_tokens})")

    print(f"\nTransitions ({len(petri_net.transitions)}):")
    for trans_id, trans in petri_net.transitions.items():
        print(f"  {trans_id}: {trans.name or 'unnamed'}")
        print(f"    Inputs:  {petri_net.input_places[trans_id]}")
        print(f"    Outputs: {petri_net.output_places[trans_id]}")

    print(f"\nArcs ({len(petri_net.arcs)}):")
    for arc in petri_net.arcs:
        print(f"  {arc.id}: {arc.source} -> {arc.target} (weight={arc.weight})")

    print(f"\nInitial Marking:")
    initial_marking = petri_net.get_initial_marking()
    for place_id, tokens in initial_marking.items():
        if tokens > 0:
            print(f"  {place_id}: {tokens}")

    print("=" * 60)


def print_marking_matrix(markings: List[Dict[str, int]], petri_net: PetriNet):
    """Print markings in matrix format"""
    if not markings:
        print("No markings to display")
        return

    # Get sorted list of place IDs
    # Put 'mutex' or 'lock' at the end if it exists, otherwise just sort normally
    all_places = list(petri_net.places.keys())

    # Find if there's a mutex/lock place (common names for synchronization)
    mutex_like = [p for p in all_places if p.lower() in ['mutex', 'lock', 'semaphore']]

    # Sort places, but put mutex-like places at the end
    if mutex_like:
        place_ids = sorted([p for p in all_places if p not in mutex_like])
        place_ids.extend(sorted(mutex_like))
    else:
        # Just sort all places normally
        place_ids = sorted(all_places)

    # Print header
    header = "  ".join(f"{pid:>6}" for pid in place_ids)
    print(header)

    # Print each marking as a row
    for marking in markings:
        row = "  ".join(f"{marking.get(pid, 0):>6}" for pid in place_ids)
        print(row)


# Example usage
if __name__ == "__main__":
    import glob
    import os

    # Find all .pnml files in current directory
    pnml_files = glob.glob("*.pnml")

    if not pnml_files:
        print("No .pnml files found in current directory")
        print("\nTo use this parser:")
        print("1. Place your .pnml files in the same directory as this script")
        print("2. Run the parser again")
    else:
        parser = PNMLParser()

        for filename in pnml_files:
            print(f"\n{'=' * 60}")
            print(f"Processing file: {filename}")
            print(f"{'=' * 60}")

            try:
                petri_net = parser.parse(filename)

                # Only display markings in matrix format
                markings = []
                marking = petri_net.get_initial_marking()
                markings.append(marking.copy())

                # Fire a few transitions to generate more markings
                for trans_id in petri_net.transitions:
                    if petri_net.is_transition_enabled(trans_id, marking):
                        marking = petri_net.fire_transition(trans_id, marking)
                        markings.append(marking.copy())
                        if len(markings) >= 10:  # Limit to 10 markings for display
                            break

                print_marking_matrix(markings, petri_net)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
