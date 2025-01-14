import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pulp
from tqdm import tqdm

logger = logging.getLogger("assigning_primary_zone")


@dataclass
class WorkZoneAssignment:
    """
    A class to handle the assignment of individuals to zones while respecting origin-destination (OD) constraints.

    Attributes
    ----------
    activities_to_assign : dict
        A dictionary where the keys are the person/activity IDs and the values are dictionaries where keys are the origin zones and the values are the feasible destination zones.
        Example: {164: {'E00059011': ['E00056917','E00056922', 'E00056923']},
                  165: {'E00059012': ['E00056918','E00056952', 'E00056923']}}
    actual_flows : dict
        A dictionary where the keys are the origin-destination zone pairs and the values are the number of flows between the origin and destination zones.
        Example: {('E00059011', 'E00056917'): 10,
                  ('E00059011', 'E00056922'): 5}
    remaining_flows : dict
        A dictionary to track the remaining flows between origin-destination pairs. Initialized from actual_flows.
    total_flows : dict
        A dictionary to track the total flows for each origin zone. Calculated in the __post_init__ method.
    percentages : dict
        A dictionary to store the flow percentages for each origin-destination pair. Calculated in the __post_init__ method.

    Methods
    -------
    __post_init__():
        Initializes remaining_flows, total_flows, and percentages after the dataclass is created.
    _calculate_total_flows() -> Dict[str, int]:
        Calculates the total flows for each origin zone.
    _calculate_percentages() -> Dict[Tuple[str, str], float]:
        Calculates the flow percentages for each origin-destination pair.
    select_work_zone_iterative(random_assignment: bool = False) -> pd.DataFrame:
        Assigns individuals to work zones iteratively while respecting OD constraints.
    select_work_zone_optimization(use_percentages: bool = False) -> pd.DataFrame:
        Assigns individuals to work zones using optimization to minimize deviation from actual flows.
    """

    activities_to_assign: Dict[int, Dict[str, List[str]]]
    actual_flows: Dict[Tuple[str, str], int]
    remaining_flows: Dict[Tuple[str, str], int] = field(init=False)
    total_flows: Dict[str, int] = field(init=False)
    percentages: Dict[Tuple[str, str], float] = field(init=False)

    def __post_init__(self):
        """
        Post-initialization method to set up remaining_flows, total_flows, and percentages.
        """
        # Initialize remaining flows dictionary directly from actual_flows
        self.remaining_flows = self.actual_flows.copy()
        # Calculate total flows originating from each origin zone
        self.total_flows = self._calculate_total_flows()
        # Calculate percentages for each origin-destination pair
        self.percentages = self._calculate_percentages()

    def _calculate_total_flows(self) -> Dict[str, int]:
        """
        Calculate the total flows originating from each origin zone.

        Returns
        -------
        total_flows : dict
            A dictionary with origin zones as keys and total flows as values.
        """
        total_flows = {}
        for (from_zone, _), flow in self.actual_flows.items():
            if from_zone in total_flows:
                total_flows[from_zone] += flow
            else:
                total_flows[from_zone] = flow
        return total_flows

    def _calculate_percentages(self) -> Dict[Tuple[str, str], float]:
        """
        Calculate the flow percentages for each origin-destination pair.
        Percentage for ech Origin-Destination is in porportion to the total flows from the origin zone.
        Percentage = flow / total_flows[from_zone]

        Returns
        -------
        percentages : dict
            A dictionary with origin-destination pairs as keys and flow percentages as values.
        """
        percentages = {}
        for (from_zone, to_zone), flow in self.actual_flows.items():
            percentage = flow / self.total_flows[from_zone]
            percentages[(from_zone, to_zone)] = percentage
        return percentages

    def select_work_zone_iterative(
        self, random_assignment: bool = False
    ) -> pd.DataFrame:
        """
        Assigns individuals to work zones while respecting the OD constraints from an external
        dataset (e.g. census flow data).

        This function iterates over each individual and their origin zones. For each individual,
        it creates a list of feasible destination zones that still have remaining flows. If there
        are such zones, it assigns the individual to one of them using a weighted random selection
        based on the remaining flows. If there are no feasible zones with remaining flows, the
        function either assigns the individual to a random feasible zone (if random_assignment is
        True) or skips the assignment (if random_assignment is False).

        After each assignment, the function updates the remaining flows between the origin and
        destination zones. The process continues until all individuals are assigned to a work zone .

        The function returns a DataFrame with the following columns:
        - 'activity_id': the ID of the individual activity
        - 'origin_zone': the origin zone of the individual activity
        - 'assigned_zone': the zone to which the individual activity was assigned
        - 'assignment_type': the method used to assign the individual activity ('Weighted' for weighted
        random selection based on remaining flows, 'Random' for random selection, or None if
        the assignment was skipped)

        Parameters
        ----------
        activities_to_assign : dict
            A dictionary where the keys are the person/activity IDs and the values are dictionaries
            where keys are the origin zones and the values are the feasible destination zones.
            example: {164: {'E00059011': ['E00056917','E00056922', 'E00056923']},
                      165: {'E00059012': ['E00056918','E00056952', 'E00056923']}}

        actual_flows : dict
            A dictionary where the keys are the origin-destination zone pairs and the values are the
            number of flows between the origin and destination zones. The intended use case is the
            UK census flow data.
            example: {('E00059011', 'E00056917'): 10,
                      ('E00059011', 'E00056922'): 5}

        random_assignment : bool, optional
            If True, the assignment of individuals to zones will be random when there are no feasible
            zones with remaining flows. If False, the assignment will be skipped for that individual.
            Default is False.
        """

        logger.info("Starting the iterative assignment process.")
        assignments = []

        for activity_id, origins in self.activities_to_assign.items():
            for origin_id, feasible_zones in origins.items():
                logger.info(
                    f"Processing activity {activity_id} from origin {origin_id}."
                )
                logger.debug(f"{activity_id}: {len(feasible_zones)} feasible zones")
                if feasible_zones:
                    weighted_zones = []
                    for zone in feasible_zones:
                        flow = self.remaining_flows.get((origin_id, zone), 0)
                        if flow > 0:
                            weighted_zones.append((zone, flow))

                    logger.debug(
                        f"{activity_id}: {len(weighted_zones)} feasible zones with remaining flows"
                    )
                    if weighted_zones:
                        zones, weights = zip(*weighted_zones)
                        assigned_zone = np.random.choice(
                            zones, p=weights / np.sum(weights)
                        )
                        assignment_type = "Weighted"
                        self.remaining_flows[(origin_id, assigned_zone)] -= 1
                        logger.info(
                            f"Assigned zone {assigned_zone} to person {activity_id} using weighted random selection."
                        )

                    elif random_assignment:
                        assigned_zone = np.random.choice(feasible_zones)
                        assignment_type = "Random"
                        logger.info(
                            f"Assigned zone {assigned_zone} to person {activity_id} using random selection."
                        )
                    else:
                        assigned_zone = None
                        assignment_type = None
                        logger.info(
                            f"{activity_id}: No feasible zones with remaining flows for person. Assigned NA"
                        )
                else:
                    logger.info(f"{activity_id}: No feasible zones. Assigned NA")
                    assigned_zone = None
                    assignment_type = None

                assignments.append(
                    {
                        "activity_id": activity_id,
                        "origin_zone": origin_id,
                        "assigned_zone": assigned_zone,
                        "assignment_type": assignment_type,
                    }
                )

        logger.info("Iterative assignment process completed.")
        return pd.DataFrame(assignments)

    # def select_work_zone_optimization(
    #     self,
    #     use_percentages: bool = False,
    #     weight_max_dev: int = 0.5,
    #     weight_total_dev: int = 0.5,
    # ) -> pd.DataFrame:
    #     """
    #     Assigns individuals to zones while minimizing the deviation between the assigned and actual flows.
    #     We minimize two deviations
    #     1. The sum of the deviations between the assigned and actual flows for each origin-destination pair.
    #     2. The maximum deviation across all origin-destination pairs.

    #     If we use (1) only, positive and negative deviations could cancel each other out, and
    #     the results would not match the actual flows even though the sum of deviations is minimized.

    #     The assigned flows are constrained to match the actual flows as closely as possible, either in terms
    #     of absolute numbers or percentages of the total flows from each origin zone.

    #     The function returns a DataFrame with the following columns:

    #     - 'person_id': the ID of the individual
    #     - 'origin_zone': the origin zone of the individual
    #     - 'assigned_zone': the zone to which the individual was assigned

    #     Parameters
    #     ----------

    #     activities_to_assign : dict
    #         A dictionary where the keys are the person/activity IDs and the values are dictionaries
    #         where keys are the origin zones and the values are the feasible destination zones.
    #         example: {164: {'E00059011': ['E00056917','E00056922', 'E00056923']},
    #                   165: {'E00059012': ['E00056918','E00056952', 'E00056923']}}

    #     actual_flows : dict
    #         A dictionary where the keys are the origin-destination zone pairs and the values are the
    #         number of flows between the origin and destination zones. The intended use case is the
    #         UK census flow data.
    #         example: {('E00059011', 'E00056917'): 10,
    #                   ('E00059011', 'E00056922'): 5}

    #     use_percentages : bool, optional
    #         If True, the optimization problem will minimize the deviation between the assigned and actual
    #         flows as percentages of the total flows from each origin zone. If False, the deviation will be
    #         minimized in terms of the absolute number of flows. Default is False.

    #     weight_max_dev : int, optional
    #         The weight assigned to the maximum deviation in the objective function. Default is 0.5.

    #     weight_total_dev : int, optional
    #         The weight assigned to the total deviation in the objective function. Default is

    #     Returns
    #     -------

    #     pd.DataFrame
    #         A DataFrame containing the assigned zones for each individual. The columns are 'person_id',
    #         'origin_zone', and 'assigned_zone
    #     """
    #     # Step 1: Compute total flows per origin
    #     # Step 2: Calculate percentages (normalize over total flow)

    #     # Step 1 and 2 are already done in the __post_init__ method

    #     # Step 3: Initialize the optimization problem
    #     logger.info("Starting the optimization assignment process.")
    #     prob = pulp.LpProblem("ZoneAssignment", pulp.LpMinimize)
    #     # Create dictionaries to store the assignment and deviation variables
    #     assignment_vars = {}
    #     deviation_vars = {}

    #     # Variable to track the maximum deviation across all OD pairs
    #     max_dev = pulp.LpVariable("max_dev", 0, None, pulp.LpContinuous)

    #     # Create binary variables for each person, origin, and destination
    #     for person_id, origins in self.activities_to_assign.items():
    #         for origin_id, feasible_zones in origins.items():
    #             person_vars = []
    #             for zone in feasible_zones:
    #                 if (origin_id, zone) in self.percentages:
    #                     var = pulp.LpVariable(
    #                         f"assign_{person_id}_{origin_id}_{zone}",
    #                         0,
    #                         1,
    #                         pulp.LpBinary,
    #                     )
    #                     person_vars.append(var)
    #                     assignment_vars[(person_id, origin_id, zone)] = var
    #             # Constraint: Assign each person to exactly one zone
    #             if person_vars:
    #                 prob += pulp.lpSum(person_vars) == 1
    #     # Calculate assigned percentages and deviation for each origin-destination pair
    #     for (from_zone, to_zone), percentage in self.percentages.items():
    #         # Create a variable for the absolute deviation (with lower bound of 0)
    #         abs_dev = pulp.LpVariable(
    #             f"abs_dev_{from_zone}_{to_zone}", 0, None, pulp.LpContinuous
    #         )
    #         deviation_vars[(from_zone, to_zone)] = abs_dev
    #         # Calculate the assigned flow for the origin-destination pair
    #         assigned_flow = pulp.lpSum(
    #             assignment_vars.get((person_id, from_zone, to_zone), 0)
    #             for person_id, origins in self.activities_to_assign.items()
    #             if (person_id, from_zone, to_zone) in assignment_vars
    #         )
    #         # Calculate the assigned percentage based on the total flows from the origin zone
    #         if from_zone in self.total_flows:
    #             total_people = self.total_flows[from_zone]
    #             assigned_percentage = assigned_flow / total_people
    #         # If the origin zone is not in the total flows, set the assigned percentage to 0
    #         else:
    #             assigned_percentage = 0
    #             logger.warning(f"Warning: Origin {from_zone} not found in total_flows.")

    #         if use_percentages:
    #             # to satisfy both constraints, abs_dev will be a positive number (it is the larger of the two)
    #             prob += assigned_percentage - percentage <= abs_dev
    #             prob += percentage - assigned_percentage <= abs_dev
    #         else:
    #             prob += (
    #                 assigned_flow - self.actual_flows[(from_zone, to_zone)] <= abs_dev
    #             )
    #             prob += (
    #                 self.actual_flows[(from_zone, to_zone)] - assigned_flow <= abs_dev
    #             )

    #         # Update the maximum deviation variable: it is the maximum of all deviations
    #         prob += max_dev >= abs_dev

    #     # Weighted objective function
    #     prob += (
    #         weight_max_dev * max_dev
    #         + weight_total_dev * pulp.lpSum(deviation_vars.values()),
    #         "WeightedObjective",
    #     )
    #     prob.solve()

    #     if pulp.LpStatus[prob.status] != "Optimal":
    #         logger.warning("Problem is infeasible.")
    #         return pd.DataFrame()

    #     assignments = [
    #         {"person_id": person_id, "origin_zone": origin_id, "assigned_zone": zone}
    #         for (person_id, origin_id, zone), var in assignment_vars.items()
    #         if var.varValue == 1
    #     ]

    #     logger.info("Optimization assignment process completed.")
    #     return pd.DataFrame(assignments)

    def select_work_zone_optimization(
        self,
        use_percentages: bool = False,
        weight_max_dev: float = 0.5,
        weight_total_dev: float = 0.5,
        max_zones: int = 10,
    ) -> pd.DataFrame:
        """
        Assigns individuals to zones while minimizing the deviation between the assigned and actual flows.
        We minimize two deviations
        1. The sum of the deviations between the assigned and actual flows for each origin-destination pair.
        2. The maximum deviation across all origin-destination pairs.

        If we use (1) only, positive and negative deviations could cancel each other out, and
        the results would not match the actual flows even though the sum of deviations is minimized.

        The assigned flows are constrained to match the actual flows as closely as possible, either in terms
        of absolute numbers or percentages of the total flows from each origin zone.

        The function returns a DataFrame with the following columns:

        - 'person_id': the ID of the individual
        - 'origin_zone': the origin zone of the individual
        - 'assigned_zone': the zone to which the individual was assigned

        Parameters
        ----------

        activities_to_assign : dict
            A dictionary where the keys are the person/activity IDs and the values are dictionaries
            where keys are the origin zones and the values are the feasible destination zones.
            example: {164: {'E00059011': ['E00056917','E00056922', 'E00056923']},
                    165: {'E00059012': ['E00056918','E00056952', 'E00056923']}}

        actual_flows : dict
            A dictionary where the keys are the origin-destination zone pairs and the values are the
            number of flows between the origin and destination zones. The intended use case is the
            UK census flow data.
            example: {('E00059011', 'E00056917'): 10,
                    ('E00059011', 'E00056922'): 5}

        use_percentages : bool, optional
            If True, the optimization problem will minimize the deviation between the assigned and actual
            flows as percentages of the total flows from each origin zone. If False, the deviation will be
            minimized in terms of the absolute number of flows. Default is False.

        weight_max_dev : int, optional
            The weight assigned to the maximum deviation in the objective function. Default is 0.5.

        weight_total_dev : int, optional
            The weight assigned to the total deviation in the objective function. Default is 0.5.

        max_zones : int, optional
            The maximum number of zones to consider for each person. Default is 10. For some activites,
            we can have a large number of zones to consider. To limit the number of variables in the
            optimization problem, we select a subset of zones based on the actual flows.

        Returns
        -------

        pd.DataFrame
            A DataFrame containing the assigned zones for each individual. The columns are 'person_id',
            'origin_zone', and 'assigned_zone
        """
        # Step 1: Compute total flows per origin
        # Step 2: Calculate percentages (normalize over total flow)

        # Step 1 and 2 are already done in the __post_init__ method

        # Step 3: Initialize the optimization problem
        logger.info("Starting the optimization assignment process.")
        prob = pulp.LpProblem("ZoneAssignment", pulp.LpMinimize)
        # Create dictionaries to store the assignment and deviation variables
        assignment_vars = {}
        deviation_vars = {}

        # Variable to track the maximum deviation across all OD pairs
        max_dev = pulp.LpVariable("max_dev", 0, None, pulp.LpContinuous)

        # Create binary variables for each person, origin, and destination
        logger.info("Adding people to assignment problem")
        for person_id, origins in tqdm(self.activities_to_assign.items()):
            # logger.info(f"person id: {person_id}, origins: {origins}")
            for origin_id, feasible_zones in origins.items():
                person_vars = []
                # Select up to 10 zones for each person based on the actual flows (to limit number of variables)
                # Identify zones in actual_flows
                zones_in_flows = [
                    zone
                    for zone in feasible_zones
                    if (origin_id, zone) in self.actual_flows
                ]

                selected_zones = []
                if zones_in_flows:
                    # Extract flow values to use as weights
                    weights = [
                        self.actual_flows[(origin_id, zone)] for zone in zones_in_flows
                    ]
                    # Select up to 10 zones based on their flow values
                    selected_zones_indices = np.random.choice(
                        range(len(zones_in_flows)),
                        size=min(max_zones, len(zones_in_flows)),
                        replace=False,
                        p=weights / np.sum(weights),  # Normalize weights to sum to 1
                    )
                    selected_zones = [zones_in_flows[i] for i in selected_zones_indices]
                elif feasible_zones:
                    # If no zones in actual_flows, select up to 10 random zones from feasible zones
                    selected_zones = np.random.choice(
                        feasible_zones,
                        min(max_zones, len(feasible_zones)),
                        replace=False,
                    )

                for zone in selected_zones:
                    var = pulp.LpVariable(
                        f"assign_{person_id}_{origin_id}_{zone}",
                        0,
                        1,
                        pulp.LpBinary,
                    )
                    person_vars.append(var)
                    assignment_vars[(person_id, origin_id, zone)] = var

                # Constraint: Assign each person to exactly one zone
                if person_vars:
                    prob += pulp.lpSum(person_vars) == 1

        if weight_max_dev > 0.0:
            logger.info("Added max deviations to assignment problem")
            # Calculate assigned percentages and deviation for each origin-destination pair
            for (from_zone, to_zone), percentage in tqdm(self.percentages.items()):
                # Create a variable for the absolute deviation (with lower bound of 0)
                abs_dev = pulp.LpVariable(
                    f"abs_dev_{from_zone}_{to_zone}", 0, None, pulp.LpContinuous
                )
                deviation_vars[(from_zone, to_zone)] = abs_dev
                # Calculate the assigned flow for the origin-destination pair
                assigned_flow = pulp.lpSum(
                    assignment_vars.get((person_id, from_zone, to_zone), 0)
                    for person_id, origins in self.activities_to_assign.items()
                    if (person_id, from_zone, to_zone) in assignment_vars
                )
                # Calculate the assigned percentage based on the total flows from each origin zone
                if from_zone in self.total_flows:
                    total_people = self.total_flows[from_zone]
                    assigned_percentage = assigned_flow / total_people
                else:
                    assigned_percentage = 0
                    logger.warning(
                        f"Warning: Origin {from_zone} not found in total_flows."
                    )

                if use_percentages:
                    # to satisfy both constraints, abs_dev will be a positive number (it is the larger of the two)
                    prob += assigned_percentage - percentage <= abs_dev
                    prob += percentage - assigned_percentage <= abs_dev
                else:
                    prob += (
                        assigned_flow - self.actual_flows[(from_zone, to_zone)]
                        <= abs_dev
                    )
                    prob += (
                        self.actual_flows[(from_zone, to_zone)] - assigned_flow
                        <= abs_dev
                    )

                # Update the maximum deviation variable: it is the maximum of all deviations
                prob += max_dev >= abs_dev

            logger.info("Adding weighted objective to assignment problem")
            # Weighted objective function
            prob += (
                weight_max_dev * max_dev
                + weight_total_dev * pulp.lpSum(deviation_vars.values()),
                "WeightedObjective",
            )
        else:
            logger.info("Adding weighted objective to assignment problem")
            # Weighted objective function
            prob += (
                weight_total_dev * pulp.lpSum(deviation_vars.values()),
                "WeightedObjective",
            )

        logger.info("Solving assignment problem")
        prob.solve()

        if pulp.LpStatus[prob.status] != "Optimal":
            logger.warning("Problem is infeasible.")
            return pd.DataFrame()

        assignments = [
            {"person_id": person_id, "origin_zone": origin_id, "assigned_zone": zone}
            for (person_id, origin_id, zone), var in assignment_vars.items()
            if var.varValue == 1
        ]

        logger.info("Optimization assignment process completed.")
        return pd.DataFrame(assignments)
