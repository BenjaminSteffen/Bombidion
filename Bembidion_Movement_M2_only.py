"""
This file is intended as a starting point to test implementations of bembidion movement. The initial implementation is
reflecting the now outdated code of the ALMaSS bembidion available under
https://gitlab.com/ALMaSS/ALMaSS_stable/-/blob/master/Bembidion/Bembidion_all.cpp. However, instead of initializing
each bembidion as an object, applies expressions to a population of bembidions represented in a polars data frame.
"""
from pathlib import Path

import polars_random
import polars as pl

# The set of alternative headings (in degrees) evaluated when an individual's current heading fails. They are
# applied relative to the direction the individual had before the search started this step and cover the entire
# remaining circle (45-degree steps) exactly once each. All seven headings are evaluated simultaneously in Phase 2
# of move_to below, and one of the successful ones is then chosen uniformly at random, so the order of this list
# has no effect on the outcome.
DIRECTION_SEARCH_OFFSETS = [45, -45, 90, -90, 135, -135, 180]


def move_to(
        individuals: pl.DataFrame,
        landscape: pl.DataFrame,
        max_distance: int,
        turning_probability: float
) -> pl.DataFrame:
    """
    This function implements the daily movement of dispersing bembidions. In its initial form, it should be functionally
    equal to the ALMaSS function Bembidion_Adult::MoveTo. However, instead of specifying the movement for an individual
    bembidion, it operates on the entire population of bembidions passed to the function.

    Args:
        individuals: A polars dataframe representing the individuals to move. Each row of the dataframe represents one
            individual. The dataframe must have at least three columns characterizing the individual: a column x with
            the current x-coordinate of the individual, a column y with its current y-coordinate and a column direction.
            The direction column must be present (it is used and updated throughout the day's movement and returned
            as part of the result), but its incoming values are ignored: per M2, each day's starting heading is drawn
            freshly and independently of the previous day, so the direction a bembidion ends the day with has no
            influence on the direction it starts the next day with. Valid values are one of eight headings: 0, 45,
            90, 135, 180, 225, 270, or 315.
        landscape: A polars dataframe containing information about the accessibility of patches for bembidions. The
            dataframe has at least three columns: a column x with the integer x-coordinate of the one square
            meter-patch, a column y with its y-coordinate, and a column accessibility with one of the three string
            literals full, partial or none indicating the accessibility of the patch for the bembidions.
        max_distance: The maximum distance each individual will move in steps. A step can be either one meter in
            horizontal or vertical direction or sqrt(1) m in diagonal direction. The actual number of steps of each
            bembidion is randomly sampled from a uniform distribution. The lower bound of movement is always one step.
        turning_probability: Unused by this M2-only fix (see the direction fix above) - kept as a parameter only for
            interface compatibility with existing call sites.

    Returns:
        A Polars dataframe with the same schema as the individuals argument, but with updated coordinates and direction
        of bembidions.
    """
    # We have to initially prepare additiona state variables defining the movement of the individuals.
    individuals = (

        # Our starting point is the population of individuals passed to the function.
        individuals

        # The first thing a bembidion does in the ALMaSS code prior movement is determining how many steps it does. We
        # add this information by randomly drawing from a uniform distribution with max_distance as its upper bound.
        .with_columns(

            # We add their desired movement distance in steps. The upper bound is exlcusive, so we have to add 1 to
            # include it as a possible sampling result.
            distance=polars_random.randint(1, max_distance + 1),

            # M2: the direction a bembidion starts moving in on a new day should not depend on the direction it
            # ended up with the previous day. We therefore overwrite the incoming direction with a freshly and
            # uniformly drawn heading out of the eight allowed values (0, 45, ..., 315), drawn once at the start of
            # the day (this deliberately does NOT redraw direction at every movement step - that would be M1, which
            # is out of scope here). The direction then stays fixed for the rest of the day, same as before this fix.
            #
            # Note this replaces the previous turning_probability-based logic (turn the incoming direction left/right
            # with some probability) rather than keeping it and just changing what it turns from: turning a heading
            # that was already drawn uniformly at random would not change its distribution (turning is symmetric
            # left/right), so it would add computation without adding information. turning_probability is kept as a
            # parameter for interface compatibility with existing call sites, but is unused by this fix.
            direction=polars_random.randint(high=8) * 45
        )
    )

    # In this list, we collect all individuals that have finished their movement.
    individuals_after_movement = []

    # Bembidions do one step after each other.
    for i in range(max_distance):
        # We output the current step for better development feedback.
        print(f"Movement step {i + 1}")

        # Stop early if there are no more individuals wanting to move
        if individuals is None:
            break

        # As a first action for each step, we partition the individuals according to their need to move further.
        movement = individuals.with_columns(step_done=pl.col("distance").le(i)).partition_by("step_done", as_dict=True)

        # There up to two different groups: individuals that still need to move and those that are finished moving.
        movement_finished = movement.get((True,))
        movement_required = movement.get((False,))

        # If there are individuals with finished movement, their current state is taken as their new state after
        # movement.
        if movement_finished is not None:
            individuals_after_movement.append(movement_finished.drop("step_done", "distance"))

        # For individuals that still need to move this step, we add a synthetic row id. It is used below to group
        # several candidate headings evaluated for the same individual back together when picking an escape
        # direction after a failed movement attempt.
        if movement_required is not None:
            movement_required = movement_required.with_row_index("individual_id")

        # In this list, we collect the results of this step.
        step_results = []

        # Each individual first tries to keep moving in its current heading (Phase 1). Individuals for whom that
        # fails go through a vectorized search step (Phase 2) that evaluates all seven remaining headings at once
        # and picks uniformly at random among whichever of them succeed, instead of retrying with a random turn.
        if movement_required is not None:

            # --- Phase 1: try the individual's current heading. ---
            trial_results = (

                # The starting point of each movement step is always the current state of all individuals requiring
                # movement.
                movement_required

                # The current positions of the bembidions and their movement directions have to be translated into
                # target coordinates of the movement.
                .with_columns(

                    # The x-coordinate is straightforward.
                    new_x=pl
                    .when(pl.col("direction").is_in((225, 270, 315))).then(pl.col("x") - 1)
                    .when(pl.col("direction").is_in((45, 90, 135))).then(pl.col("x") + 1)
                    .otherwise("x"),

                    # For the y-coordinate, it should be considered that in the Northern Hemisphere coordinates increase
                    # in the direction of North.
                    new_y=pl
                    .when(pl.col("direction").is_in((315, 0, 45))).then(pl.col("y") + 1)
                    .when(pl.col("direction").is_in((225, 180, 135))).then(pl.col("y") - 1)
                    .otherwise("y"),

                    # We add also a random number that indicates the decision of a bembidion to access an only partially
                    # accessible patch.
                    access=polars_random.uniform()
                )

                # We can now learn about the accessibility of the target patch by joining the bembidion and the movement
                # map dataframes by coordinates. We need a left join so that we also keep bembidions that move outside
                # the defined landscape.
                .join(landscape, left_on=("new_x", "new_y"), right_on=("x", "y"), how="left")

                # We can now check whether the intended movement is successful.
                .with_columns(
                    success=(
                            pl.col("accessibility").eq("full") |
                            (pl.col("accessibility").eq("partial") & pl.col("access").lt(.4))
                    )
                )
                .partition_by("success", as_dict=True)
            )

            # There are up to three groups of movement trial results: individuals successful with successful movement,
            # individuals with unsuccessful movements and individuals that want to move outside the defined landscape.
            successful_trial = trial_results.get((True,))
            unsuccessful_trial = trial_results.get((False,))
            leaving_map = trial_results.get((None,))

            # Individuals with successful movement update their coordinates and are added to the list of step results.
            if successful_trial is not None:
                step_results.append(
                    successful_trial.select(x="new_x", y="new_y", direction="direction", distance="distance")
                )

            # Individuals for whom Phase 1 failed (destination inaccessible, or outside the landscape) go through
            # Phase 2 below.
            blocked = [df for df in (unsuccessful_trial, leaving_map) if df is not None]
            blocked = pl.concat(blocked) if len(blocked) > 0 else None

            if blocked is not None:

                # --- Phase 2: Alle 7 Ausweichrichtungen gleichzeitig testen ---
                candidates = (
                    blocked
                    .select("individual_id", "x", "y", "direction", "distance")
                    .join(pl.DataFrame({"offset": DIRECTION_SEARCH_OFFSETS}), how="cross")
                    .with_columns(
                        heading=(((pl.col("direction") + pl.col("offset")) % 360) + 360) % 360
                    )
                    .with_columns(
                        new_x=pl.when(pl.col("heading").is_in((225, 270, 315))).then(pl.col("x") - 1)
                                .when(pl.col("heading").is_in((45, 90, 135))).then(pl.col("x") + 1)
                                .otherwise(pl.col("x")),
                        new_y=pl.when(pl.col("heading").is_in((315, 0, 45))).then(pl.col("y") + 1)
                                .when(pl.col("heading").is_in((225, 180, 135))).then(pl.col("y") - 1)
                                .otherwise(pl.col("y")),
                        access=polars_random.uniform(),
                        priority=polars_random.uniform()
                    )
                    .join(landscape, left_on=("new_x", "new_y"), right_on=("x", "y"), how="left")
                    .with_columns(
                        success=(
                            pl.col("accessibility").eq("full") |
                            (pl.col("accessibility").eq("partial") & pl.col("access").lt(.4))
                        ).fill_null(False)  # Verhindert Null-Werte am Kartenrand
                    )
                )
                successful_candidates = candidates.filter(pl.col("success"))
                if not successful_candidates.is_empty():
                    chosen = (
                        successful_candidates
                        .filter(pl.col("priority") == pl.col("priority").max().over("individual_id"))
                        .unique(subset="individual_id", keep="first")
                    )
                    step_results.append(
                        chosen.select(x="new_x", y="new_y", direction="heading", distance="distance")
                    )
                    # Blockierte Käfer ermitteln, die KEINE Ausweichrichtung gefunden haben:
                    stuck = blocked.join(chosen.select("individual_id"), on="individual_id", how="anti")
                else:
                    # Niemand konnte ausweichen -> alle Blockierten stecken fest
                    stuck = blocked
                if not stuck.is_empty():
                    step_results.append(
                        stuck.select("x", "y", "direction", "distance")
                    )

        # Set the individuals for the next step.
        if len(step_results) > 0:
            individuals = pl.concat(step_results)
        else:
            individuals = None

        # Add the last batch of individuals to the result.
        if i == max_distance - 1 and individuals is not None:
            individuals_after_movement.append(individuals.drop("distance"))

    # Return the movement results
    return pl.concat(individuals_after_movement)


# All result CSVs are written into a dedicated subfolder of "data" instead of directly into "data" itself. This way,
# different runs can be executed one after another without overwriting each other's output. Just change run_name
# before a run and every write_csv call below will land in its own folder.
run_name = "run"
output_dir = Path("data") / run_name
output_dir.mkdir(parents=True, exist_ok=True)

# To test the emergent effects of our movement implementation, we need a landscape scenario. For the bembidion, this
# is basically a map that tells for each square meter in the landscape whether a bembidion would go there. Such a
# scenario can be prepared in a GIS process (see Scenario preparation Jupyter notebook) or by any other means.
scenario = "data/lulc.parquet"
movement_map = pl.read_parquet(scenario)

# We can learn something about the landscape scenario by looking into the movement map. For instance, we can analyze
# the areal fractions (in square meters) of the landscape that are full, partially or not accessible for bembidions.
print(scenario)
print(movement_map.group_by("accessibility").agg(pl.len()))

# Next, we need an initial population of bembidions. We want to randomly scatter them around the landscape, place not
# more than one into a square meter-cell and avoid spawning them in inaccessible habitat. We can all do that by sampling
# the movement map after filtering for at least partially accessible terrain and keeping only the x- and y- coordinates.
bembidions = movement_map.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1_000, shuffle=True)

# In this case, we spawned 1,000 bembidions, each represented by a row in a data frame.
print(bembidions)

# If we like to see where these bembidions are, we can save the list as a CSV file and add them together with the
# original LULC layer of the scenario (see scenario preparation) to a GIS project. Coordinates are in the same CRS as
# the original LULC layer. See "example start locations.png" in the data folder for an example.
bembidions.write_csv(output_dir / "example_start_locations.csv")

# According to the ALMaSS code, each bembidion also has to remember its last movement directions, latter which can only
# have one of eight values. We will assign directions to the initial bembidions randomly.
bembidions = bembidions.with_columns(direction=polars_random.randint(high=8) * 45)

# Once again, this is nice to visualize on a map. See "example start locations with directions.png".
bembidions.write_csv(output_dir / "example_start_locations_with_directions.csv")

# We can now simulate a single day's movement of all bembidions using our movement implementation. We have to pass a
# probability for each bembidion to change its direction of the previous day (80% in ALMaSS) and the upper bound for
# the movement distance in steps (14 in ALMaSS).
print(move_to(bembidions, movement_map, 14, .8))

# And we can simulate an entire year of movement.
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions = move_to(bembidions, movement_map, 14, .8)
bembidions.write_csv(output_dir / "locations_with_directions_after_one_year.csv")

# The same for a small sub-landscape
movement_map2 = movement_map.filter(pl.col("x") <= 319977).filter(pl.col("y") <= 5702396)
bembidions2 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1_000, shuffle=True)
bembidions2 = bembidions2.with_columns(direction=polars_random.randint(high=8) * 45)
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions2 = move_to(bembidions2, movement_map2, 14, .8)
bembidions2.write_csv(output_dir / "locations_with_directions_after_one_year_small.csv")

# And for ten years
bembidions2 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1_000, shuffle=True)
bembidions2 = bembidions2.with_columns(direction=polars_random.randint(high=8) * 45)
for day in range(3650):
    print(f"*** DAY {day + 1} ***")
    bembidions2 = move_to(bembidions2, movement_map2, 14, .8)
bembidions2.write_csv(output_dir / "locations_with_directions_after_ten_years_small.csv")

# here, we keep track of the movement of a single individual to study its behavior
bembidions3 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1, shuffle=True)
bembidions3 = bembidions3.with_columns(direction=polars_random.randint(high=8) * 45)
results = []
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions3 = move_to(bembidions3, movement_map2, 14, .8)
    results.append(bembidions3.with_columns(t=day))
pl.concat(results).write_csv(output_dir / "movement_one_year_small.csv")
