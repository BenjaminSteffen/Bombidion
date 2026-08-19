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

# M3: default standard deviation (in meters) of the Gaussian home-range kernel used in move_to below. Chosen as a
# starting point given that B. lampros is reported to rarely move more than a few meters from its daily starting
# point; adjust and calibrate against field/telemetry data as needed.
HOME_RANGE_SIGMA = 5.0

# M4: which boundary condition move_to applies when a step would take an individual outside the extent of the
# landscape dataframe (i.e. beyond its min/max x and y). Valid values:
#   "none"       - status quo: the step is simply inaccessible there (like stepping into "none" habitat), triggering
#                  the Phase 2 alternate-heading search; if that also fails the individual stays put. This is the
#                  behaviour responsible for the unrealistic edge aggregation described in M4.
#   "torus"      - the landscape wraps around: leaving one edge re-enters on the opposite edge.
#   "reflecting" - the edge acts as a wall: the step is mirrored back inside, and the heading component that would
#                  have crossed the edge is inverted, so the individual "bounces off" the boundary.
#   "absorbing"  - the individual is removed from the simulation the moment a step would cross the edge (e.g.
#                  modelling emigration/death at the landscape edge).
# Change this to compare behaviours; see also the boundary_condition parameter of move_to.
BOUNDARY_CONDITION = "torus"


def _apply_boundary_condition(
        df: pl.DataFrame,
        boundary_condition: str,
        x_min: int, x_max: int, y_min: int, y_max: int,
        x_col: str = "new_x", y_col: str = "new_y", heading_col: str = "direction"
) -> pl.DataFrame:
    """
    Adjusts candidate target coordinates in x_col/y_col (and, for "reflecting", the heading in heading_col) of df
    according to boundary_condition (M4), before they are checked against the landscape's accessibility. Since
    every step changes x_col/y_col by at most 1 relative to the individual's previous, in-bounds position, a step
    can leave the [x_min, x_max] x [y_min, y_max] extent by at most 1 unit, which keeps the wrapping/mirroring math
    below a single, non-iterative correction.

    "none" and "absorbing" do not need any coordinate adjustment here: "none" is handled implicitly by the
    landscape join returning null accessibility for out-of-bounds coordinates, and "absorbing" individuals are
    filtered out by the caller before this function would otherwise apply - both are left untouched if passed in.
    """
    if boundary_condition == "torus":
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        return df.with_columns(**{
            x_col: x_min + (pl.col(x_col) - x_min) % width,
            y_col: y_min + (pl.col(y_col) - y_min) % height,
        })

    if boundary_condition == "reflecting":
        return (
            df
            # First record which edge(s) a step would cross, before the coordinates themselves are overwritten.
            .with_columns(
                _crossed_x=(pl.col(x_col) < x_min) | (pl.col(x_col) > x_max),
                _crossed_y=(pl.col(y_col) < y_min) | (pl.col(y_col) > y_max),
            )
            # Mirror the coordinate back across the edge it crossed.
            .with_columns(**{
                x_col: pl.when(pl.col(x_col) < x_min).then(2 * x_min - pl.col(x_col))
                        .when(pl.col(x_col) > x_max).then(2 * x_max - pl.col(x_col))
                        .otherwise(pl.col(x_col)),
                y_col: pl.when(pl.col(y_col) < y_min).then(2 * y_min - pl.col(y_col))
                        .when(pl.col(y_col) > y_max).then(2 * y_max - pl.col(y_col))
                        .otherwise(pl.col(y_col)),
            })
            # Invert the heading component(s) responsible for crossing the edge, so the individual moves away from
            # the boundary afterwards instead of immediately trying to cross it again next step. Flipping the
            # x-component of a heading is (360 - heading) % 360, flipping the y-component is (180 - heading) % 360
            # (kept positive with a + 360 offset), and a corner hit (both crossed) flips both, equivalent to a
            # 180-degree turn.
            .with_columns(**{
                heading_col: pl.when(pl.col("_crossed_x") & pl.col("_crossed_y"))
                                .then((pl.col(heading_col) + 180) % 360)
                             .when(pl.col("_crossed_x"))
                                .then((360 - pl.col(heading_col)) % 360)
                             .when(pl.col("_crossed_y"))
                                .then((540 - pl.col(heading_col)) % 360)
                             .otherwise(pl.col(heading_col))
            })
            .drop("_crossed_x", "_crossed_y")
        )

    # "none" and "absorbing": no coordinate/heading adjustment here.
    return df


def move_to(
        individuals: pl.DataFrame,
        landscape: pl.DataFrame,
        max_distance: int,
        turning_probability: float,
        home_range_sigma: float,
        boundary_condition: str
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
        turning_probability: A value between 0 and 1 indicating the probability by which a bembidion changes its
            current direction. Clockwise and counterclockwise turns always have the same probability of one half of
            the turning probability.
        home_range_sigma: Standard deviation, in meters, of the Gaussian home-range kernel around each individual's
            position at the start of the day (M3). Each candidate step is accepted or rejected with a probability
            that decays with its distance from that daily activity centre, following
            exp(-distance^2 / (2 * home_range_sigma^2)): a step right at the centre is always accepted on that
            criterion, while acceptance probability drops off the farther the step would take the individual from
            where it started the day. Smaller values keep individuals close to their daily starting point; larger
            values let them range more freely. This makes the daily activity region a "soft" disc around the day's
            starting position rather than a hard cutoff.
        boundary_condition: How to handle steps that would take an individual outside the extent of landscape (M4).
            One of "none" (status quo - such steps are simply inaccessible, which is what produces the unrealistic
            edge aggregation described in M4), "torus" (wrap around to the opposite edge), "reflecting" (bounce back
            off the edge, inverting the relevant heading component), or "absorbing" (remove the individual from the
            simulation the moment it would cross the edge).

    Returns:
        A Polars dataframe with the same schema as the individuals argument, but with updated coordinates and direction
        of bembidions.
    """
    # M4: the landscape's extent is needed for all boundary conditions ("torus" and "reflecting" need it to wrap or
    # mirror coordinates, "absorbing" needs it to detect when an individual leaves it, and "none" is unaffected).
    # Computed once per day rather than once per step, since the landscape does not change during a single day.
    x_min, x_max, y_min, y_max = (
        landscape.select(
            pl.col("x").min().alias("x_min"),
            pl.col("x").max().alias("x_max"),
            pl.col("y").min().alias("y_min"),
            pl.col("y").max().alias("y_max"),
        ).row(0)
    )

    # We have to initially prepare additiona state variables defining the movement of the individuals.
    individuals = (

        # Our starting point is the population of individuals passed to the function.
        individuals

        # The first thing a bembidion does in the ALMaSS code prior movement is determining how many steps it does. We
        # add this information by randomly drawing from a uniform distribution with max_distance as its upper bound.
        # Note: only the daily step budget is drawn once per day here. The turning decision itself (M1) is drawn
        # freshly for every single step inside the loop below, instead of once per day.
        .with_columns(

            # We add their desired movement distance in steps. The upper bound is exlcusive, so we have to add 1 to
            # include it as a possible sampling result.
            distance=polars_random.randint(1, max_distance + 1),

            # M2: the direction an individual starts moving in on a new day should not depend on the direction it
            # ended up with the previous day. We therefore overwrite the incoming direction with a freshly and
            # uniformly drawn heading out of the eight allowed values (0, 45, ..., 315) at the start of every day.
            # From here on, the per-step turning behaviour above (M1) takes over for the rest of the day.
            direction=polars_random.randint(high=8) * 45
        )

        # M3: we record each individual's position at the start of the day as its daily activity centre. All steps
        # taken during the day are then weighted by their distance from this centre, so movement stays concentrated
        # around a "home range" for the day instead of drifting arbitrarily far.
        .with_columns(
            center_x=pl.col("x"),
            center_y=pl.col("y")
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
            individuals_after_movement.append(
                movement_finished.drop("step_done", "distance", "center_x", "center_y")
            )

        # For individuals that still need to move this step, we add a synthetic row id. It is used below to group
        # several candidate headings evaluated for the same individual back together when picking an escape
        # direction after a failed movement attempt.
        if movement_required is not None:
            movement_required = movement_required.with_row_index("individual_id")

            # Before each movement step, the ALMaSS code checks whether a bembidion changes its direction, which has
            # a probability defined by turning_probability. The probability is symmetrical for clockwise and
            # counterclockwise turns. Unlike the original implementation, this decision is now made freshly for
            # every single step (M1) rather than once per day, so a bembidion can change heading mid-day.
            movement_required = (
                movement_required
                .with_columns(turning=polars_random.uniform())
                .with_columns(
                    direction=(
                        pl
                        .when(pl.col("turning") < turning_probability / 2).then(pl.col("direction") - 45)
                        .when(pl.col("turning") < turning_probability).then(pl.col("direction") + 45)
                        .otherwise("direction")
                    )
                )
                # After turning, it might be that directions are no longer encoded by the eight allowed values only.
                # This needs to be corrected.
                .with_columns(
                    direction=pl
                        .when(pl.col("direction") == -45).then(315)
                        .when(pl.col("direction") == 360).then(0)
                        .otherwise("direction")
                )
                .drop("turning")
            )

        # In this list, we collect the results of this step.
        step_results = []

        # Each individual first tries to keep moving in its current heading (Phase 1). Individuals for whom that
        # fails go through a vectorized search step (Phase 2) that evaluates all seven remaining headings at once
        # and picks uniformly at random among whichever of them succeed, instead of retrying with a random turn.
        if movement_required is not None:

            # --- Phase 1: try the individual's current heading. ---
            trial_candidates = (

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
            )

            # M4: "absorbing" removes individuals from the simulation the moment a step would take them outside the
            # landscape's extent - they never even attempt Phase 2's escape-direction search, they simply leave the
            # population for good. "torus" and "reflecting" instead adjust new_x/new_y (and, for "reflecting", the
            # heading) so the step stays within bounds. "none" leaves the candidates untouched, matching the status
            # quo where such a step is treated as inaccessible below (null accessibility after the landscape join).
            if boundary_condition == "absorbing":
                trial_candidates = trial_candidates.filter(
                    pl.col("new_x").is_between(x_min, x_max) & pl.col("new_y").is_between(y_min, y_max)
                )
            else:
                trial_candidates = _apply_boundary_condition(
                    trial_candidates, boundary_condition, x_min, x_max, y_min, y_max
                )

            trial_results = (
                trial_candidates

                # We can now learn about the accessibility of the target patch by joining the bembidion and the movement
                # map dataframes by coordinates. We need a left join so that we also keep bembidions that move outside
                # the defined landscape.
                .join(landscape, left_on=("new_x", "new_y"), right_on=("x", "y"), how="left")

                # We can now check whether the intended movement is accessible in principle.
                .with_columns(
                    accessible=(
                            pl.col("accessibility").eq("full") |
                            (pl.col("accessibility").eq("partial") & pl.col("access").lt(.4))
                    )
                )

                # M3: on top of accessibility, we weight the step by how far it would take the individual from its
                # daily activity centre. The acceptance probability follows a Gaussian kernel centred on
                # (center_x, center_y), so steps that stay close to the day's starting point are accepted almost
                # always, while steps far away are increasingly likely to be rejected.
                .with_columns(
                    home_range_distance=(
                        (pl.col("new_x") - pl.col("center_x")) ** 2 + (pl.col("new_y") - pl.col("center_y")) ** 2
                    ).sqrt(),
                    home_range_draw=polars_random.uniform()
                )
                .with_columns(
                    home_range_probability=(
                        -(pl.col("home_range_distance") ** 2) / (2 * home_range_sigma ** 2)
                    ).exp()
                )

                # A step only succeeds if it is both accessible and accepted by the home-range kernel. We use a
                # pl.when here (instead of a plain "&") to keep "accessible" being null (meaning: the step leaves
                # the defined landscape) as null in "success" too - a plain "&" would follow Kleene logic and could
                # incorrectly turn some of these into False, merging the "leaving the map" case into "unsuccessful".
                .with_columns(
                    success=pl.when(pl.col("accessible").is_null())
                        .then(None)
                        .otherwise(pl.col("accessible") & (pl.col("home_range_draw") < pl.col("home_range_probability")))
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
                    successful_trial.select(
                        x="new_x", y="new_y", direction="direction", distance="distance",
                        center_x="center_x", center_y="center_y"
                    )
                )

            # Individuals for whom Phase 1 failed (destination inaccessible, or outside the landscape) go through
            # Phase 2 below.
            blocked = [df for df in (unsuccessful_trial, leaving_map) if df is not None]
            blocked = pl.concat(blocked) if len(blocked) > 0 else None

            if blocked is not None:

                # --- Phase 2: Alle 7 Ausweichrichtungen gleichzeitig testen ---
                candidates = (
                    blocked
                    .select("individual_id", "x", "y", "direction", "distance", "center_x", "center_y")
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
                )

                # M4: the same "torus"/"reflecting" adjustment as in Phase 1, applied to the heading column here
                # ("heading" rather than "direction", since that is what carries each escape candidate's proposed
                # direction). "absorbing" and "none" both need no special handling here: a candidate that steps
                # outside the landscape is simply marked inaccessible below (fill_null(False)) and therefore never
                # gets chosen as an escape route - functionally the same as excluding it as a candidate outright.
                candidates = _apply_boundary_condition(
                    candidates, boundary_condition, x_min, x_max, y_min, y_max, heading_col="heading"
                )

                candidates = (
                    candidates
                    .join(landscape, left_on=("new_x", "new_y"), right_on=("x", "y"), how="left")
                    .with_columns(
                        accessible=(
                            pl.col("accessibility").eq("full") |
                            (pl.col("accessibility").eq("partial") & pl.col("access").lt(.4))
                        ).fill_null(False)  # Verhindert Null-Werte am Kartenrand
                    )

                    # M3: same Gaussian home-range weighting as in Phase 1, so an escape heading that would carry the
                    # individual far from its daily activity centre is less likely to be picked than one that keeps
                    # it close by. No null-handling needed here since "accessible" is already fill_null'd to False.
                    .with_columns(
                        home_range_distance=(
                            (pl.col("new_x") - pl.col("center_x")) ** 2 + (pl.col("new_y") - pl.col("center_y")) ** 2
                        ).sqrt(),
                        home_range_draw=polars_random.uniform()
                    )
                    .with_columns(
                        home_range_probability=(
                            -(pl.col("home_range_distance") ** 2) / (2 * home_range_sigma ** 2)
                        ).exp()
                    )
                    .with_columns(
                        success=pl.col("accessible") & (pl.col("home_range_draw") < pl.col("home_range_probability"))
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
                        chosen.select(
                            x="new_x", y="new_y", direction="heading", distance="distance",
                            center_x="center_x", center_y="center_y"
                        )
                    )
                    # Blockierte Käfer ermitteln, die KEINE Ausweichrichtung gefunden haben:
                    stuck = blocked.join(chosen.select("individual_id"), on="individual_id", how="anti")
                else:
                    # Niemand konnte ausweichen -> alle Blockierten stecken fest
                    stuck = blocked
                if not stuck.is_empty():
                    step_results.append(
                        stuck.select("x", "y", "direction", "distance", "center_x", "center_y")
                    )

        # Set the individuals for the next step.
        if len(step_results) > 0:
            individuals = pl.concat(step_results)
        else:
            individuals = None

        # Add the last batch of individuals to the result.
        if i == max_distance - 1 and individuals is not None:
            individuals_after_movement.append(individuals.drop("distance", "center_x", "center_y"))

    # Return the movement results
    return pl.concat(individuals_after_movement)


# All result CSVs are written into a dedicated subfolder of "data" instead of directly into "data" itself. This way,
# different runs can be executed one after another without overwriting each other's output. Just change run_name
# before a run and every write_csv call below will land in its own folder.
run_name = "run3"
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
# probability for each bembidion to change its direction of the previous day (80% in ALMaSS), the upper bound for
# the movement distance in steps (14 in ALMaSS), and the home-range sigma (M3) that constrains movement to stay
# around the day's starting point.
print(move_to(bembidions, movement_map, 14, .8, HOME_RANGE_SIGMA, BOUNDARY_CONDITION))

# And we can simulate an entire year of movement.
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions = move_to(bembidions, movement_map, 14, .8, HOME_RANGE_SIGMA, BOUNDARY_CONDITION)
bembidions.write_csv(output_dir / "locations_with_directions_after_one_year.csv")

# The same for a small sub-landscape
movement_map2 = movement_map.filter(pl.col("x") <= 319977).filter(pl.col("y") <= 5702396)
bembidions2 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1_000, shuffle=True)
bembidions2 = bembidions2.with_columns(direction=polars_random.randint(high=8) * 45)
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions2 = move_to(bembidions2, movement_map2, 14, .8, HOME_RANGE_SIGMA, BOUNDARY_CONDITION)
bembidions2.write_csv(output_dir / "locations_with_directions_after_one_year_small.csv")

# And for ten years
bembidions2 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1_000, shuffle=True)
bembidions2 = bembidions2.with_columns(direction=polars_random.randint(high=8) * 45)
for day in range(3650):
    print(f"*** DAY {day + 1} ***")
    bembidions2 = move_to(bembidions2, movement_map2, 14, .8, HOME_RANGE_SIGMA, BOUNDARY_CONDITION)
bembidions2.write_csv(output_dir / "locations_with_directions_after_ten_years_small.csv")

# here, we keep track of the movement of a single individual to study its behavior
bembidions3 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1, shuffle=True)
bembidions3 = bembidions3.with_columns(direction=polars_random.randint(high=8) * 45)
results = []
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions3 = move_to(bembidions3, movement_map2, 14, .8, HOME_RANGE_SIGMA, BOUNDARY_CONDITION)
    results.append(bembidions3.with_columns(t=day))
pl.concat(results).write_csv(output_dir / "movement_one_year_small.csv")
