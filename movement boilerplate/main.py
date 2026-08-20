"""
This file is intended as a starting point to test implementations of bembidion movement. The initial implementation is
reflecting the now outdated code of the ALMaSS bembidion available under
https://gitlab.com/ALMaSS/ALMaSS_stable/-/blob/master/Bembidion/Bembidion_all.cpp. However, instead of initializing
each bembidion as an object, applies expressions to a population of bembidions represented in a polars data frame.
"""
import polars_random
import polars as pl


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
            the current x-coordinate of the individual, a column y with its current y-coordinate and a column direction
            with the last movement direction of the individual. Latter can be one of eight values: 0, 45, 90, 135, 180,
            225, 270, or 315.
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

            # We draw a random number between 0 and 1 that represents the random turning behavior of each bembidion.
            turning=polars_random.uniform()
        )

        # Before movement, the ALMaSS code checks whether a bembidion changes its direction, which has a probability
        # defined by the turning_probability. The probability is symmetrical for clockwise and counterclockwise turns.
        # We do the same here.
        .with_columns(
            direction=(
                pl
                .when(pl.col("turning") < turning_probability / 2).then(pl.col("direction") - 45)
                .when(pl.col("turning") < turning_probability).then(pl.col("direction") + 45)
                .otherwise("direction")
            )
        )

        # After turning, it might be that directions are no longer encoded by the eight allowed values only. This needs
        # to be corrected.
        .with_columns(
            direction=pl.when(direction=-45).then(315).when(direction=360).then(0).otherwise("direction")
        )

        # The turning column is no longer needed.
        .drop("turning")
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

        # In this list, we collect the results of this step.
        step_results = []

        # All other individuals try to move (up to ten times)
        tries = 0
        while movement_required is not None and tries < 10:
            # Increase the trial-counter and give some output to facilitate development.
            tries += 1
            print(f"Trial #{tries} ({len(movement_required)} individual(s))...")

            # Let the individuals that want to move try to move. Partition them by the result of their movement trial.
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

            # If there are no individuals with other needs, set the list of individuals with required movements to None.
            if unsuccessful_trial is None and leaving_map is None:
                movement_required = None

            # Else start to collect bembidions still requiring movement.
            else:
                movement_required = []

                # In this implementation, treat world boundaries as inaccessible patches.
                for case_to_handle in (unsuccessful_trial, leaving_map):

                    # These are the bembidions that could not move because their destination was inaccessible.
                    if case_to_handle is not None:
                        # Alter their direction and add them to the list of individuals still requiring movement.
                        movement_required.append(
                            # Consider all individuals with unsuccessful movement.
                            case_to_handle

                            # Remove unnecessary columns and add a random strategy (turn clockwise or counterclockwise)
                            # for the next trial.
                            .select("x", "y", "direction", "distance", turning=polars_random.randint())

                            # Modify the direction according to the individual's strategy.
                            .select(
                                "x",
                                "y",
                                direction=pl
                                .when(turning=0).then(pl.col("direction") - 45)
                                .otherwise(pl.col("direction") + 45),
                                distance="distance"
                            )

                            # After turning, it might be that directions are no longer encoded by the eight allowed
                            # values only. This needs to be corrected.
                            .with_columns(
                                direction=pl
                                .when(direction=-45).then(315)
                                .when(direction=360).then(0)
                                .otherwise("direction")
                            )
                        )

                # Collect all individuals requiring movement.
                movement_required = pl.concat(movement_required)

        # If, after 10 tries, individuals still require movement, skip this step for them.
        if movement_required is not None:
            step_results.append(movement_required)

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
bembidions.write_csv("data/example_start_locations.csv")

# According to the ALMaSS code, each bembidion also has to remember its last movement directions, latter which can only
# have one of eight values. We will assign directions to the initial bembidions randomly.
bembidions = bembidions.with_columns(direction=polars_random.randint(high=8) * 45)

# Once again, this is nice to visualize on a map. See "example start locations with directions.png".
bembidions.write_csv("data/example_start_locations_with_directions.csv")

# We can now simulate a single day's movement of all bembidions using our movement implementation. We have to pass a
# probability for each bembidion to change its direction of the previous day (80% in ALMaSS) and the upper bound for
# the movement distance in steps (14 in ALMaSS).
print(move_to(bembidions, movement_map, 14, .8))

# And we can simulate an entire year of movement.
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions = move_to(bembidions, movement_map, 14, .8)
bembidions.write_csv("data/locations_with_directions_after_one_year.csv")

# The same for a small sub-landscape
movement_map2 = movement_map.filter(pl.col("x") <= 319977).filter(pl.col("y") <= 5702396)
bembidions2 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1_000, shuffle=True)
bembidions2 = bembidions2.with_columns(direction=polars_random.randint(high=8) * 45)
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions2 = move_to(bembidions2, movement_map2, 14, .8)
bembidions2.write_csv("data/locations_with_directions_after_one_year.csv")

# And for ten years
bembidions2 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1_000, shuffle=True)
bembidions2 = bembidions2.with_columns(direction=polars_random.randint(high=8) * 45)
for day in range(3650):
    print(f"*** DAY {day + 1} ***")
    bembidions2 = move_to(bembidions2, movement_map2, 14, .8)
bembidions2.write_csv("data/locations_with_directions_after_ten_years_small.csv")

# here, we keep track of the movement of a single individual to study its behavior
bembidions3 = movement_map2.filter(pl.col("accessibility").ne("none")).select("x", "y").sample(1, shuffle=True)
bembidions3 = bembidions3.with_columns(direction=polars_random.randint(high=8) * 45)
results = []
for day in range(365):
    print(f"*** DAY {day + 1} ***")
    bembidions3 = move_to(bembidions3, movement_map2, 14, .8)
    results.append(bembidions3.with_columns(t=day))
pl.concat(results).write_csv("data/movement_one_year_small.csv")