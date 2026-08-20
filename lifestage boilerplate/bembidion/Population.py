import polars_random
import polars as pl


class Population:
    def __init__(self, landscape):
        self.landscape = landscape
        initial_number_adults = 1_000
        self.individuals = (
            pl.DataFrame(
                {
                    "lifestage": ["Adult"] * initial_number_adults,
                    "state": (
                        pl.DataFrame({"state": ["Foraging", "Aggregating", "Hibernating", "Dispersing", "Dying"]})
                        .sample(initial_number_adults, shuffle=True, with_replacement=True)
                    )
                }
            )
            .with_columns(
                x=polars_random.randint(-500, 501),
                y=polars_random.randint(-500, 501),
                direction=polars_random.randint(high=8) * 45,
            )
        )

    def step(self):
        self.individuals = pl.concat(
            [
                getattr(self, f"{activity[0].lower()}_{activity[1].lower()}")(individuals)
                for activity, individuals in self.individuals.partition_by("lifestage", "state", as_dict=True).items()
            ]
        )

    def adult_foraging(self, individuals):
        # todo
        return self.daily_movement(individuals, True)

    def adult_aggregating(self, individuals):
        # todo
        return individuals

    def adult_hibernating(self, individuals):
        # todo
        return individuals

    def adult_dispersing(self, individuals):
        # todo
        return individuals

    def adult_dying(self, individuals):
        return individuals[:0]

    def daily_movement(self, individuals, dispersing):
        # todo
        if self.landscape.temperature < 1:
            return individuals
        if dispersing:
            return self.move_to_dispersing(individuals)
        return self.move_to_aggregating(individuals)

    def move_to_dispersing(self, individuals):
        tmp = (
            individuals
            .with_columns(
                distance=polars_random.randint(1, 15),
                turning=polars_random.uniform()
            )
            .with_columns(
                direction=(
                    pl
                    .when(pl.col("turning") < .4).then(pl.col("direction") - 45)
                    .when(pl.col("turning") > .4).then(pl.col("direction") + 45)
                    .otherwise("direction")
                )
            )
            .with_columns(
                pl
                .when(pl.col("direction") < 0).then(-pl.col("direction"))
                .when(pl.col("turning") > 315).then(pl.col("direction") - 360)
                .otherwise("direction")
            )
        )





        return individuals

    def move_to_aggregating(self, individuals):
        # todo
        return individuals