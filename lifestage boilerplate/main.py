import bembidion
import polars as pl


# Create initial adults in different states
landscape = bembidion.Landscape()


# Do ten timesteps
for _ in range(10):
    landscape.step()
    print(landscape.population.individuals)
