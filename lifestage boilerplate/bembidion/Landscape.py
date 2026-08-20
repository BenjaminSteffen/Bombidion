import bembidion
import random


class Landscape:
    def __init__(self):
        self.population = bembidion.Population(self)
        self.timestep = -1
        self.temperature = None

    def step(self):
        self.timestep += 1
        self.temperature = random.randint(-20, 40)
        self.population.step()
