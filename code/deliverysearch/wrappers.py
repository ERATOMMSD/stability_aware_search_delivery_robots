import math
import logging as log
from jmetal.core.algorithm import Algorithm
from simulator import Simulator


class AlgorithmWrapper:

    def __init__(self, algorithm: Algorithm, simulator: Simulator, generations: int):
        self.algorithm = algorithm
        self.original_step = self.algorithm.step
        self.algorithm.__setattr__('step', self.step)
        self.simulator = simulator
        self.generation = 0
        self.generations = generations

    def step(self):
        self.generation += 1
        self.algorithm.problem.update_generation(generation=self.generation)
        return self.original_step()

    def __getattr__(self, called_method):
        if called_method != 'step':
            return getattr(self.algorithm, called_method)
        else:
            return self.step()


class IncrementalSimulationsWrapper(AlgorithmWrapper):

    def __init__(self, algorithm: Algorithm, simulator: Simulator, generations: int, max_simulations: int):
        self.max_simulations = max_simulations
        super(IncrementalSimulationsWrapper, self).__init__(algorithm, simulator, generations)
        self.simulator.set_number_of_simulations(1)

    def update_simulations(self):
        simulations = math.ceil(self.generation / self.generations * self.max_simulations)
        log.getLogger().info(f'Starting generation {self.generation} with {simulations} simulations.')
        self.simulator.set_number_of_simulations(simulations)

    def step(self):
        res = super().step()
        self.update_simulations()
        return res


class IncrementalTimeWrapper(AlgorithmWrapper):

    def __init__(self, algorithm: Algorithm, simulator: Simulator, generations: int, max_time: int, min_time: int = 1):
        self.max_time = max_time
        self.min_time = min_time
        super(IncrementalTimeWrapper, self).__init__(algorithm, simulator, generations)
        self.update_simulation_duration()

    def update_simulation_duration(self):
        simulation_duration = max(self.min_time, math.ceil(self.generation / self.generations * self.max_time))
        log.getLogger().info(f'Starting generation {self.generation} with {simulation_duration} hours simulation.')
        self.simulator.update_simulation_duration(simulation_duration)

    def step(self):
        res = super().step()
        self.update_simulation_duration()
        return res
