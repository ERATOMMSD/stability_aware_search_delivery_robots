import os

from jmetal.lab.experiment import Job
from jmetal.core.algorithm import EvolutionaryAlgorithm
from typing import Dict
import random
import json

from deliverysearch.problem import DeliveryProblem


class DeliveryJob(Job):

    def __init__(self, algorithm: EvolutionaryAlgorithm, run_tag: int, seed: int, config: Dict):
        self.seed = seed
        self.config = config
        super().__init__(algorithm=algorithm, algorithm_tag=algorithm.get_name(),
                         problem_tag=algorithm.problem.get_name(), run=run_tag)

    def execute(self, output_path: str = ''):
        random.seed(self.seed)
        super().execute(output_path)
        if output_path:
            file_name = os.path.join(output_path, 'IND.{}.csv'.format(self.run_tag))

            # Abusing typing. The algorithm should be more specific than EvolutionaryAlgorithm to have a DeliveryProblem
            problem: DeliveryProblem = self.algorithm.problem
            problem.get_individuals_data_frame().to_csv(file_name)

            file_name = os.path.join(output_path, 'CONFIG.{}'.format(self.run_tag))
            with open(file_name, 'w+') as of:
                json.dump(self.config, of)

            file_name = os.path.join(output_path, 'REC_IND.{}.csv'.format(self.run_tag))
            problem.fill_missing_data()
            problem.get_individuals_data_frame().to_csv(file_name)
