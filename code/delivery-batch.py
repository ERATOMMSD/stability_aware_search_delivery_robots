#!/usr/bin/env python
# coding: utf-8
import json
import os
import logging as log
from deliverysearch.problem import RobotQuantityProblem, RobotScheduleAssignmentProblem, WeightedProblem
from simulator import Simulator, MockSimulator
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option('--mock', action="store_true", dest="mock", default=False,
                      help="it uses the mock simulator for testing purposes")
    parser.add_option('--speed', action="store", type="int", dest="speed", default=10,
                      help="maximum speed")
    parser.add_option('--max-requests', action="store", type="int", dest="max_requests", default=100,
                      help="minimum number of requests per hour to be received")
    parser.add_option('--min-requests', action="store", type="int", dest="min_requests", default=20,
                      help="maximum number of requests per hour to be received")
    parser.add_option('--capacity', action="store", type="int", dest="capacity", default=5,
                      help="number of packages that a robot can load")
    parser.add_option('--duration', action="store", type="int", dest="duration", default=3,
                      help="duration of the simulation")
    parser.add_option('--start-time', action="store", type="int", dest="start_time", default=9,
                      help="time when the simulation starts in hours (e.g, 9 means 9:00)")
    parser.add_option('--threads', action="store", type="int", dest="threads", default=None,
                      help="number of threads to be used (by default it uses the total number of cores")
    parser.add_option('--seed', action="store", type="int", dest="seed", default=59713643048,
                      help="seed that is used to initialize both the simulator and the search")
    parser.add_option('--jmetal-seed', action="store", type="int", dest="jmetal_seed", default=73912398701,
                      help="seed that is used to initialize both the simulator and the search")
    parser.add_option('--problem', action="store", type="str", dest="problem", default="robot-quantity",
                      help="It can be 'robot-quantity' or 'schedule-assignment'")
    parser.add_option('--robots', action="store", type="int", dest="robots", default=10,
                      help="maximum number of robots")
    parser.add_option('--simulations', action="store", type="int", dest="simulations", default=5,
                      help="number of samples in which an individual is evaluated")
    parser.add_option('--results-dir', action="store", type="str", dest="result_dir", default="batch",
                      help="name of folder in which results are stored")
    parser.add_option('--low-bound', action="store", type="int", dest="low_bound", default=1,
                      help="lower bound for the search (both robots and speed)")
    parser.add_option('--weighted', action="store_true", dest="weighted", default=False,
                      help="If weighted, it uses a single objective search.")

    options, args = parser.parse_args()
    log.getLogger().setLevel(log.INFO)

    if options.mock:
        Sim = MockSimulator
    else:
        Sim = Simulator

    results_dir = f'{options.result_dir}/{options.problem}/{options.min_requests}_{options.max_requests}'

    os.makedirs(results_dir, exist_ok=True)

    log.info("Batch run!")
    log.info(f"Problem: {options.problem}")
    log.info(f"Robots capacity: {options.capacity}")
    log.info(f"Customer requests per hour: {options.min_requests} - {options.max_requests}")
    log.info(f"Max simulation duration (hours): {options.duration}")
    log.info(f"Results folder: {results_dir}")
    log.info(f"Robots: {options.robots}")
    log.info(f"Speed: {options.speed}")

    log.info("Starting simulation in 5 seconds...")

    sim = Sim(min_customer_requests_per_hour=options.min_requests,
              max_customer_requests_per_hour=options.max_requests,
              robot_loading_capacity=options.capacity,
              simulation_duration=options.duration,
              hour=options.start_time,
              number_of_simulations=options.simulations,
              threads=options.threads,
              utilization_time_period=None,  # They can be specified in the following format: [(9, 12)]
              seed=options.seed,
              results_folder=f'{options.problem}/batch/reqs_{options.min_requests}_{options.max_requests}/{options.robots}_{options.speed}')

    if options.problem == 'robot-quantity':
        problem = RobotQuantityProblem(sim=sim,
                                       speed_bounds=(options.low_bound, options.speed),
                                       robot_bounds=(options.low_bound, options.robots))
    elif options.problem == 'schedule-assignment':
        end_time = options.start_time + options.duration
        problem = RobotScheduleAssignmentProblem(sim=sim,
                                                 number_of_robots=options.robots,
                                                 speed_bounds=(options.low_bound, options.speed),
                                                 business_hours=(options.start_time, end_time))

        raise RuntimeError(f"This is not implemented for {options.problem}")

    else:
        raise RuntimeError(f"{options.problem} is not a valid problem input")

    if options.weighted == 'weighted-classic':
        problem = WeightedProblem(problem=problem, weights=[1000.0, 100.0, 1.0])

    solution = problem.create_solution()

    solution.variables = [options.robots, options.speed]

    solution = problem.evaluate(solution)

    with open(f'{results_dir}/r{options.robots}_s{options.speed}.json', 'w') as outfile:
        json.dump(problem.records[0], outfile)
