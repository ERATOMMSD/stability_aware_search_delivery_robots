from abc import ABC
from simulator import Simulator


class DeliveryObjective(ABC):

    MAXIMIZE = 1
    MINIMIZE = -1

    def __init__(self, label: str, direction: int):
        self.label = label
        self.direction = direction


class MaximizeDelivery(DeliveryObjective):
    def __init__(self):
        super().__init__(Simulator.NUM_DELIVERED, DeliveryObjective.MAXIMIZE)


class MaximizeDeliveryRate(DeliveryObjective):
    def __init__(self):
        super().__init__(Simulator.DELIVERY_RATE, DeliveryObjective.MAXIMIZE)


class MaximizeUtilization(DeliveryObjective):
    def __init__(self):
        super().__init__(Simulator.UTILIZATION_RATE, DeliveryObjective.MAXIMIZE)


class MinimizeRisks(DeliveryObjective):
    def __init__(self):
        super().__init__(Simulator.NUM_RISKS, DeliveryObjective.MINIMIZE)

class MinimizeDeliveryRateStd(DeliveryObjective):
    def __init__(self):
        super().__init__(Simulator.DELIVERY_RATE_STD, DeliveryObjective.MINIMIZE)


class MinimizeUtilizationStd(DeliveryObjective):
    def __init__(self):
        super().__init__(Simulator.UTILIZATION_RATE_STD, DeliveryObjective.MINIMIZE)


class MinimizeRisksStd(DeliveryObjective):
    def __init__(self):
        super().__init__(Simulator.NUM_RISKS_STD, DeliveryObjective.MINIMIZE)
