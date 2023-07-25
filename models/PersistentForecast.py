import numpy as np


class PersistentForecast:
    def __init__(self, num_classes: int):
        """
        Persistent Forecast, output the recently observed node label for the node.
        :param num_classes: int, number of label classes
        """
        self.memory = {}
        self.num_classes = num_classes

    def update_memory(self, node_id: int, node_label: np.ndarray):
        """
        update the memory of node
        :param node_id: int, node id
        :param node_label: ndarray, shape (num_classes, )
        :return:
        """
        self.memory[node_id] = node_label

    def get_memory(self, node_id: int):
        """
        get the memory of node
        :param node_id: int, node id
        :return:
        """
        # if the memory does not exist, return zero vector
        return self.memory[node_id] if node_id in self.memory else np.zeros(self.num_classes)
