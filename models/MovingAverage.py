import numpy as np


class MovingAverage:
    def __init__(self, num_classes: int, window_size: int = 7):
        """
        Moving Average, output the average of node labels observed in the previous window_size steps.
        :param num_classes: int, number of label classes
        :param window_size: int, window size
        """
        self.memory = {}
        self.num_classes = num_classes
        self.window_size = window_size

    def update_memory(self, node_id: int, node_label: np.ndarray):
        """
        update the memory of node
        :param node_id: int, node id
        :param node_label: ndarray, shape (num_classes, )
        :return:
        """
        if node_id in self.memory:
            self.memory[node_id] = (self.memory[node_id] * (self.window_size - 1) + node_label) / self.window_size
        else:
            self.memory[node_id] = node_label

    def get_memory(self, node_id: int):
        """
        get the memory of node
        :param node_id: int, node id
        :return:
        """
        # if the memory does not exist, return zero vector
        return self.memory[node_id] if node_id in self.memory else np.zeros(self.num_classes)
