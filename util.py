import numpy as np


class Util:
    @staticmethod
    def train_multiple(train_func, num_iterations, **kwargs):
        """


        :param train_func: A function that returns a scalar or an iterable of scalars
        :param num_iterations: Number of iterations to run the train_func for
        :param kwargs: Keyword arguments which will be passed onto the train_function
        :return: Averages of the return values of train_func
        """
        results = []
        for i in range(num_iterations):
            results.append(train_func(**kwargs))

        averages = np.mean(np.array(results), axis=0)
        return averages


