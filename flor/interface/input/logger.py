
class log:

    """
    Transparent interface used for tagging important elements in code and processed by the parser.
    Every log statement must appear inside a flor-tracked function.
    """

    @staticmethod
    def read(path):
        """
        Input reference

        Log the event of reading a file located at `path`.
        Usage: https://github.com/ucbrise/flor/blob/master/examples/logger/basic.py

        :param path: The location of the file to be read
        :return: `path`
        """
        return path

    @staticmethod
    def write(path):
        """
        Output reference

        Log the event of writing a file to `path`.
        Usage: https://github.com/ucbrise/flor/blob/master/examples/logger/basic.py

        :param path: The destination of the file to be written
        :return: `path`
        """
        return path

    @staticmethod
    def param(value):
        """
        Input value

        Log a parameter's value and its surrounding context.
        Usage: https://github.com/ucbrise/flor/blob/master/examples/logger/basic.py

        :param value: The value of the parameter to be logged
        :return: `value`
        """
        return value

    @staticmethod
    def metric(value):
        """
        Output value

        Log a metric's value and its surrounding context.
        Usage: https://github.com/ucbrise/flor/blob/master/examples/logger/basic.py

        :param value: The value of the metric to be logged
        :return: `value`
        """
        return value

