
class log:

    """
    Transparent interface
    Used for tagging important elements in code
    Processed by the parser
    """

    @staticmethod
    def read(path):
        """
        Input reference
        """
        return path

    @staticmethod
    def write(path):
        """
        Output reference
        """
        return path

    @staticmethod
    def parameter(value):
        """
        Input value
        """
        return value

    @staticmethod
    def metric(value):
        """
        Output value
        """
        return value
