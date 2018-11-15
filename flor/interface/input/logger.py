
class log:

    """
    Transparent interface
    Used for tagging important elements in code
    Processed by the parser
    """

    @staticmethod
    def read(path, **kwargs):
        """
        Input reference
        """
        return path

    @staticmethod
    def write(path, **kwargs):
        """
        Output reference
        """
        return path

    @staticmethod
    def param(value, **kwargs):
        """
        Input value
        """
        return value

    @staticmethod
    def metric(value, **kwargs):
        """
        Output value
        """
        return value

