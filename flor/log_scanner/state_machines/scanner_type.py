
class ScannerType:

    def __init__(self):
        self.follow_lsn = None

    def set_follow_lsn(self, lsn):
        self.follow_lsn = lsn

    def consume_func_name(self, log_record, trailing_ctx, contexts):
        raise NotImplementedError()

    def consume_data(self, log_record, trailing_ctx, contexts):
        raise NotImplementedError()