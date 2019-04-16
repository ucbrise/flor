
class ScannerType:

    def consume_func_name(self, log_record, trailing_ctx, contexts):
        raise NotImplementedError()

    def consume_data(self, log_record, trailing_ctx, contexts):
        raise NotImplementedError()