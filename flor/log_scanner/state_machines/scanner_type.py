
class ScannerType:

    def __init__(self, name, file_path, class_ctx, func_ctx, prev_lsn,):
        self.name = name
        self.file_path = file_path
        self.class_ctx = class_ctx
        self.func_ctx = func_ctx
        self.prev_lsn = prev_lsn
        self.follow_lsn = None

        self.prev_lsn_enabled = False

    def set_follow_lsn(self, lsn):
        self.follow_lsn = lsn

    def consume_lsn(self, log_record, trailing_ctx):
        if trailing_ctx is not None and trailing_ctx.is_enabled(self):
            if log_record['lsn'] == self.prev_lsn:
                self.prev_lsn_enabled = True

    def consume_func_name(self, log_record, trailing_ctx):
        raise NotImplementedError()

    def consume_data(self, log_record, trailing_ctx):
        raise NotImplementedError()