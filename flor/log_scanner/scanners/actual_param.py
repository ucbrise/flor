
class ActualParam:
    """
    This class will act like a finite state machine
    Every log record in the log is an input
    """

    def __init__(self, file_path, ctx, prev_lsn, func_name, pos_kw):
        """

        :param file_path: file path of pre-annotated source where the highlight appears
        :param ctx: the function or root context where this highlight appears.
                    a.k.a. the context of the prev_lsn
        :param prev_lsn: The lsn of the log statement that statically immediately precedes the highlight
                            the purpose is to narrow the scope of search and limit possibility of ambiguities
        :param func_name: The name of the function that the highlighted actual param is passed to
        :param pos_kw: {'pos': int} or {'kw': str} ... To resolve the value of the parameter
        """
        self.file_path = file_path
        self.ctx = ctx
        self.prev_lsn = prev_lsn
        self.func_name = func_name
        self.pos_kw = pos_kw

        # State
        self.active = False

    def transition(self, log_record):
        """

        :param log_record: dict from log
        :return:
        """
        if 'file_path' in log_record:
            pass
