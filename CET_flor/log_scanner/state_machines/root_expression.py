import pickle as cloudpickle

from .scanner_type import ScannerType

class RootExpression(ScannerType):
    """
    Collects from an assign context.
    Example

    x = FLOR_METRIC('x', e ) ... where e is an arbitrary root expression

    """

    def __init__(self, name, file_path, class_ctx, func_ctx, prev_lsn, tuple_idx):
        """

        :param file_path: file path of pre-annotated source where the highlight appears
        :param class_ctx: if func_ctx is a method, class to which the method belongs
        :param func_ctx: the function or root context (e.g.  None) where this highlight appears.
                    a.k.a. the context of the prev_lsn
        :param prev_lsn: The lsn of the log statement that statically immediately precedes the highlight
                            the purpose is to narrow the scope of search and limit possibility of ambiguities
        :param follow_lsn: The lsn from which we want to extract data
        """
        super().__init__(name, file_path, class_ctx, func_ctx, prev_lsn)
        self.tuple_idx = tuple_idx

        #State


    def consume_data(self, log_record, trailing_ctx):
        if trailing_ctx is not None and trailing_ctx.is_enabled(self):
            if log_record['lsn'] == self.prev_lsn:
                self.prev_lsn_enabled = True
            elif self.prev_lsn_enabled and log_record['lsn'] == self.follow_lsn:
                # Consume data from an assign context
                root_expressions = list(map(lambda d: {list(d.keys()).pop():
                                                       cloudpickle.loads(eval(list(d.values()).pop()))} ,
                                            log_record['locals']))
                self.prev_lsn_enabled = False
                if self.tuple_idx is None:
                    assert len(root_expressions) == 1
                    return root_expressions.pop()
                else:
                    return root_expressions[self.tuple_idx]
