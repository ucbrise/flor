import cloudpickle

class Ctx:

    def __init__(self):
        self.file_path = None
        self.class_ctx = None
        self.func_ctx = None

    def is_enabled(self, obj):
        return (self.file_path == obj.file_path
                and self.class_ctx == obj.class_ctx
                and (self.func_ctx == obj.func_ctx
                     or self.func_ctx == '__init__'))


class ActualParam:
    """
    This class will act like a finite state machine
    Every log record in the log is an input
    """

    def __init__(self, file_path, class_ctx, func_ctx, prev_lsn, follow_lsn, func_name, pos_kw):
        """

        :param file_path: file path of pre-annotated source where the highlight appears
        :param class_ctx: ...
        :param func_ctx: the function or root context (e.g.  None) where this highlight appears.
                    a.k.a. the context of the prev_lsn
        :param prev_lsn: The lsn of the log statement that statically immediately precedes the highlight
                            the purpose is to narrow the scope of search and limit possibility of ambiguities
        :param follow_lsn: ...
        :param func_name: The name of the function that the highlighted actual param is passed to
        :param pos_kw: {'pos': int} or {'kw': str} ... To resolve the value of the parameter
        """
        self.file_path = file_path
        self.class_ctx = class_ctx
        self.func_ctx = func_ctx
        self.prev_lsn = prev_lsn
        self.follow_lsn = follow_lsn
        self.func_name = func_name
        self.pos_kw = pos_kw

        # State
        self.contexts = []
        self.func_enabled = False
        self.prev_lsn_enabled = False

        # Outputs
        self.collected = []

    def transition(self, log_record):
        """

        :param log_record: dict from log
        :return:
        """
        if 'file_path' in log_record:
            ctx = Ctx()
            ctx.file_path = log_record['file_path']
            self.contexts.append(ctx)
        elif 'class_name' in log_record:
            self.contexts[-1].class_ctx = log_record['class_name']
        elif 'start_function' in log_record:
            self.contexts[-1].func_ctx = log_record['start_function']
            if log_record['start_function'] == self.func_name: self.func_enabled = True
            elif log_record['start_function'] == '__init__' and self.contexts[-1].class_ctx == self.func_name:
                self.func_enabled = True
        elif 'end_function' in log_record:
            if log_record['end_function'] == self.func_name: self.func_enabled = False
            elif log_record['end_function'] == '__init__' and self.contexts[-1].class_ctx == self.func_name:
                self.func_enabled = False
            self.contexts.pop()

        else:

            #TODO: must resolve train_test_split(iris.data, iris.target, test_size=0.15, random_state=430) --> train_test_split(*arrays, **options)
            #TODO: Logger must be more robust
            # data log record
            if self.contexts and self.contexts[-1].is_enabled(self):
                if log_record['lsn'] == self.prev_lsn:
                    self.prev_lsn_enabled = True
                elif log_record['lsn'] == self.follow_lsn:
                    self.prev_lsn_enabled = False
            if (
                len(self.contexts) >= 2
                and self.contexts[-2].is_enabled(self)
            ) and self.prev_lsn_enabled and self.func_enabled:
                # active only means search
                if 'params' in log_record:
                    params = []
                    unfolded_idx = 0
                    for param in log_record['params']:
                        # param is a singleton dict. k -> value
                        k = list(param.keys()).pop()
                        idx, typ, name = k.split('.')
                        if typ == 'raw':
                            params.append({k: cloudpickle.loads(eval(param[k]))})
                            unfolded_idx += 1
                        elif typ == 'vararg':
                            list_of_params = cloudpickle.loads(eval(param[k]))
                            for each in list_of_params:
                                new_key = '.'.join((str(unfolded_idx),'raw','$'))
                                params.append({new_key: each})
                                unfolded_idx += 1
                        elif typ == 'kwarg':
                            dict_of_params = cloudpickle.loads(eval(param[k]))
                            for k in dict_of_params:
                                new_key = '.'.join(('$','raw',k))
                                params.append({new_key: dict_of_params[k]})
                        else:
                            raise RuntimeError()


                    #params: List[Singleton_Dict[(idx, 'raw', kw_name), deserialized_val]]


                    if 'pos' in self.pos_kw:

                        for single_dict in params:
                            k = list(single_dict.keys()).pop()
                            if int(k.split('.')[0]) == self.pos_kw['pos']:
                                self.collected.append(single_dict)
                                break
                    else:
                        assert 'kw' in self.pos_kw
                        for single_dict in params:
                            k = list(single_dict.keys()).pop()
                            if k.split('.')[2] == self.pos_kw['kw']:
                                self.collected.append(single_dict)
                    try:
                        print("It's a match: {}".format(self.collected[-1]))
                    except:
                        print("ERROR ... log record: {}".format(params))


