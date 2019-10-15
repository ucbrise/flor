class Ctx:

    def __init__(self):
        self.file_path = None
        self.class_ctx = None
        self.func_ctx = None

        self.parent_ctx = None

    def is_enabled(self, obj):
        """
        Checks whether state machine obj is enabled in this context
        :obj: a state machine
        """
        return (self.file_path == obj.file_path
                and self.class_ctx == obj.class_ctx
                and (self.func_ctx == obj.func_ctx
                     or self.func_ctx == '__init__'))

    def __eq__(self, other):
        return (other is not None and self.file_path == other.file_path
        and self.class_ctx == other.class_ctx
        and self.func_ctx == other.func_ctx
        and self.parent_ctx == other.parent_ctx)

    def __str__(self):
        return "{}.{}.{}".format(self.file_path, self.class_ctx, self.func_ctx)