import ast
from flor.complete_capture.trace_generator import ClientRoot


class Header():
    """
    Helper Class for generic_visit
    It is used to insert Flog statements at the beginning of file
    And do some garbage collection at EOF
    """

    def __init__(self,
                 client_root: ClientRoot,
                 docstring=None):
        self.client_root = client_root
        self.docstring = docstring

    def contains_docstring(self):
        return self.docstring is not None

    def get_heads(self):
        return self.client_root.parse_heads()

    def get_foot(self):
        return self.client_root.parse_foot()

    def proc_imports(self, node):
        """
        Helper method for generic_visit
        :param node:
        :return:
        """

        def is_future(node):
            if isinstance(node, ast.ImportFrom):
                return node.module == '__future__'
            elif isinstance(node, ast.Import):
                for each in node.names:
                    if is_future(each):
                        return True
                return False
            elif isinstance(node, ast.alias):
                return node.name == '__future__'
            else:
                return False

        contains_future = any(map(is_future, node.body))

        first_import = None
        last_import = None
        if contains_future:
            for i, child in enumerate(node.body):
                if first_import is None and last_import is None and not (
                        isinstance(child, ast.ImportFrom) or isinstance(child, ast.Import)):
                    continue
                elif first_import is None and last_import is None and (
                        isinstance(child, ast.ImportFrom) or isinstance(child, ast.Import)):
                    first_import = i
                elif first_import is not None and last_import is None and (
                        isinstance(child, ast.ImportFrom) or isinstance(child, ast.Import)):
                    continue
                elif first_import is not None and last_import is None and not (
                        isinstance(child, ast.ImportFrom) or isinstance(child, ast.Import)):
                    last_import = i - 1
                    break
                else:
                    raise RuntimeError(
                        "Case not handled. [first_import is None, {}]. [last_import is None, {}]. [is import, {}]".format(
                            first_import is None, last_import is None,
                            isinstance(child, ast.ImportFrom) or isinstance(child, ast.Import)
                        ))

            if last_import is None:
                # This is true whenever the full file is nothing but imports
                return node

            prefix = node.body[0:last_import + 1]
            postfix = node.body[last_import + 1:]

            heads = self.get_heads()

            heads.extend(postfix)
            prefix.extend(heads)
            node.body = prefix
            node.body.extend(self.get_foot())
        else:
            heads = self.get_heads()
            heads.extend(node.body)
            node.body = heads
            node.body.extend(self.get_foot())
