from florist.transformer import Transformer

import tempfile
import os
import shutil
import ast
import astor

class Walker():

    def __init__(self, rootpath):
        self.rootpath = os.path.abspath(rootpath)
        self.targetbasis = tempfile.mkdtemp(prefix='florist')
        self.targetpath = os.path.join(self.targetbasis, os.path.basename(self.rootpath))

        print("Target directory at: {}".format(self.targetpath))

        shutil.copytree(self.rootpath, self.targetpath)

    def compile_tree(self):
        for ((src_root, _, _), (dest_root, dirs, files)) in zip(os.walk(self.rootpath), os.walk(self.targetpath)):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == '.py':
                    print('transforming {}'.format(os.path.join(src_root, file)))
                    transformer = Transformer(os.path.join(src_root, file))
                    with open(os.path.join(dest_root, file), 'r') as f:
                        astree = ast.parse(f.read())
                    new_astree = transformer.visit(astree)
                    with open(os.path.join(dest_root, file), 'w') as f:
                        f.write(astor.to_source(new_astree))



