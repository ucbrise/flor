from flor.complete_capture.transformer import LibTransformer, ClientTransformer
from flor.constants import *

import tempfile
import os
import shutil
import ast
import astor


class Walker():

    def __init__(self, rootpath):

        def transformer_parameterizer(src_root, dst_root):
            src_root = src_root.split(os.path.sep)
            dst_root = dst_root.split(os.path.sep)
            def transformer(dst_path):
                dst_path = dst_path.split(os.path.sep)
                return os.path.sep.join(src_root + dst_path[len(dst_root) :])
            return transformer


        self.rootpath = os.path.abspath(rootpath)
        self.targetbasis = tempfile.mkdtemp(prefix='florist')
        self.targetpath = os.path.join(self.targetbasis, os.path.basename(self.rootpath))

        with open(os.path.join(FLOR_DIR, '.conda_map'), 'r') as f:
            src_root, dst_root = f.read().strip().split(',')

        self.transformer = transformer_parameterizer(src_root, dst_root)

        shutil.copytree(self.rootpath, self.targetpath)

    def compile_tree(self, lib_code=True):

        failed_transforms = []

        def keep(abs_path):
            if not lib_code:
                return True

            j = os.path.join
            ROOT = 'site-packages'
            SKIP_SET = {'attr',
                        'backcall',
                        'backports',
                        'Cython',
                        'future',
                        'ipykernel',
                        'IPython',
                        'jsonpickle',
                        'jupyter_core',
                        'pip',
                        'pkg_resources',
                        'pkginfo',
                        'py',
                        'setuptools',
                        j('scipy', 'misc'),
                        'flor',
                        'pyflor',
                        'pkg_resources',
                        'six.py',
                        'pytest.py',
                        '_pytest'}

            KEEP_SET = {'numpy',
                        'scipy',
                        'pandas',
                        'statsmodels',
                        'matplotlib',
                        'seaborn',
                        'plotly',
                        'bokeh',
                        'pydot.py',
                        'sklearn',
                        'xgboost',
                        'lightgbm',
                        'eli5',
                        'tensorboard',
                        'tensorflow',
                        'tensorflow_estimator'
                        'torch',
                        'torchvision',
                        'keras',
                        'nltk',
                        'spacy',
                        'scrapy'
                        }

            abs_path = abs_path.split(os.path.sep)
            abs_path = (os.path.sep).join(abs_path[abs_path.index(ROOT) + 1 : ])

            # for skip_element in SKIP_SET:
            #     if skip_element == abs_path[0:len(skip_element)]: return False
            # return True
            for keep_element in KEEP_SET:
                if keep_element == abs_path[0:len(keep_element)]: return True
            return False

        lib_code and print("Target directory at: {}".format(self.targetpath))
        for ((src_root, _, _), (dest_root, dirs, files)) in zip(os.walk(self.rootpath), os.walk(self.targetpath)):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == '.py' and keep(os.path.join(src_root, file)):
                    lib_code and print('transforming {}'.format(os.path.join(src_root, file)))
                    try:
                        if lib_code:
                            transformer = LibTransformer(self.transformer(os.path.join(src_root, file)))
                        else:
                            transformer = ClientTransformer(os.path.join(src_root, file))
                        try:
                            with open(os.path.join(dest_root, file), 'r') as f:
                                astree = ast.parse(f.read())
                        except:
                            with open(os.path.join(dest_root, file), 'r', encoding='ISO-8859-1') as f:
                                astree = ast.parse(f.read())
                        new_astree = transformer.visit(astree)
                        with open(os.path.join(dest_root, file), 'w') as f:
                            f.write(astor.to_source(new_astree))
                    except:
                        # Failed to transform
                        failed_transforms.append(os.path.join(src_root, file))
                        print("FAILED TO TRANSFORM: {}".format(failed_transforms[-1]))
        if failed_transforms:
            with open('failures.txt', 'w') as f:
                print("FAILED TO TRANSFORM:")
                for each in failed_transforms:
                    print(each)
                    f.write(each + '\n')



