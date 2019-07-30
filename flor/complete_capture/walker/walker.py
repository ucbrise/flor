from flor.complete_capture.transformer import LibTransformer, ClientTransformer
from flor.constants import *

import tempfile
import os
import shutil
import ast
import astor


class Walker():

    def __init__(self, rootpath):
        """
        :param rootpath: Absolute path we want to transform
        """

        def transformer_parameterizer(src_root, dst_root):
            """
            See nested transformer, this is just a parameterizer
            :param src_root: the path to the root anaconda environment that was cloned to produce the flor environment e.g. base
            :param dst_root: the path to the destination anaconda environment, the clone e.g. flor
            :return:
            """
            src_root = src_root.split(os.path.sep)
            dst_root = dst_root.split(os.path.sep)
            def transformer(dst_path):
                """
                When we walk the Conda-Clone, paths are in terms of the Conda Clone
                However, we don't want to show the programmer transformed code,
                So we will keep track of the source of the transformed file and show that instead.
                Given a Conda-Clone path, this function returns the Base-path.
                :param dst_path: Conda-clone path (absolute path of file in e.g. Flor conda env)
                :return: The path where you can find this untransformed file in Base (where we cloned from)
                """
                dst_path = dst_path.split(os.path.sep)
                return os.path.sep.join(src_root + dst_path[len(dst_root) :])
            return transformer


        self.rootpath = os.path.abspath(rootpath)


        with open(os.path.join(FLOR_DIR, '.conda_map'), 'r') as f:
            src_root, dst_root = f.read().strip().split(',')

        self.transformer = transformer_parameterizer(src_root, dst_root)

    def compile_tree(self, lib_code=True):
        """
        We now do transformation in place
        :param lib_code:
        :return:
        """

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

        def patch_keep(abs_path):
            if not lib_code:
                return True

            try:
                if 'flor_transformed' in open(abs_path).readline():
                    return False
            except:
                if 'flor_transformed' in open(abs_path, encoding = 'ISO-8859-1').readline():
                    return False

            j = os.path.join
            ROOT = 'site-packages'

            KEEP_SET = {'sklearn', 'scipy'}

            abs_path = abs_path.split(os.path.sep)
            abs_path = (os.path.sep).join(abs_path[abs_path.index(ROOT) + 1 : ])

            # for skip_element in SKIP_SET:
            #     if skip_element == abs_path[0:len(skip_element)]: return False
            # return True
            for keep_element in KEEP_SET:
                if keep_element == abs_path[0:len(keep_element)]: return True
            return False

        for (src_root, dirs, files) in os.walk(self.rootpath):
            # SRC_ROOT: in terms of Conda-Cloned environment
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == '.py' and patch_keep(os.path.join(src_root, file)):
                    lib_code and print('transforming {}'.format(os.path.join(src_root, file)))
                    try:
                        save_in_case = None
                        if lib_code:
                            transformer = LibTransformer(self.transformer(os.path.join(src_root, file)))
                        else:
                            transformer = ClientTransformer(os.path.join(src_root, file))
                        try:
                            with open(os.path.join(src_root, file), 'r') as f:
                                save_in_case = f.read()
                                astree = ast.parse(save_in_case)
                        except:
                            with open(os.path.join(src_root, file), 'r', encoding='ISO-8859-1') as f:
                                save_in_case = f.read()
                                astree = ast.parse(save_in_case)
                        new_astree = transformer.visit(astree)
                        to_source = astor.to_source(new_astree)

                        # Conda symlinks --- no copy. We need to remove symlink first
                        try:
                            os.unlink(os.path.join(src_root, file))
                        except:
                            pass
                        # Now we can write without disturbing source conda.
                        try:
                            with open(os.path.join(src_root, file), 'w') as f:
                                f.write('#flor_transformed' + '\n')
                                f.write(to_source)
                                f.close()
                        except:
                            with open(os.path.join(src_root, file), 'w') as f:
                                f.write(save_in_case)
                    except:
                        # Failed to transform
                        failed_transforms.append(os.path.join(src_root, file))
                        print("FAILED TO TRANSFORM: {}".format(failed_transforms[-1]))
        if failed_transforms:
            with open(os.path.join(FLOR_DIR, 'failures.txt'), 'w') as f:
                print("FAILED TO TRANSFORM:")
                for each in failed_transforms:
                    print(each)
                    f.write(each + '\n')



