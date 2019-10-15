from flor.complete_capture.transformer import Transformer
from flor.constants import *
from flor.utils import write_failure_msg 

import tempfile
import os
import shutil
import ast
import astor
import traceback

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
            
            ROOT = 'site-packages'
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
                        'tensorflow_estimator',
                        'torch',
                        'torchvision',
                        'keras',
                        'nltk',
                        'spacy',
                        'scrapy'
                        }

            try:
                with open(abs_path, 'r') as f:
                    if 'flor_transformed' in f.readline():
                        return False
            except:
                with open(abs_path, 'r', encoding = 'ISO-8859-1') as f:
                    if 'flor_transformed' in f.readline():
                        return False

            abs_path = abs_path.split(os.path.sep)
            abs_path = (os.path.sep).join(abs_path[abs_path.index(ROOT) + 1:])

            for keep_element in KEEP_SET:
                if keep_element == abs_path[0:len(keep_element)]:
                    return True
            return False

        for (src_root, dirs, files) in os.walk(self.rootpath):
            # SRC_ROOT: in terms of Conda-Cloned environment
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == '.py' and keep(os.path.join(src_root, file)):
                    lib_code and print('transforming {}'.format(os.path.join(src_root, file)))
                    try:
                        save_in_case = None
                        if lib_code:
                            transformer = Transformer(self.transformer(os.path.join(src_root, file)))
                        else:
                            transformer = Transformer(os.path.join(src_root, file))
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
                        except:
                            with open(os.path.join(src_root, file), 'w') as f:
                                f.write(save_in_case)
                    except Exception as e:
                        # Failed to transform
                        # TODO: Better policy is to transform much of a file rather than to fully ignore a file
                        print(e)
                        write_failure_msg(str(e))
                        tback = traceback.format_exc()
                        print(tback)
                        write_failure_msg(str(tback))
                        failed_transforms.append(os.path.join(src_root, file))
                        print("FAILED TO TRANSFORM: {}".format(failed_transforms[-1]))
                        write_failure_msg("FAILED TO TRANSFORM: {}".format(failed_transforms[-1]))
        if failed_transforms:
            print("FAILED TO TRANSFORM:")
            for each in failed_transforms:
                print(each)



