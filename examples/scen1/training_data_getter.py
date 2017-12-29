import project
import os
import pickle
import numpy as np

@project.func
def get_training_data(in_artifacts, out_artifacts):
    if os.path.isfile(out_artifacts[0].getLocation()):
        #run once
        return
    training_data = []
    startdir = os.getcwd()
    os.chdir('../deprecated/train')
    for _, _, filenames in os.walk('.'):
        for filename in filenames:
            try:
                int(filename[0])
                assert filename.split('.')[-1] == 'png'
                with open(filename, 'rb') as f:
                    encoded_string = f.read()
                    training_data.append((encoded_string, filename.split('_')[0]))
            except:
                pass
    os.chdir(startdir)
    with open(out_artifacts[0].getLocation(), 'wb') as f:
        # I'm going to sample because... this takes very long
        indices = np.random.permutation(range(len(training_data)))
        indices = indices[0:10]
        new_list = []
        for i in indices:
            new_list.append(training_data[i])
        pickle.dump(new_list, f)