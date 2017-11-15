import jarvis
import pickle

@jarvis.func
def multiply(in_artifacts, out_artifacts):
    in_locations = [in_art.getLocation() for in_art in in_artifacts]
    out_loc = out_artifacts[0].getLocation()
    try:
        isPickle = in_locations[0].split('.')[1] == 'pkl'
        if isPickle:
            with open(in_locations[0], 'rb') as f:
                x = pickle.load(f)
        else:
            with open(in_locations[0], 'r') as f:
                x = f.readline().strip()
    except:
        print('cant load product')
    try:
        isPickle = in_locations[1].split('.')[1] == 'pkl'
        if isPickle:
            with open(in_locations[1], 'rb') as f:
                y = pickle.load(f)
        else:
            with open(in_locations[1], 'r') as f:
                y = f.readline().strip()
    except:
        print('cant load ith_param3')
    z = int(x)*int(y)
    with open(out_loc, 'w') as f:
        f.write(str(z) + '\n')
    if out_loc == 'product2.txt':
        print(z)