import jarvis
import pickle

@jarvis.func
def multiply(in_artifacts, out_artifacts):
    in_locations = [in_art.getLocation() for in_art in in_artifacts]
    out_loc = out_artifacts[0].getLocation()
    with open(in_locations[0], 'rb') as f:
        x = pickle.load(f)
    with open(in_locations[1], 'rb') as f:
        y = pickle.load(f)
    z = int(x)*int(y)
    with open(out_loc, 'w') as f:
        f.write(str(z) + '\n')
    print(z)