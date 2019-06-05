import pandas as pd
from flor import OpenLog

with OpenLog('dummy'):
    movie_revies = pd.read_json('data.json')
    print("movies loaded successfully")
