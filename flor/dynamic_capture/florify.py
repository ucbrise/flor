from flor.dynamic_capture.walker import Walker

if __name__ == "__main__":
    walker = Walker('/anaconda3/envs/flor/lib/python3.7/site-packages/pandas.backup')
    # walker = Walker('/Users/rogarcia/Desktop/dask')
    walker.compile_tree()