from flor.complete_capture.walker import Walker

if __name__ == "__main__":
    """
    Warning: Walker overwrites the path that you give it. Don't try this on your anaconda environment
    """
    walker = Walker('/anaconda3/envs/flor/lib/python3.7/site-packages/')
    walker.compile_tree()
