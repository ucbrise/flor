import os

def exec_cp(args):
    src = args.src
    dst = args.dst

    assert os.path.splitext(src)[1] == '.py'
    assert os.path.splitext(dst)[1] == '.py'

    src = os.path.abspath(src)
    dst = os.path.abspath(dst)

    # Write neecessary data header
    # TODO: Also write the Git commit hash

    with open(src, 'r') as f:
        with open(dst, 'w') as g:
            g.write("#" + src + "\n")
            g.write(f.read())
