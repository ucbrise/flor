import flor

ex = flor.Experiment('plate_demo')

ex.groundClient('ground')

ones = ex.literal([1, 2, 3], "ones")
ones.forEach()

tens = ex.literal([10, 100], "tens")
tens.forEach()

@flor.func
def multiply(x, y):
    z = x*y
    print(z)
    return z

doMultiply = ex.action(multiply, [ones, tens])
product = ex.artifact('product.txt', doMultiply)

product.parallelPull()
import os
import subprocess
def get_sha(repo):
	sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo).decode('ascii').strip()
	return sha

original = os.getcwd()
repo = ex.xp_state.versioningDirectory + '/plate_demo'
hashed = get_sha(repo)
flor.above_ground.commit(ex.xp_state, hashed)
# product.plot()
