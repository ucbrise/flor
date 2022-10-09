from flor import Flor
import flor

import random
import datetime

print("Hello World")

coin = flor.recall("coin", "HEADS" if random.randint(0, 1) else "TAILS")
now = flor.recall("time", str(datetime.datetime.now()))

print(f"You flipped a {coin} on {now}")
Flor.commit()
