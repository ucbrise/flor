import flor
import random
import datetime

print("Hello World")

coin = flor.pin("coin", "HEADS" if random.randint(0, 1) else "TAILS")
now = flor.pin("time", str(datetime.datetime.now()))

print(f"You flipped a {coin} on {now}")
flor.report_end()
