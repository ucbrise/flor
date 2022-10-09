from inspect import stack


class Flor:
    """
    nesting_lvl == 0 (==>) before loop
    nesting_lvl >= 1:
        nesting_lvl == 1 (==>) main loop 
        nesting_lvl >  1 (==>) nested loop
    """

    nesting_lvl = 0

    @staticmethod
    def loop(iter8r):

        try:
            Flor.nesting_lvl += 1
            lineno = stack()[1].lineno
            try:
                iter8r = iter(iter8r)  # type: ignore
                for each in iter8r:
                    if Flor.nesting_lvl == 2:
                        print(f"Entering loop at line {lineno}")
                    yield each
            except:
                return iter8r
        finally:
            Flor.nesting_lvl -= 1


for epoch in Flor.loop(range(5)):
    for batch in Flor.loop(range(10)):
        print(epoch, batch)
