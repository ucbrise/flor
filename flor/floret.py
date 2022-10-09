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
    def loop(iter8r, name=None):
        try:
            Flor.nesting_lvl += 1
            lineno = stack()[1].lineno
            src = stack()[1].filename
            try:
                iter8r = iter(iter8r)  # type: ignore
                for each in iter8r:
                    if Flor.nesting_lvl == 1:
                        static_id = {
                            "name": "outer loop" if name is None else name,
                            "lineno": lineno,
                            "src": src,
                        }
                    elif Flor.nesting_lvl > 1:
                        static_id = {
                            "name": "nested loop" if name is None else name,
                            "lineno": lineno,
                            "src": src
                        }
                    else:
                        raise ValueError(f"Invalid nesting level {Flor.nesting_lvl}")
                    print(static_id)  # type: ignore
                    yield each
            except:
                return iter8r
        finally:
            Flor.nesting_lvl -= 1


if __name__ == "__main__":
    for epoch in Flor.loop(range(5)):
        for batch in Flor.loop(range(10)):
            pass
