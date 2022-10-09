from inspect import stack
from .iterator import it, load_kvs, report_end, replay_clock
from .skipblock import SkipBlock


class Flor:
    """
    nesting_lvl == 0 (==>) before loop
    nesting_lvl >= 1:
        nesting_lvl == 1 (==>) main loop 
        nesting_lvl >  1 (==>) nested loop
    """

    nesting_lvl = 0
    load_kvs = load_kvs
    chckpts = []  # type: ignore

    @staticmethod
    def checkpoints(*args):
        Flor.chckpts.extend(list(args))

    @staticmethod
    def loop(iter8r, name=None, probed=None):
        """
        Commits after every outer loop
        """
        try:
            Flor.nesting_lvl += 1
            assert Flor.nesting_lvl >= 1
            static_id = {
                "name": "outer loop" if Flor.nesting_lvl == 1 else "nested loop",
                "lineno": stack()[1].lineno,
                "src": stack()[1].filename,
            }
            name = str(static_id) if name is None else name
            if Flor.nesting_lvl == 1:
                # Outer loop
                for each in it(iter8r):
                    replay_clock.epoch += 1
                    yield each
            else:
                assert Flor.nesting_lvl > 1
                # Nested loop
                if SkipBlock.step_into(name, probed):
                    for each in iter8r:
                        yield each
                SkipBlock.end(*Flor.chckpts)
        finally:
            Flor.nesting_lvl -= 1

    @staticmethod
    def commit():
        report_end()


if __name__ == "__main__":
    for epoch in Flor.loop(range(5)):
        for batch in Flor.loop(range(10)):
            pass
