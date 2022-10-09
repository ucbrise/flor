from inspect import stack


class Thing:
    @property
    def my_line_num(self):
        return stack()[1].lineno

    def _method(self, x):
        s = stack()
        me, parent = s[0], s[1]
        me_fna, p_fna = me.function, parent.function
        print(x, "method called")


# print(f"Entering loop at line {int(Thing().my_line_num) + 1}")
# for each in range(5):
#     print(f"Entering loop at line {int(Thing().my_line_num) + 1}")
#     for batch in range(10):
#         print(each, batch)


class Floret:
    nesting_lvl = 0

    @staticmethod
    def loop(iter8r):
        """
        nesting_lvl >= 1
            nesting_lvl == 1 (==>) main loop 
            nesting_lvl >  1 (==>) nested loop
        """
        try:
            Floret.nesting_lvl += 1
            lineno = stack()[1].lineno
            try:
                iter8r = iter(iter8r)  # type: ignore
                for each in iter8r:
                    if Floret.nesting_lvl == 2:
                        print(f"Entering loop at line {lineno}")
                    yield each
            except:
                return iter8r
        finally:
            Floret.nesting_lvl -= 1


for epoch in Floret.loop(range(5)):
    for batch in Floret.loop(range(10)):
        print(epoch, batch)
