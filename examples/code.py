import flor

@flor.func
def action(literal, artifact, **kwargs):
    with open(artifact, 'w') as f:
        f.write('Hello\n')

with flor.Experiment('code') as ex:
    lit = ex.literal(v=0, name='literal')

    do_action = ex.action(action, [lit, ])
    art = ex.artifact('artifact', 'artifact', do_action)

art.plot()
