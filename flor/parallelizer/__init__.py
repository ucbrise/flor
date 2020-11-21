from flor.constants import *
from flor import stateful, utils
from flor.skipblock.skip_block import SkipBlock
from flor.writer import Writer

import sys
import functools

def partition(iterator, partition_id, num_partitions):
    if stateful.MODE is EXEC:
        # This method is pass through on exec
        return iterator
    assert partition_id >= 0 and partition_id < num_partitions
    stateful.PSEUDORESUMING = True
    partition_id = int(partition_id)
    SkipBlock.parallel = True

    pretraining = stateful.pretraining
    iterations_count = len(iterator)
    period = stateful.period

    psl = Writer.partitioned_store_load
    if len(psl) > iterations_count:
        # This is true when Train & Eval loop share the same looper (see Rnn Translator)
        assert len(psl) % iterations_count == 0
        # We will stitch adjacents together
        new_group_size = int(len(psl) / iterations_count)
        new_psl = []
        current_group = None
        for i,each in enumerate(psl):
            if i % new_group_size == 0:
                new_psl.append(current_group)
                current_group = []
            current_group += each
        new_psl.append(current_group)
        assert new_psl.pop(0) is None
        assert len(new_psl) == iterations_count
        Writer.partitioned_store_load = new_psl
    del psl



    stateful.iterations_count = iterations_count

    epoch_partitions = utils.get_partitions(len(iterator), num_partitions, pretraining, period)

    our_epochs = epoch_partitions[partition_id]
    if not our_epochs:
        sys.exit(0)

    predecessor_id = our_epochs[0] - 1
    if predecessor_id >= 0 and stateful.PRED_INIT_MODE is WEAK:
        Writer.store_load = functools.reduce(lambda x,y: x+y, Writer.partitioned_store_load[predecessor_id:])
    # In case of STRONG init mode, just leave store_load as it is, it already has
    # What it needs to start from 0. It doesn't need to start at some k.

    if stateful.PRED_INIT_MODE is WEAK:
        predecessor_epochs = [predecessor_id,] if predecessor_id >= 0 else []
    else:
        predecessor_epochs = range(predecessor_id + 1)

    for pred in predecessor_epochs:
        print(f"Initializing epoch {pred}")
        yield iterator[pred]

    import flor
    flor.SKIP = False
    stateful.PSEUDORESUMING = False

    for epoch in our_epochs:
        print(f"Executing epoch {epoch}")
        yield iterator[epoch]




