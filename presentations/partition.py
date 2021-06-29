def partition(iterator, partition_id, num_gpus):
    # Assert that we are Re-executing

    # Get program data from First-Execution
    # The number of iterations of outermost loop, etc.

    # Partition the N epochs into NUM_GPUs Partitions
    epoch_partitions = utils.get_partitions(len(iterator), num_gpus)
    # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]  --- 12 epochs on 4 GPUs

    our_epochs = epoch_partitions[partition_id]
    # [3, 4, 5]

    # SEEK: Advance the head of the log to the correct position
    predecessor_id = our_epochs[0] - 1
    # predecessor_id = 2
    if predecessor_id >= 0 and PRED_INIT_MODE is WEAK:
        # Skip forward in the log to the correct predecessor epoch
        # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]  --- 12 epochs on 4 GPUs
        #         ^                                           for partition_id = 1, PRED_INIT_MODE WEAK
        ...
    # In case of STRONG init mode, just leave store_load as it is
    # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]  --- 12 epochs on 4 GPUs
    #   ^                                                 for partition_id = 1, PRED_INIT_MODE STRONG

    if PRED_INIT_MODE is WEAK:
        predecessor_epochs = [predecessor_id,] if predecessor_id >= 0 else []
        # [2]         for weak initialization
    else:
        predecessor_epochs = range(predecessor_id + 1)
        # [0, 1, 2]   for strong initialization

    # Initialize every predecessor
    for pred in predecessor_epochs:
        print(f"Initializing epoch {pred}")
        yield iterator[pred]

    # Set a global Flor flag to force full re-execution
    import flor

    flor.SKIP = False

    # Do the work in your partition
    for epoch in our_epochs:
        print(f"Executing epoch {epoch}")
        yield iterator[epoch]


# EXAMPLE
for epoch in flor.partition(range(12), 1, 4):
    train()
    eval()
