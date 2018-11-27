Context Free Grammar of the Log
================================

Below is the Context Free Grammar for the log files emitted by Flor.

.. code-block:: bash

    ROOT -> BLOCK_NODE
            | ROOT, BLOCK_NODE

    BLOCK_NODE -> {
        "block_type" : "(function_body FUNC_NAME
                        | loop_body)",
        ("consumes_from" : [ FUNC_NAME_SEQ ], )?
        "log_sequence" : LOG_SEQUENCE
    }

    FUNC_NAME -> STR
    FUNC_NAME_SEQ -> FUNC_NAME
                     | FUNC_NAME_SEQ, FUNC_NAME

    LOG_SEQUENCE -> [ LOG_SEQUENCE_MEMBERS ]

    LOG_SEQUENCE_MEMBERS -> LOG_RECORD
                            | BLOCK_NODE
                            | LOG_SEQUENCE_MEMBERS, LOG_RECORD
                            | LOG_SEQUENCE_MEMBERS, BLOCK_NODE

    LOG_RECORD -> {
        "assignee" : STR,
        "caller" : STR,
        "from_arg": BOOL,
        "in_execution" : FUNC_NAME,
        "in_file" : PATH,
        "instruction_no" : INT,
        "keyword_name" : STR,
        "pos": INT,
        "runtime_value" : STR_SERIALIZABLE,
        "typ" : STR,
        "value" : STR
    }

    STR_SERIALIZABLE -> STR



Every Flor log has exactly one `ROOT` node as the root.

We call a value **string serializable** if and only if:

.. code-block:: python

    x == eval(str(x))

This means, for example, that a Pandas Dataframe is not string-serializable.

Documentation for the elements in `LOG_RECORD` to follow...