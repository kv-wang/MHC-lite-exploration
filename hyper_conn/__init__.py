from .hyper_connections import (
    HyperConnections,
    get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions,
    Residual,
    StreamEmbed,
    AttentionPoolReduceStream
)

# export with mc prefix, as well as mHC

from .mhc import (
    ManifoldConstrainedHyperConnections,
    get_expand_reduce_stream_functions as mc_get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as mc_get_init_and_expand_reduce_stream_functions
)

from .mhc_lite import (
    MHCLite,
    get_expand_reduce_stream_functions as mhclite_get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as mhclite_get_init_and_expand_reduce_stream_functions
)

from .mhc_analysis import (
    MHCAnalysis,
    get_expand_reduce_stream_functions as mhc_analysis_get_expand_reduce_stream_functions,
    get_init_and_expand_reduce_stream_functions as mhc_analysis_get_init_and_expand_reduce_stream_functions
)

flag = False

def hyper_conn_init_func(hyper_conn_type: str, hyper_conn_n: int, **kwargs):
    global flag
    if not flag:
        print(f"HYPER_CONN: USING {hyper_conn_type} with {hyper_conn_n} streams, kwargs={kwargs}")
        flag = True

    if hyper_conn_type == "none":
        return get_init_and_expand_reduce_stream_functions(hyper_conn_n, disable = True)
    elif hyper_conn_type == "hc":
        return get_init_and_expand_reduce_stream_functions(hyper_conn_n)
    elif hyper_conn_type == "mhc":
        return mc_get_init_and_expand_reduce_stream_functions(hyper_conn_n, **kwargs)
    elif hyper_conn_type == "mhc_lite":
        return mhclite_get_init_and_expand_reduce_stream_functions(hyper_conn_n, **kwargs)
    elif hyper_conn_type == "analysis":
        return mhc_analysis_get_init_and_expand_reduce_stream_functions(hyper_conn_n, **kwargs)
    else:
        raise ValueError(f"Invalid hyper connection type: {hyper_conn_type}")

