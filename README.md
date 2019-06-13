# A Simple Tensor Network Program

For my thesis.

## TODO
[v] Generate a MultiLayer type of `h_list` from a string graph and a `h_list` as in `SpinBosonModel`. 

[x] Save wfns and partial envs in `MultiLayer` object rather than tensor: do not trust any array saved in `Tensor` or `Leaf`.  Treat them as some *templates*.  Leave some space for Multi-threading.

[x] Try: remove all array attributes in `Tensor` and `Leaf` for safe Multi-threading.

[x] Try: better partial env_cache in projector-splitting method.

[x] Add MPO type of Hamiltonian.

[x] Try `unittest`.

[x] Try new model*s*.

[x] Try ...
