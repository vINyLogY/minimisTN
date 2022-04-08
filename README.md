# A Simple Tensor Network Program

Use ffmpeg to make movie:

    ffmpeg -framerate 50 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

## TODO
[v] Generate a MultiLayer type of `h_list` from a string graph and a `h_list` as in `SpinBosonModel`. 

[x] Save wfns and partial envs in `MultiLayer` object rather than tensor: do not trust any array saved in `Tensor` or `Leaf`.  Treat them as some *templates*.  Leave some space for Multi-threading.

[x] Try: remove all array attributes in `Tensor` and `Leaf` for safe Multi-threading.

[x] Try: better partial env_cache in projector-splitting method.

[x] Add MPO type of Hamiltonian.

[x] Try `unittest`.

[x] Try new model*s*.

[x] Try ...
