# A Simple Tensor Network Program


Use ffmpeg to make movie:

    ffmpeg -framerate 50 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

A comprehensive python package for quantum dynamics, including: 
  - Davidson diagonalization `minitn.lib.numerical.DavidsonAlgorithm`,
  - Discrete variable representation basis `minitn.bases.dvr`, 
  - Grid basis `minitn.bases.grids`, 
  - Tensor Network framework `minitn.tensor`
  - Multi-layer multi-configuration time-dependent Hartree (ML-MCTDH) with standard and projector-splitting propagators `minitn.algorithm.ml`, 
  - General hierarchical equations of motion for discrete vibrations with tensor network decomposition `minitn.heom`,
  
 Deprecated:
  - Single layer MCTDH `minitn.algorithm.mctdh`
  - Special DMRG `minitn.algorithm.dmrg`

## TODO
[v] Generate a MultiLayer type of `h_list` from a string graph and a `h_list` as in `SpinBosonModel`. 

[x] Save wfns and partial envs in `MultiLayer` object rather than tensor: do not trust any array saved in `Tensor` or `Leaf`.  Treat them as some *templates*.  Leave some space for Multi-threading.

