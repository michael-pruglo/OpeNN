Creating a neural net from first principles.

**Note:**
 - when adding new src files, be sure to add them to both targets: src and tests.
 - to compile, we need to use `MSVC compiler` + change `cmake` from bundled to external
 - we also need `xtensor` and `xtensor-blas`(which depends on `blas` and `lapack`). The easiest way to achieve this is to install Conda package manager and run
```sh
conda install -c conda-forge cmake xtensor xtensor-blas openblas lapack
```

Note:
https://lecture-demo.ira.uka.de/neural-network-demo/

TODO:
maybe replace assertions in src/ by exceptions?

