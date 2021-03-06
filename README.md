# heterogeneous-exposure-hons
[![DOI](https://zenodo.org/badge/364075936.svg)](https://zenodo.org/badge/latestdoi/364075936)

Modules and scripts to model contagions on higher-order networks with heterogeneous exposure.

## Reference

If you use this code, please consider citing:

"[Universal nonlinear infection kernel from heterogeneous exposure on higher-order networks](https://arxiv.org/abs/2101.07229)" <br>
[Guillaume St-Onge](https://www.gstonge.ca), [Hanlin Sun](https://scholar.google.com/citations?user=b3mTmVgAAAAJ&hl=zh-CN), [Antoine Allard](https://antoineallard.github.io), [Laurent Hébert-Dufresne](https://laurenthebertdufresne.github.io), [Ginestra Bianconi](https://maths.qmul.ac.uk/~gbianconi/) <br>
arXiv:2101.07229

## Requirements

The following packages are used by certain modules:

- numpy
- scipy
- numba
- pandas
- matplotlib
- matplotlib-label-lines

Also, if you wish to reproduce the simulation, you will need the following
packages:

- [horgg](https://github.com/gstonge/horgg), for the generation of synthetic hypergraphs
- [schon](https://github.com/gstonge/schon), for the simulation of contagions on higher-order networks
