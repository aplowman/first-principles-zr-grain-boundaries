[Binder](Binder)

TODO: 
  - Add section describing the `parameters/` folder.

This repository demonstrates the supporting Python code and input/output data files for the following manuscripts:

1. Zr GB properties...
2. Effect of defects in Zr GBs...

The easiest way to explore both the code and data is to use Binder, by clicking the link above.

## Data overview

Each sub-folder in the `data` folder is briefly explained below. Note that simulation input and output files have been individually compressed using the `lzma` Python module. When the data in these files is utilised in the Jupyter notebooks in this repository, the files are decompressed on the fly using the `decompress_file` function in the `utilities.py` module.

### `data/ab_initio_tensile_tests`

This folder contains the key CASTEP simulation input and output files (`.cell`, `.param`, `.castep`, `.geom`) for each simulated ab initio tensile test. For each simulated system, files are organised into sub-folders like "08_+0.200", which, for example, contains the simulation results for which 0.2 Angstroms of vacuum was added at each of the two pre-determined cleavage planes (i.e. at the grain boundary for the grain boundary systems). The "rgs" in these folders stands for "rigid grain shift".

### `data/dft_gamma_surfaces`

This folder contains the key CASTEP simulation input and output files (`.cell`, `.param`, `.castep`, `.geom`) for each simulated gamma surface, which is an exploration of the energy landscape formed by translating one micro-grain relative to the other. For each simulated system, the results are organised into folders such as "0.4_3.6__+0.6200", which refers to a simulation for which one of the micro-grains was translated in the boundary plane by a vector of (0/4, 3/6) (which simplifies to (0, 0.5)) in fractional coordinates of the supercell boundary area. The final component refers to the amount of additional vacuum added between the two micro-grain; here, it is 0.62 Angstroms.

### `data/dft_sims`

This folder contains the key CASTEP simulation input and output files (`.cell`, `.param`, `.castep`, `.geom`) for the pristine and defective grain boundary calculations. See below ("Structure codes") for the meaning behind the sub-folder naming scheme.

### `data/potential`

This folder contains a copy of the EAM empirical potential (M.I. Mendelev and  G.J. Ackland, Phil. Mag. Letters 87, 349-359 (2007)) used in the LAMMPS gamma surface simulations.

### `data/processed`

This folder contains results that were derived from the simulation outputs found in the other folders. The following sub-folders exist:

#### `data/processed/ab_initio_tensile_tests`

This sub-folder contains JSON files that include the results of the ab initio tensile tests. In particular, these JSON files can be loaded by the `atomistic` Python package. The traction separation curve can then be visualised. See the `ab_initio_tensile_test` Jupyter notebook for more details.

#### `data/processed/gamma_surfaces`

This sub-folder contains JSON files that include the results from gamma surface simulations from both LAMMPS (in the `eam` sub-folder) and CASTEP (in the `dft` sub-folder). These JSON file can be loaded by the `atomistic` Python package, which enables easy visualisation of the results. See the `gamma_surfaces` Jupyter notebook for more details.

## Structure codes

The results from simulations are organised and referred to by *structure codes*. In this work, we were interested in studying grain boundaries from first principles. Subsequently, and since we used a periodic plane-wave DFT code (CASTEP), we were limited to periodic grain boundary supercells. Periodic grain boundaries are commonly described by their integer Σ value. We also simulated bulk and free surface systems (to, for instance, compute the grain boundary energy). In these cases, the supercells were always associated with a corresponding grain boundary system of approximately the same size and shape. Thus, where we have results for a grain boundary, we also have results for corresponding free surface and bulk system, which are referenced using the same Σ value as the grain boundary.

By example, the structure code `s13-tlA-gb-b1+a3` can be decoded in the following way:

- `s13`: Σ13 system
- `tlA`: tilt grain boundary
- `gb`: grain boundary supercell
- `b1+a3`: two defects are present near the boundary:
  - `b1` refers to a caesium (Cs) substitutional atom at defect site 1
  - `a3` refers to an iodine (I) substitutional atom at defect site 3

In the defect specification, `a` refers to a substitutional iodine defect, `b` refers to a substitional caesium defect, and `v` refers to a vacancy. For each system, three defect sites were chosen at roughly increasing distance from the nominal GB plane as: `1`, `2` and `3`.

## Jupyter notebook overview

This repository contains several Jupyter notebooks that demonstrate and visualise the data and the analysis methods of the work. These make use of wrapper functions in the included Python modules `utilities.py`, `gamma_surfaces.py`, and `tensile_test.py`. These modules are designed to be lightweight wrappers over the more generalised code that exists primarily in the `atomistic` Python package (see below - "Dependencies overview").

### `gamma_surfaces`

This notebook demonstrates how to visualise the computed gamma surfaces, and runs through the steps to generate simulation input files and collate the results in order to compute a gamma surface.

### `ab_initio_tensile_test`

This notebook demonstrates how to visualise the computed traction-separation curves from the ab initio tensile tests.

### `schematics`

This notebook can be used to generate schematics of the simulated systems.

## Dependencies overview

During the course of this work, a number of Python packages were developed. These packages are dependencies of this repository. Their functions are briefly outlined below. Note that the required versions of these packages are noted/tracked in the `requirements.txt` file.

### `atomistic`

- This package is for generating and manipulating atomistic structures, including grain boundaries.
- It includes convenience classes that wrap up useful functionality for generating and analysing gamma surfaces and atomistic tensile tests.
- [PyPI link](https://pypi.org/project/atomistic/)
- [GitHub link](https://github.com/aplowman/atomistic)

### `gemo`

- The **ge**ometry **mo**deller package is used to enable visualisations of `AtomisticStructure` objects.
- 3D and arbitrary 2D orthographic projections are supported, using [Plotly](https://github.com/plotly/plotly.py) as a backend. (It is, however, designed to eventually support multiple backends.)
- [PyPI link](https://pypi.org/project/gemo/)
- [GitHub link](https://github.com/aplowman/gemo)

### `bravais`

- This package is for representing Bravais lattices and is used by `atomistic`.
- [PyPI link](https://pypi.org/project/bravais/)
- [GitHub link](https://github.com/aplowman/bravais)

### `spatial-sites`

- This package is for storing and manipulating arrays of vectors using `Numpy`.
- It was created to easily allow each vector in a `Numpy` array of vectors to be associated with additional information (labels).
- For example:
  - In the `atomistic` package, the `Sites` class is used to represent atoms and lattice sites within an `AtomisticStructure`.
  - The species of each atom, for instance, is stored as a `Sites` label. 
- [PyPI link](https://pypi.org/project/spatial-sites/)
- [GitHub link](https://github.com/aplowman/spatial-sites)

### `castep-parse`

- This package was developed to write and parse CASTEP input/output files.
- [PyPI link](https://pypi.org/project/castep-parse/)
- [GitHub link](https://github.com/aplowman/castep-parse)

### `lammps-parse`

- This package was developed to write and parse LAMMPS input/output files.
- [PyPI link](https://pypi.org/project/lammps-parse/)
- [GitHub link](https://github.com/aplowman/lammps-parse)
