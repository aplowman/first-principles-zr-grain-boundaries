import lzma
from pathlib import Path
from shutil import copy2
from pprint import pprint

import numpy as np
from ruamel.yaml import YAML
from humanize import naturalsize
from atomistic.atomistic import AtomisticStructure
from atomistic.atomistic import Sites
from atomistic.api.bicrystal import (
    bicrystal_from_csl_vectors,
    surface_bicrystal_from_csl_vectors,
    bulk_bicrystal_from_csl_vectors
)
from atomistic.bicrystal import Bicrystal
from castep_parse import read_cell_file, read_castep_file, read_geom_file

DEFAULT_DIRS_LIST_PATH = 'simulation_directories.yml'
DEF_LAMMPS_FILE = 'parameters/lammps_parameters.yml'
DEF_CASTEP_FILE = 'parameters/castep_parameters.yml'
DEF_POT_PATH = 'data/potential/Zr_3.eam.fs'
DEF_PARAMS_FILE = 'parameters/structural_parameters.yml'


def get_filtered_directory_size(path, sub_dirs=None, file_formats=None, recursive=False,
                                human_readable=True):
    """Get the total size of a directory tree, including only some files.

    Parameters
    ----------
    path : Path
        Root directory whose size is to be found.
    sub_dirs : list, optional
        List of glob patterns of directories (relative to `path`) to include. If `None`,
        all sub directories are included.
    file_formats : list, optional
        List of glob patterns of files to include. If `None`, all file formats are
        included.
    human_readable : bool, optional
        If True, return the size as a human-readable string. If False, return the size as
        an integer number of bytes.

    """

    if not file_formats:
        file_formats = ['*']

    if not sub_dirs:
        sub_dirs = ['.']

    valid_files = []

    for sub_dir_name in sub_dirs:
        sub_path = path.joinpath(sub_dir_name)
        glob_path = '**/*.*' if recursive else '*.*'
        for i in sub_path.glob(glob_path):
            if any([i.match(j) for j in file_formats]):
                valid_files.append(i)

    size_bytes = 0
    for i in valid_files:
        size_bytes += i.stat().st_size

    size = size_bytes
    if human_readable:
        size = naturalsize(size, binary=True, format='%.2f')

    return size


def copy_stuff(src_path, dst_path, sub_dirs=None, file_formats=None, compress=False):
    """Copy a directory tree recursively, keeping only specified sub-directories and file
    formats."""

    if dst_path.exists():
        raise ValueError('Destination path already exists: "{}"'.format(dst_path))

    if not file_formats:
        file_formats = ['*']

    if not sub_dirs:
        sub_dirs = ['.']

    # print('dst_path: {}'.format(dst_path))

    for sub_dir_name in sub_dirs:
        sub_src_path = src_path.joinpath(sub_dir_name)
        sub_dst_path = dst_path.joinpath(sub_dir_name)

        dst_sub_dir = dst_path.joinpath(sub_dir_name)

        # print('sub_dir_name: {}'.format(sub_dir_name))
        # print('dst_sub_dir: {}'.format(dst_sub_dir))

        dst_sub_dir.mkdir(parents=True)

        for i_src in sub_src_path.glob('*'):
            if not i_src.is_file():
                continue
            if any([i_src.match(j) for j in file_formats]):
                rel = i_src.relative_to(src_path)
                i_dst = dst_path.joinpath(rel)

                if compress:
                    dst_filename = str(i_dst.with_suffix(i_dst.suffix + '.xz'))
                    with i_src.open('rb') as handle:
                        with lzma.open(dst_filename, 'w') as comp_handle:
                            comp_handle.write(handle.read())
                else:
                    copy2(str(i_src), str(i_dst))


def get_simulation_path(structure_code, dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    all_paths = get_structure_paths(dirs_list_path)
    out = {
        'path': all_paths['paths'][structure_code],
        'compressed': all_paths['compressed'],
    }
    return out


def get_structure_paths(dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    with Path(dirs_list_path).open() as handle:
        sims_dirs = YAML().load(handle)

    paths = sims_dirs['simulation_directories']
    for structure_code in paths:
        paths[structure_code] = Path(sims_dirs['root_directory'], paths[structure_code])

    out = {
        'paths': paths,
        'compressed': sims_dirs['compressed']
    }

    return out


def get_all_structure_codes(dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    paths = get_structure_paths(dirs_list_path)['paths']
    return list(paths.keys())


def get_all_simulation_paths(dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    paths = get_structure_paths(dirs_list_path)
    out = {
        'paths': list(paths['paths'].values()),
        'compressed': paths['compressed']
    }

    return out


def decompress_file(path, as_string=False):
    with path.open('rb') as handle:
        data = lzma.decompress(handle.read())
    if as_string:
        data = data.decode("utf-8")
    return data


def get_simulation_structure(structure_code, dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    sim_path = get_simulation_path(structure_code, dirs_list_path)

    cell_input = sim_path['path'].joinpath('sim.cell')
    if sim_path['compressed']:
        cell_input = cell_input.with_suffix(cell_input.suffix + '.xz')
        cell_input = decompress_file(cell_input)

    cell_dat = read_cell_file(cell_input)

    sites = {
        'atoms': Sites(
            coords=cell_dat['atom_sites'],
            vector_direction='column',
            labels={
                'species': [cell_dat['species'], cell_dat['species_idx']],
            }
        )
    }

    out = AtomisticStructure(
        sites=sites,
        supercell=cell_dat['supercell']
    )

    return out


def get_castep_outputs(structure_code):
    'Get the outputs from a castep simulation, from the compressed simulation files.'

    sim_path = Path('data/dft_sims').joinpath(structure_code)

    cst_path = sim_path.joinpath('sim.castep.xz')
    geom_path = sim_path.joinpath('sim.geom.xz')

    cst_file = decompress_file(cst_path)
    geom_file = decompress_file(geom_path)

    cst = read_castep_file(cst_file)
    geom = read_geom_file(geom_file)

    cst['geom'] = geom

    return cst


def get_simulated_bicrystal(structure_code, opt_idx=-1, parameters_file=DEF_PARAMS_FILE):
    'Construct a Bicrystal from the final structure in a simulation.'

    yaml = YAML()
    with Path(parameters_file).open(encoding='utf-8') as handle:
        parameters = yaml.load(handle)

    sim_outputs = get_castep_outputs(structure_code)
    atoms = sim_outputs['geom']['ions'][opt_idx]
    species = sim_outputs['geom']['species'][sim_outputs['geom']['species_idx']]
    supercell = sim_outputs['geom']['cells'][opt_idx]
    bicrystal = Bicrystal.from_atoms(
        atoms=atoms,
        species=species,
        supercell=supercell,
        non_boundary_idx=2,
        maintain_inv_sym=True,
        overlap_tol=parameters['overlap_tol'],
        wrap=True,
    )

    sup_type = structure_code.split('-')[2]
    if sup_type == 'b':
        bicrystal.meta['supercell_type'] = ['bulk', 'bulk_bicrystal']
    elif sup_type == 'fs':
        bicrystal.meta['supercell_type'] = ['surface', 'surface_bicrystal']

    return bicrystal


def get_lammps_parameters(potential_file=DEF_POT_PATH, parameters_file=DEF_LAMMPS_FILE,
                          base_path=None):

    if base_path is None:
        base_path = ''
    base_path = Path(base_path)

    with Path(parameters_file).open() as handle:
        params = YAML().load(handle)

    # Substitute in the potential file:
    if not Path(potential_file).exists():
        raise ValueError('Potential file "{}" does not exist.'.format(potential_file))

    for idx, i in enumerate(params['interactions']):
        if '<<POTENTIAL_PATH>>' in i:
            pot_path = base_path.joinpath(potential_file)
            params['interactions'][idx] = i.replace('<<POTENTIAL_PATH>>', str(pot_path))

    return params


def get_castep_parameters(num_atoms, parameters_file=DEF_CASTEP_FILE):

    with Path(parameters_file).open() as handle:
        params = YAML().load(handle)

    atom_const = params.get('atom_constraints')

    if atom_const is not None:

        valid_atom_cnst = {}
        for key, val in atom_const.items():

            if isinstance(val, list):
                valid_atom_cnst.update({key: np.array(val)})

            elif isinstance(val, str):

                if val.upper() == 'NONE':
                    valid_atom_cnst.update({key: None})

                elif val.upper() == 'ALL':
                    all_atm = np.arange(num_atoms) + 1
                    valid_atom_cnst.update({key: all_atm})

        params['atom_constraints'] = valid_atom_cnst

    return params


def make_structure(structure_code, configuration='base', lattice_parameters='dft',
                   parameters_file=DEF_PARAMS_FILE,
                   dirs_list_path=DEFAULT_DIRS_LIST_PATH):
    """Construct the AtomisticStructure object from parameters.


    Parameters
    ----------
    structure_code: str
        The string code that describes the sigma value, boundary type and supercell type
        of the structure. Examples include: "s7-tlA-gb" (a sigma 7 tilt GB) and
        "s13-tw-fs" (sigma 13 free surface supercell derived from the corresponding twist
        GB). The structure code must match those found in the "simulation_directories.yml"
        file.
    configuration : str
        One of "base", "sized" or "minimum_energy".
    lattice_parameters : str
        One of "dft" or "eam". Optimised lattice parameter depend on the method of atomic
        relaxation.
    parameters_file : str or Path, optional
        Path to the YAML file containing structure parameters.
    dirs_list_path : str or Path, optional
        Path to the YAML file containing directories for simulation data.

    """

    yaml = YAML()
    with Path(parameters_file).open(encoding='utf-8') as handle:
        parameters = yaml.load(handle)

    sigma_code, gb_type, sup_type = structure_code.split('-')
    gb_code = '{}-{}'.format(sigma_code, gb_type)

    crystal_structure = parameters['crystal_structures'][lattice_parameters]
    csl_vecs = parameters['csl_vecs'][sigma_code]
    box_csl = parameters['box_csl'][gb_type]

    if configuration in ['sized', 'minimum_energy']:
        box_csl *= np.array(parameters['csl_repeats'][gb_code])

    struct_params = {
        'crystal_structure': crystal_structure,
        'csl_vecs': csl_vecs,
        'box_csl': box_csl,
        'overlap_tol': parameters['overlap_tol'],
    }

    if sup_type == 'gb':

        if configuration == 'minimum_energy':
            struct_params.update({
                'relative_shift': {
                    'shift': parameters['relative_shift'][gb_code],
                    'crystal_idx': 0,
                },
                'boundary_vac': [parameters['boundary_vac'][gb_code]],
            })

        struct_params.update({
            'maintain_inv_sym': True,
        })

        out = bicrystal_from_csl_vectors(**struct_params)

    elif sup_type == 'fs':
        out = surface_bicrystal_from_csl_vectors(**struct_params)

    elif sup_type == 'b':
        struct_params['csl_vecs'] = struct_params['csl_vecs'][0]
        out = bulk_bicrystal_from_csl_vectors(**struct_params)

    return out
