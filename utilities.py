import lzma
import json
from pathlib import Path
from shutil import copy2
from time import time
from pprint import pprint

import numpy as np
from ruamel.yaml import YAML
from humanize import naturalsize
from atomistic.atomistic import AtomisticStructure
from atomistic.atomistic import Sites
from atomistic.api.bicrystal import (
    bicrystal_from_csl_vectors,
    surface_bicrystal_from_csl_vectors,
    bulk_bicrystal_from_csl_vectors,
)
from atomistic.atomistic import AtomisticSimulation
from atomistic.bicrystal import (
    Bicrystal,
    atomistic_simulation_from_bicrystal_parameters,
)
from castep_parse import (
    read_cell_file,
    read_castep_file,
    read_geom_file,
    read_den_fmt,
    read_relaxation,
    merge_cell_data,
    merge_geom_data,
)
from lammps_parse import read_lammps_output
from plotly import graph_objects
from plotly.colors import DEFAULT_PLOTLY_COLORS

DEFAULT_DIRS_LIST_PATH = "simulation_directories.yml"
DEF_LAMMPS_FILE = "parameters/lammps_parameters.yml"
DEF_CASTEP_FILE = "parameters/castep_parameters.yml"
DEF_POT_PATH = "data/potential/Zr_3.eam.fs"
DEF_PARAMS_FILE = "parameters/structural_parameters.yml"
UNIT_CONV = 16.02176565
DEFAULT_WIDTH_VOLUME_CHANGE_LIMITS = [-2, 2]


def get_filtered_directory_size(
    path, sub_dirs=None, file_formats=None, recursive=False, human_readable=True
):
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
        file_formats = ["*"]

    if not sub_dirs:
        sub_dirs = ["."]

    valid_files = []

    for sub_dir_name in sub_dirs:
        sub_path = path.joinpath(sub_dir_name)
        glob_path = "**/*.*" if recursive else "*.*"
        for i in sub_path.glob(glob_path):
            if any([i.match(j) for j in file_formats]):
                valid_files.append(i)

    size_bytes = 0
    for i in valid_files:
        size_bytes += i.stat().st_size

    size = size_bytes
    if human_readable:
        size = naturalsize(size, binary=True, format="%.2f")

    return size


def copy_stuff(src_path, dst_path, sub_dirs=None, file_formats=None, compress=False):
    """Copy a directory tree recursively, keeping only specified sub-directories and file
    formats."""

    if dst_path.exists():
        raise ValueError('Destination path already exists: "{}"'.format(dst_path))

    if not file_formats:
        file_formats = ["*"]

    if not sub_dirs:
        sub_dirs = ["."]

    # print('dst_path: {}'.format(dst_path))

    for sub_dir_name in sub_dirs:
        sub_src_path = src_path.joinpath(sub_dir_name)
        sub_dst_path = dst_path.joinpath(sub_dir_name)

        dst_sub_dir = dst_path.joinpath(sub_dir_name)

        # print('sub_dir_name: {}'.format(sub_dir_name))
        # print('dst_sub_dir: {}'.format(dst_sub_dir))

        dst_sub_dir.mkdir(parents=True)

        for i_src in sub_src_path.glob("*"):
            if not i_src.is_file():
                continue
            if any([i_src.match(j) for j in file_formats]):
                rel = i_src.relative_to(src_path)
                i_dst = dst_path.joinpath(rel)

                if compress:
                    dst_filename = str(i_dst.with_suffix(i_dst.suffix + ".xz"))
                    with i_src.open("rb") as handle:
                        with lzma.open(dst_filename, "w") as comp_handle:
                            comp_handle.write(handle.read())
                else:
                    copy2(str(i_src), str(i_dst))


def get_simulation_path(structure_code, dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    all_paths = get_structure_paths(dirs_list_path)
    out = {
        "path": all_paths["paths"][structure_code],
        "compressed": all_paths["compressed"],
    }
    return out


def get_structure_paths(dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    with Path(dirs_list_path).open() as handle:
        sims_dirs = YAML().load(handle)

    print(dict(sims_dirs["simulation_directories"]["dft_sims"]))

    paths = sims_dirs["simulation_directories"]
    for structure_code in paths:
        paths[structure_code] = Path(sims_dirs["root_directory"], paths[structure_code])

    out = {"paths": paths, "compressed": sims_dirs["compressed"]}

    return out


def get_all_structure_codes(dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    paths = get_structure_paths(dirs_list_path)["paths"]
    return list(paths.keys())


def get_all_simulation_paths(dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    paths = get_structure_paths(dirs_list_path)
    out = {"paths": list(paths["paths"].values()), "compressed": paths["compressed"]}

    return out


def decompress_file(path, as_string=False):
    with path.open("rb") as handle:
        data = lzma.decompress(handle.read())
    if as_string:
        data = data.decode("utf-8")
    return data


def get_simulation_structure(structure_code, dirs_list_path=DEFAULT_DIRS_LIST_PATH):

    sim_path = get_simulation_path(structure_code, dirs_list_path)

    cell_input = sim_path["path"].joinpath("sim.cell")
    if sim_path["compressed"]:
        cell_input = cell_input.with_suffix(cell_input.suffix + ".xz")
        cell_input = decompress_file(cell_input)

    cell_dat = read_cell_file(cell_input)

    sites = {
        "atoms": Sites(
            coords=cell_dat["atom_sites"],
            vector_direction="column",
            labels={
                "species": [cell_dat["species"], cell_dat["species_idx"]],
            },
        )
    }

    out = AtomisticStructure(sites=sites, supercell=cell_dat["supercell"])

    return out


def get_castep_outputs(structure_code):
    "Get the outputs from a castep simulation, from the compressed simulation files."

    sim_path = Path("data/dft_sims").joinpath(structure_code)

    cst_path = sim_path.joinpath("sim.castep.xz")
    geom_path = sim_path.joinpath("sim.geom.xz")

    cst_file = decompress_file(cst_path)
    geom_file = decompress_file(geom_path)

    cst = read_castep_file(cst_file)
    geom = read_geom_file(geom_file)

    out = merge_geom_data(cst, geom)

    return out


def get_simulated_bicrystal(
    structure_code, opt_idx=-1, parameters_file=DEF_PARAMS_FILE
):
    "Construct a Bicrystal from the final structure in a simulation."

    parameters = get_structural_parameter_data(parameters_file)
    sim_outputs = get_castep_outputs(structure_code)

    species = sim_outputs["geom"]["species"]
    supercell = sim_outputs["geom"]["iterations"][opt_idx]["cell"]
    atoms = sim_outputs["geom"]["iterations"][opt_idx]["atoms"]

    bicrystal = Bicrystal.from_atoms(
        atoms=atoms,
        species=species,
        supercell=supercell,
        non_boundary_idx=2,
        maintain_inv_sym=True,
        overlap_tol=parameters["overlap_tol"],
        wrap=True,
    )

    sup_type = structure_code.split("-")[2]
    if sup_type == "b":
        bicrystal.meta["supercell_type"] = ["bulk", "bulk_bicrystal"]
    elif sup_type == "fs":
        bicrystal.meta["supercell_type"] = ["surface", "surface_bicrystal"]

    return bicrystal


def get_simulation(structure_code, parameters_file=DEF_PARAMS_FILE):
    "Get an AtomisticSimulation object from CASTEP simulation output files."

    parameters = get_structural_parameter_data(parameters_file)

    cstp_file_bytes = b""
    cell_file_bytes = b""
    geom_file_bytes = b""

    structure_dir = Path("data/dft_sims").joinpath(structure_code)
    if not structure_dir.is_dir():
        msg = 'Cannot find simulation files for structure "{}"'.format(structure_code)
        raise ValueError(msg)

    if structure_dir.joinpath("0").is_dir():

        # Need to combine results from multiple castep sim directories

        print("-> Multiple directories.")

        for j_idx, j in enumerate(structure_dir.glob("*")):

            if j_idx == 0:
                # Only want the first cell file
                cell_path_j = j.joinpath("sim.cell.xz")
                cell_file_bytes = decompress_file(cell_path_j)

            cstp_path_j = j.joinpath("sim.castep.xz")
            cstp_file_bytes += decompress_file(cstp_path_j)

            geom_path_j = j.joinpath("sim.geom.xz")
            if geom_path_j.is_file():
                geom_file_bytes += decompress_file(geom_path_j)

            # Special cases
            # -------------
            if structure_code == "s7-tlA-fs-b2":
                # First castep file is corrupted, so ignore this and geom file.
                if j_idx == 0:
                    cstp_file_bytes = b""
                    geom_file_bytes = b""

            elif structure_code in ["s13-tlA-fs-a1+a2+a3", "s13-tlA-fs-a2"]:
                # First geom file only has many iters missing, so ignore first dir
                if j_idx == 0:
                    cstp_file_bytes = b""
                    geom_file_bytes = b""

            elif structure_code == "s13-tlA-fs-b2":
                # First run from first dir is in .castep file but not in .geom file
                if j_idx == 0:
                    cstp_file_bytes = b""
                    geom_file_bytes = b""

    else:

        cstp_path = structure_dir.joinpath("sim.castep.xz")
        cstp_file_bytes = decompress_file(cstp_path)

        cell_path = structure_dir.joinpath("sim.cell.xz")
        cell_file_bytes = decompress_file(cell_path)

        geom_path = structure_dir.joinpath("sim.geom.xz")
        if geom_path.is_file():
            geom_file_bytes = decompress_file(geom_path)

    cell_dat = read_cell_file(cell_file_bytes)
    cstp_dat = read_castep_file(cstp_file_bytes)
    cstp_dat = merge_cell_data(cstp_dat, cell_dat)

    if geom_file_bytes:
        geom_dat = read_geom_file(geom_file_bytes)
        cstp_dat = merge_geom_data(cstp_dat, geom_dat)

    # Special cases
    # -------------
    if structure_code == "s7-tw-gb-ghost-23":
        # Single point run repeated in .castep file
        del cstp_dat["runs"][1]
        del cstp_dat["SCF"]["cycles"][1]
        for k, v in cstp_dat["SCF"]["energies"].items():
            cstp_dat["SCF"]["energies"][k] = v[0:1]

    elif structure_code == "s19-tlA-gb":
        # Only include up to geometry iteration 29 where dE/ion = 7.00487e-7 eV / ion
        # After this, boundary seems to start to move (dislocation motion?).
        cstp_dat["geom"]["iterations"] = cstp_dat["geom"]["iterations"][:30]

    data = {}
    species = cstp_dat["structure"]["species"]

    if "geom" in cstp_dat:

        atoms = []
        supercell = []
        for geom_iter in cstp_dat["geom"]["iterations"]:

            final_step = geom_iter["steps"][-1]
            scf_idx = final_step["SCF_idx"]
            for en_name, en in cstp_dat["SCF"]["energies"].items():
                if en_name not in data:
                    data[en_name] = [en[scf_idx]]
                else:
                    data[en_name].append(en[scf_idx])

            if geom_iter.get("atoms") is not None:
                atoms.append(geom_iter["atoms"])
                supercell.append(geom_iter["cell"])
            else:
                print("no atoms in geom iter!")
                if geom_iter["iter_num"] == 0:
                    print("Warning: no atoms in geom iteration 0!")
                    # First run was missing, so use `structure` key (from cell file):
                    atoms.append(cstp_dat["structure"]["atoms"])
                    supercell.append(cstp_dat["structure"]["supercell"])
                else:
                    # Iteration is missing from the .geom file (due to structure reversion)
                    print("structure reversion?")
                    atoms.append(atoms[-1])
                    supercell.append(supercell[-1])

    else:
        data = cstp_dat["SCF"]["energies"]
        atoms = [cstp_dat["structure"]["atoms"]]
        supercell = [cstp_dat["structure"]["supercell"]]

    # Convert to arrays:
    atoms = np.array(atoms)

    supercell = np.array(supercell)
    for en_name in data:
        data[en_name] = np.array(data[en_name])

    # Make initial structure:
    # bicrystal = Bicrystal.from_atoms(
    #     atoms=np.copy(atoms[0]),
    #     species=species,
    #     supercell=supercell[0],
    #     non_boundary_idx=2,
    #     maintain_inv_sym=True,
    #     overlap_tol=parameters['overlap_tol'],
    #     wrap=True,
    # )

    # sim = AtomisticSimulation.from_bicrystal_parameters() -> but atomistic shouldn't 'know' about Bicrystal, so perhaps
    # BicrystalSimulation is better.

    sup_type = structure_code.split("-")[2]
    meta = {}
    if sup_type == "b":
        meta["supercell_type"] = ["bulk", "bulk_bicrystal"]
    elif sup_type == "fs":
        meta["supercell_type"] = ["surface", "surface_bicrystal"]

    iodine_idx = np.where(species == "I")[0]

    sim = atomistic_simulation_from_bicrystal_parameters(
        all_atoms=atoms,
        all_supercells=supercell,
        species=species,
        data=data,
        meta=meta,
        non_boundary_idx=2,
        maintain_inv_sym=True,
        overlap_tol=parameters["overlap_tol"],
        wrap=True,
    )

    iodine_idx_sim = np.where(sim.structure.atoms.species == "I")[0]

    return sim


def get_lammps_parameters(
    potential_file=DEF_POT_PATH, parameters_file=DEF_LAMMPS_FILE, base_path=None
):

    if base_path is None:
        base_path = ""
    base_path = Path(base_path)

    with Path(parameters_file).open() as handle:
        params = YAML().load(handle)

    # Substitute in the potential file:
    if not Path(potential_file).exists():
        raise ValueError('Potential file "{}" does not exist.'.format(potential_file))

    for idx, i in enumerate(params["interactions"]):
        if "<<POTENTIAL_PATH>>" in i:
            pot_path = base_path.joinpath(potential_file)
            params["interactions"][idx] = i.replace("<<POTENTIAL_PATH>>", str(pot_path))

    return params


def get_castep_parameters(num_atoms, parameters_file=DEF_CASTEP_FILE):

    with Path(parameters_file).open() as handle:
        params = YAML().load(handle)

    atom_const = params.get("atom_constraints")

    if atom_const is not None:

        valid_atom_cnst = {}
        for key, val in atom_const.items():

            if isinstance(val, list):
                valid_atom_cnst.update({key: np.array(val)})

            elif isinstance(val, str):

                if val.upper() == "NONE":
                    valid_atom_cnst.update({key: None})

                elif val.upper() == "ALL":
                    all_atm = np.arange(num_atoms) + 1
                    valid_atom_cnst.update({key: all_atm})

        params["atom_constraints"] = valid_atom_cnst

    return params


def get_structural_parameter_data(parameters_file=DEF_PARAMS_FILE):
    yaml = YAML()
    with Path(parameters_file).open(encoding="utf-8") as handle:
        parameters = yaml.load(handle)
    return parameters


def make_structure(
    structure_code,
    configuration="base",
    method="dft",
    parameters_file=DEF_PARAMS_FILE,
    dirs_list_path=DEFAULT_DIRS_LIST_PATH,
    shift=None,
    expansion=None,
):
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
    method : str
        One of "dft" or "eam". Optimised lattice parameter depend on the method of atomic
        relaxation. For the minimum energy structures, the relative shift and GB expansion
        also differ between methods.
    parameters_file : str or Path, optional
        Path to the YAML file containing structure parameters.
    dirs_list_path : str or Path, optional
        Path to the YAML file containing directories for simulation data.
    shift : list of length two, optional
        In-boundary-plane translation to apply (overrides value in parameter file) if
        configuration is "minimum_energy".
    expansion : float, optional
        GB expansion to apply (overrides value in parameter file) if configuration is
        "minimum_energy".

    """

    parameters = get_structural_parameter_data(parameters_file)

    sigma_code, gb_type, sup_type = structure_code.split("-")
    gb_code = "{}-{}".format(sigma_code, gb_type)

    crystal_structure = parameters["crystal_structures"][method]
    csl_vecs = parameters["csl_vecs"][sigma_code]
    box_csl = parameters["box_csl"][gb_type]

    if configuration in ["sized", "minimum_energy"]:
        box_csl *= np.array(parameters["csl_repeats"][gb_code])

    struct_params = {
        "crystal_structure": crystal_structure,
        "csl_vecs": csl_vecs,
        "box_csl": box_csl,
        "overlap_tol": parameters["overlap_tol"],
    }

    if sup_type == "gb":

        if configuration == "minimum_energy":

            shift = (
                shift
                if shift is not None
                else parameters["relative_shift"][method][gb_code]
            )
            if expansion is not None:
                expansion = [
                    {
                        "thickness": expansion,
                        "func": "sigmoid",
                        "sharpness": 1,
                    }
                ]
            else:
                expansion = [parameters["boundary_vac"][method][gb_code]]

            struct_params.update(
                {
                    "relative_shift": {
                        "shift": shift,
                        "crystal_idx": 0,
                    },
                    "boundary_vac": expansion,
                }
            )

        struct_params.update(
            {
                "maintain_inv_sym": True,
            }
        )

        out = bicrystal_from_csl_vectors(**struct_params)

    elif sup_type == "fs":
        out = surface_bicrystal_from_csl_vectors(**struct_params)

    elif sup_type == "b":
        struct_params["csl_vecs"] = struct_params["csl_vecs"][0]
        out = bulk_bicrystal_from_csl_vectors(**struct_params)

    return out


def decode_structure_info(structure_code):

    # TODO check values.
    THETA = {
        7: 21.79,
        13: 27.80,
        19: 13.1735511,
        31: 17.90,
    }

    key_split = structure_code.split("-")
    pristine_key = "-".join(key_split[:3])
    gb_type = "-".join(key_split[:2])
    sigma = int(key_split[0][1:])
    misorientation = THETA[sigma]

    info = {
        "sigma": sigma,
        "misorientation": misorientation,
        "pristine_key": pristine_key,
        "supercell_type": key_split[2],
        "gb_type": gb_type,
        "gb_plane": key_split[1],
    }

    return info


def load_structures(method, json_path):
    'Load the energies (and structures in the case of method="DFT") from a JSON file.'

    with Path(json_path).open() as handle:
        sims_data = json.load(handle)

    for structure_code in sims_data:

        sims_data[structure_code]["structure"]["atoms"] = np.array(
            sims_data[structure_code]["structure"]["atoms"]
        )

        sims_data[structure_code]["structure"]["supercell"] = np.array(
            sims_data[structure_code]["structure"]["supercell"]
        )

        sims_data[structure_code]["structure"]["species"] = np.array(
            sims_data[structure_code]["structure"]["species"]
        )

        sims_data[structure_code]["structure"][
            "atom_site_geometries"
        ] = unjsonified_site_geometries(
            sims_data[structure_code]["structure"]["atom_site_geometries"]
        )

        if "final_structure" in sims_data[structure_code]:
            sims_data[structure_code]["final_structure"]["atoms"] = np.array(
                sims_data[structure_code]["final_structure"]["atoms"]
            )

            sims_data[structure_code]["final_structure"]["supercell"] = np.array(
                sims_data[structure_code]["final_structure"]["supercell"]
            )

            sims_data[structure_code]["final_structure"]["species"] = np.array(
                sims_data[structure_code]["final_structure"]["species"]
            )

            sims_data[structure_code]["final_structure"][
                "atom_site_geometries"
            ] = unjsonified_site_geometries(
                sims_data[structure_code]["final_structure"]["atom_site_geometries"]
            )

    return sims_data


def export_structures(
    method,
    json_path=None,
    log_path="collation_log.txt",
    exclude_structures=None,
    include_structures=None,
):
    "Parse and export the structures and energies to a JSON file."

    method = method.upper()
    if not json_path:
        json_path = f"{method}_sims.json"

    sims_data = collate_structures(
        method,
        log_path,
        exclude_structures=exclude_structures,
        include_structures=include_structures,
    )

    with Path(json_path).open("w") as handle:
        json.dump(sims_data, handle, indent=2, sort_keys=True)

    return json_path


def get_bicrystal_thickness(supercell):
    """Get bicrystal thickness in grain boundary normal direction."""

    n_unit = np.array([[0, 0, 1]]).T
    sup_nb = supercell[:, 2][:, None]
    ein = np.einsum("ij,ij", sup_nb, n_unit)
    return ein


def jsonified_site_geometries(site_geoms, neighbour_keys=["area"]):

    out = {
        "neighbours": [],
        "volume": site_geoms["volume"].tolist(),
    }

    if "interface_distance" in site_geoms:
        out.update({"interface_distance": site_geoms["interface_distance"].tolist()})

    for i in site_geoms["neighbours"]:
        out["neighbours"].append(
            {k: v.tolist() for k, v in i.items() if k in neighbour_keys}
        )

    return out


def unjsonified_site_geometries(site_geoms):

    out = {
        "neighbours": [],
        "volume": np.array(site_geoms["volume"]),
    }

    if "interface_distance" in site_geoms:
        out.update({"interface_distance": np.array(site_geoms["interface_distance"])})

    for i in site_geoms["neighbours"]:
        out["neighbours"].append({k: np.array(v) for k, v in i.items()})

    return out


def collate_structures(
    method,
    log_path="collation_log.txt",
    exclude_structures=None,
    include_structures=None,
):
    """Collate final energies from all EAM or DFT simulations For DFT simulations,
    detailed structural information is also included.

    Parameters
    ----------
    method : str "EAM" or "DFT"
    ...

    Returns
    -------
    sims_data : dict of (str : dict)
        For each structure code, a dict containing simulation results.

    """

    if not exclude_structures:
        exclude_structures = []

    if not include_structures:
        include_structures = []

    log_path = Path(log_path)

    count = 0
    sims_data = {}
    t1 = time()

    method = method.lower()
    if method == "dft":
        data_path = "data/dft_sims"
    elif method == "eam":
        data_path = "data/eam_sims"

    for i in Path(data_path).glob("*"):

        # Here we should invoke get_simulation to get an AtomisticSimulation object; and
        # then save info from that object in JSON format.

        structure_code = i.name

        if include_structures:
            if structure_code not in include_structures:
                continue

        if structure_code in exclude_structures:
            continue

        count += 1

        print("________{:_<80s}".format(structure_code))

        structure_info = decode_structure_info(structure_code)

        if method == "dft":

            sim = get_simulation(structure_code)

            structure_info.update(
                {
                    "area": sim.structure.boundary_area,
                    "num_atoms": sim.structure.num_atoms,
                }
            )
            if structure_info["supercell_type"] == "gb":
                structure_info.update(
                    {
                        "bicrystal_thickness": sim.structure.bicrystal_thickness,
                    }
                )

            initial_structure = sim.get_step(0, atom_site_geometries=True)["structure"]
            initial_structure.swap_crystal_sites()
            sim_dict = {
                "structure_info": structure_info,
                "structure": {
                    "atoms": initial_structure.atoms.coords.tolist(),
                    "supercell": initial_structure.supercell.tolist(),
                    "species": initial_structure.atoms.species.tolist(),
                    "atom_site_geometries": jsonified_site_geometries(
                        initial_structure.atom_site_geometries
                    ),
                },
                "data": {
                    "final_zero_energy": [sim.data["final_zero_energy"][0]],
                },
            }

            if sim.num_steps > 1:
                sim_dict["data"]["final_zero_energy"].append(
                    sim.data["final_zero_energy"][-1]
                )
                final_structure = sim.get_step(-1, atom_site_geometries=True)[
                    "structure"
                ]
                final_structure.swap_crystal_sites()
                sim_dict.update(
                    {
                        "final_structure": {
                            "atoms": final_structure.atoms.coords.tolist(),
                            "supercell": final_structure.supercell.tolist(),
                            "species": final_structure.atoms.species.tolist(),
                            "atom_site_geometries": jsonified_site_geometries(
                                final_structure.atom_site_geometries
                            ),
                        }
                    }
                )

        elif method == "eam":

            lammps_out = read_lammps_output(dir_path=i)

            initial_structure = Bicrystal.from_atoms(
                atoms=lammps_out["atoms"][0],
                species=np.array(["Zr"] * lammps_out["atoms"][0].shape[1]),
                supercell=lammps_out["supercell"][0],
                non_boundary_idx=2,
            )
            initial_structure.set_atom_site_geometries()  # are atom site geometries then wrong?
            initial_structure.swap_crystal_sites()

            structure_info.update(
                {
                    "area": initial_structure.boundary_area,
                    "num_atoms": initial_structure.num_atoms,
                }
            )
            if structure_info["supercell_type"] == "gb":
                structure_info.update(
                    {
                        "bicrystal_thickness": initial_structure.bicrystal_thickness,
                    }
                )

            sim_dict = {
                "structure_info": structure_info,
                "structure": {
                    "atoms": initial_structure.atoms.coords.tolist(),
                    "supercell": initial_structure.supercell.tolist(),
                    "species": initial_structure.atoms.species.tolist(),
                    "atom_site_geometries": jsonified_site_geometries(
                        initial_structure.atom_site_geometries
                    ),
                },
                "data": {
                    "final_zero_energy": [lammps_out["final_energy"][0]],
                },
            }

            if lammps_out["atoms"].shape[0] > 1:
                final_structure = Bicrystal.from_atoms(
                    atoms=lammps_out["atoms"][-1],
                    species=np.array(["Zr"] * lammps_out["atoms"][0].shape[1]),
                    supercell=lammps_out["supercell"][0],
                    non_boundary_idx=2,
                )
                final_structure.set_atom_site_geometries()
                final_structure.swap_crystal_sites()  # are atom site geometries then wrong?

                sim_dict["data"]["final_zero_energy"].append(
                    lammps_out["final_energy"][-1]
                )

                sim_dict.update(
                    {
                        "final_structure": {
                            "atoms": final_structure.atoms.coords.tolist(),
                            "supercell": final_structure.supercell.tolist(),
                            "species": final_structure.atoms.species.tolist(),
                            "atom_site_geometries": jsonified_site_geometries(
                                final_structure.atom_site_geometries
                            ),
                        }
                    }
                )

        sims_data.update({structure_code: sim_dict})
        print()

    t2 = time()
    t_mins = (t2 - t1) / 60

    print("{:_<88}".format(""))
    print(
        f"Collated {method.upper()} results for {count} structures in "
        f"{t_mins:.2f} minutes."
    )
    print("{:_<88}".format(""))

    return sims_data


def get_interplanar_spacing_data(
    DFT_sims, structure_code, step, add_one, bulk_val, average_by=None
):

    geoms = DFT_sims[structure_code]["final_structure"]["atom_site_geometries"]

    int_dist = geoms["interface_distance"]
    int_dist *= -1
    int_dist_srt = np.sort(int_dist)

    if average_by:
        int_dist_srt = np.mean(np.reshape(int_dist_srt, (-1, average_by)), axis=1)

    w = np.where(int_dist_srt > 0)[0]
    w2 = [w[0] - 1] + list(w)

    int_dist_pos = int_dist_srt[w2][::step]
    int_dist_diff = np.diff(int_dist_pos)
    int_dist_diff = int_dist_diff[: int(len(int_dist_diff) / 2) + (1 if add_one else 0)]

    x_range = [0, len(int_dist_diff) - 1]

    out = {
        "structure_code": structure_code,
        "xrange": x_range,
        "y": int_dist_diff,
        "bulk_val": bulk_val,
    }

    return out


def get_local_volume_change_data(DFT_sims, structure_code, vol_bulk):

    final_structure = DFT_sims[structure_code]["final_structure"]
    geoms = final_structure["atom_site_geometries"]

    x = geoms["interface_distance"]
    y = geoms["volume"]

    x_srt_idx = np.argsort(x)
    x = x[x_srt_idx]
    y = y[x_srt_idx]

    supercell = final_structure["supercell"]
    range_mag = supercell[2, 2] / 2
    range_lims = [-range_mag / 2, range_mag / 2]

    x_rangd = np.where(
        np.logical_and(
            x >= range_lims[0],
            x <= range_lims[1],
        )
    )

    x = x[x_rangd]
    y = y[x_rangd]

    # Percentage change in local volume:
    y = 100 * (y - vol_bulk) / y

    out = {
        "structure_code": structure_code,
        "vol_change": y,
        "int_dist": x,
        "vol_bulk": vol_bulk,
    }

    return out


def get_coordination_change_data(DFT_sims, structure_code, area_threshold):

    final_structure = DFT_sims[structure_code]["final_structure"]
    neighbours = final_structure["atom_site_geometries"]["neighbours"]
    int_dist = final_structure["atom_site_geometries"]["interface_distance"]

    x = int_dist
    y = np.array([np.sum(i["area"] > area_threshold) for i in neighbours])

    x_srt_idx = np.argsort(x)
    x = x[x_srt_idx]
    y = y[x_srt_idx]

    supercell = final_structure["supercell"]
    range_mag = supercell[2, 2] / 2
    range_lims = [-range_mag / 2, range_mag / 2]

    x_rangd = np.where(
        np.logical_and(
            x >= range_lims[0],
            x <= range_lims[1],
        )
    )

    x = x[x_rangd]
    y = y[x_rangd]

    out = {
        "structure_code": structure_code,
        "coord": y,
        "int_dist": x,
    }

    return out


def get_interface_energy(interface_sim, bulk_sim):

    bulk_ratio = (
        interface_sim["structure_info"]["num_atoms"]
        / bulk_sim["structure_info"]["num_atoms"]
    )
    E_int = (
        interface_sim["data"]["final_zero_energy"][-1]
        - bulk_ratio * bulk_sim["data"]["final_zero_energy"][-1]
    ) / (2 * interface_sim["structure_info"]["area"])
    E_int *= UNIT_CONV
    return E_int


def get_all_interface_energies(DFT_sims, interface):

    if interface not in ["gb", "fs"]:
        raise ValueError('Interface must be one of "gb" or "fs".')

    E_int = {}

    # Search for pristine GBs:
    for key, val in DFT_sims.items():

        if key.endswith(interface):

            # Find complimentary bulk sim:
            bulk_key = key.split("-" + interface)[0] + "-b"

            if bulk_key in DFT_sims:
                E_int.update({key: get_interface_energy(val, DFT_sims[bulk_key])})

    return E_int


def get_wsep(pristine_sim, defective_sim, fs_defective_sim):
    """Calculate work of separation of GB or bulk supercell where, "defective"
    supercells have one or zero defective interfaces.

    Parameters
    ----------
    pristine_sim : dict
        Represents the pristine GB/bulk supercell
    defective_sim : dict
        Represents the defective GB/bulk supercell
    fs_defective_sim : dict
        Represents the defective FS supercell

    """

    wos = pristine_sim["data"]["final_zero_energy"][-1] + 2 * (
        fs_defective_sim["data"]["final_zero_energy"][-1]
        - defective_sim["data"]["final_zero_energy"][-1]
    )

    wos /= 2 * pristine_sim["structure_info"]["area"]
    wos *= UNIT_CONV

    return wos


def get_all_wsep(DFT_sims):

    wsep_all = {}

    # Search for GBs and Bulks:
    for key, val in DFT_sims.items():

        sup_type = val["structure_info"]["supercell_type"]

        if sup_type in ["gb", "b"]:

            # Get pristine GB/B key:
            key_split = key.split("-")
            prs_key = "-".join(key_split[:3])

            # Get "defective" FS key:
            key_split_fs = list(key_split)
            key_split_fs[2] = "fs"
            fs_def_key = "-".join(key_split_fs)

            if prs_key in DFT_sims and fs_def_key in DFT_sims:
                wsep = get_wsep(DFT_sims[prs_key], val, DFT_sims[fs_def_key])
                wsep_all.update({key: wsep})

    map_vacancy_ghosts(wsep_all)

    return wsep_all


def map_vacancy_ghosts(wsep_all):
    """Modify list of work of separations so that ghost vacancy keys are also
    included.

    If a calculation has only vacancy defects, then the work of separation of the
    fully relaxed supercell is identical to that of the ghost system "C/C_s" as
    defined in Lozovoi. Therefore, here, we add additional keys to `wsep_all`.
    For example:

    {
        ...,
        's7-tw-gb-v1':       `val1`,
        's7-tw-fs-v1':       `val2`,
        ...,
    }

    goes to

    {
        ...,
        's7-tw-gb-v1':       `val1`,
        's7-tw-gb-v1-ghost': `val1`,
        's7-tw-fs-v1':       `val2`,
        's7-tw-fs-v1-ghost': `val2`,
        ...,
    }

    """
    wsep_keys = list(wsep_all.keys())

    for key in wsep_keys:

        key_split = key.split("-")

        only_vacancies = False
        if len(key_split) > 3:
            def_list = key_split[3].split("+")
            is_vacancy = [i[0] == "v" for i in def_list]
            if all(is_vacancy):
                only_vacancies = True

        if only_vacancies:
            new_key = "-".join(key_split + ["ghost"])
            wsep_all.update({new_key: wsep_all[key]})


def plot_vol_change(vol_change):
    plot_data = []
    for i in vol_change:
        plot_data.extend(
            [
                {
                    "x": i["int_dist"],
                    "y": i["vol_change"],
                    "name": "{}".format(i["structure_code"]),
                    "mode": "markers",
                },
            ]
        )

    fig = graph_objects.FigureWidget(
        data=plot_data,
        layout={
            "width": 600,
            "height": 500,
            "xaxis": {
                "title": "Distance from GB /Ang.",
            },
            "yaxis": {
                "title": "Change in local volume",
            },
        },
    )
    return fig


def plot_coordination(coord_dat):

    plot_data = []
    for i in coord_dat:
        plot_data.extend(
            [
                {
                    "x": i["int_dist"],
                    "y": i["coord"],
                    "name": "{}".format(i["structure_code"]),
                    "mode": "markers",
                },
            ]
        )

    fig = graph_objects.FigureWidget(
        data=plot_data,
        layout={
            "width": 600,
            "height": 500,
            "xaxis": {
                "title": "Distance from GB /Ang.",
            },
            "yaxis": {
                "title": "Coordination",
            },
        },
    )
    return fig


def plot_interplanar_spacing_data(data):

    plot_data = []
    for i in data:
        plot_data.extend(
            [
                {"y": i["y"], "name": "{}".format(i["structure_code"])},
                {
                    "x": i["xrange"],
                    "y": [i["bulk_val"]] * 2,
                    "cliponaxis": False,
                    "mode": "lines",
                    "line": {
                        "width": 1,
                        "color": "gray",
                    },
                    "name": "Bulk ({})".format(i["structure_code"]),
                },
            ]
        )

    fig = graph_objects.FigureWidget(
        data=plot_data,
        layout={
            "width": 600,
            "height": 500,
            "xaxis": {
                "title": "Layer",
            },
            "yaxis": {
                "title": "Interplanar spacing /Ang.",
            },
        },
    )
    return fig


def plot_GB_energies(DFT_sims, E_GB):

    plot_dat = []
    for structure_code, value in E_GB.items():
        structure_info = DFT_sims[structure_code]["structure_info"]
        theta = structure_info["misorientation"]
        plot_dat.append({"x": [theta], "y": [value], "name": structure_code})
    fig = graph_objects.FigureWidget(
        data=plot_dat,
        layout={
            "width": 600,
            "height": 500,
            "xaxis": {
                "title": "Misorientation / deg.",
            },
            "yaxis": {
                "title": "E_GB / J/m^2",
            },
        },
    )
    return fig


def plot_FS_energies(DFT_sims, E_FS):

    plot_dat = []
    for structure_code, value in E_FS.items():
        structure_info = DFT_sims[structure_code]["structure_info"]
        theta = structure_info["misorientation"]
        plot_dat.append({"x": [theta], "y": [value], "name": structure_code})
    fig = graph_objects.FigureWidget(
        data=plot_dat,
        layout={
            "width": 600,
            "height": 500,
            "xaxis": {
                "title": "Misorientation / deg.",
            },
            "yaxis": {
                "title": "E_FS / J/m^2",
            },
        },
    )
    return fig


def plot_works_of_separation(DFT_sims, W):

    plot_dat = []
    for structure_code, value in W.items():
        structure_info = DFT_sims[structure_code]["structure_info"]
        theta = structure_info["misorientation"]
        plot_dat.append({"x": [theta], "y": [value], "name": structure_code})
    fig = graph_objects.FigureWidget(
        data=plot_dat,
        layout={
            "width": 600,
            "height": 500,
            "xaxis": {
                "title": "Misorientation / deg.",
            },
            "yaxis": {
                "title": "W_sep / J/m^2",
            },
        },
    )
    return fig


def get_GB_width(vol_change_i, width_vol_change_lims, range_lims):
    in_gb_idx_DFT = np.where(
        np.abs(vol_change_i["vol_change"]) >= width_vol_change_lims[1]
    )[0]
    out_gb_idx_DFT = np.where(
        np.abs(vol_change_i["vol_change"]) < width_vol_change_lims[1]
    )[0]

    if len(in_gb_idx_DFT) > 0:

        most_neg_gb = in_gb_idx_DFT[0]
        least_neg_non_gb = most_neg_gb - 1
        most_pos_gb = in_gb_idx_DFT[-1]
        least_pos_non_gb = most_pos_gb + 1

        if least_pos_non_gb > (len(vol_change_i["int_dist"]) - 1):
            gb_width_upper = range_lims[1]
        else:
            gb_width_upper = (
                vol_change_i["int_dist"][least_pos_non_gb]
                + vol_change_i["int_dist"][most_pos_gb]
            ) / 2

        if least_neg_non_gb < 0:
            gb_width_lower = range_lims[0]
        else:
            gb_width_lower = (
                vol_change_i["int_dist"][most_neg_gb]
                + vol_change_i["int_dist"][least_neg_non_gb]
            ) / 2

        vol_change_sym = -gb_width_upper / gb_width_lower

    else:
        gb_width_upper = 0
        gb_width_lower = 0
        vol_change_sym = 1

    gb_width = gb_width_upper - gb_width_lower

    return gb_width, gb_width_lower, gb_width_upper


def get_predicted_GB_widths(
    DFT_sims, vol_change_DFT, vol_change_EAM, width_vol_change_lims=None
):

    width_vol_change_lims = width_vol_change_lims or DEFAULT_WIDTH_VOLUME_CHANGE_LIMITS
    gb_widths_DFT = {}
    gb_widths_EAM = {}

    for vol_change_DFT_i in vol_change_DFT:

        code = vol_change_DFT_i["structure_code"]
        vol_change_EAM_i = [i for i in vol_change_EAM if i["structure_code"] == code][0]

        range_mag = (
            DFT_sims[code]["structure"]["supercell"][
                2,
                2,
            ]
            / 2
        )
        range_lims = [-range_mag / 2, range_mag / 2]

        gb_width_DFT, gb_width_lower_DFT, gb_width_upper_DFT = get_GB_width(
            vol_change_DFT_i, width_vol_change_lims, range_lims
        )
        gb_width_EAM, gb_width_lower_EAM, gb_width_upper_EAM = get_GB_width(
            vol_change_EAM_i, width_vol_change_lims, range_lims
        )

        gb_widths_DFT[code] = (gb_width_DFT, gb_width_lower_DFT, gb_width_upper_DFT)
        gb_widths_EAM[code] = (gb_width_EAM, gb_width_lower_EAM, gb_width_upper_EAM)

    return gb_widths_DFT, gb_widths_EAM


def write_manuscript_figures_coordination(
    DFT_sims, coord_change_dat_DFT, coord_change_dat_EAM
):

    pixels_per_unit = 4.1
    BULK_ALPHA_ZR_COORD = 12
    figs = []
    for coord_change_DFT_i in coord_change_dat_DFT:

        code = coord_change_DFT_i["structure_code"]
        coord_change_EAM_i = [
            i for i in coord_change_dat_EAM if i["structure_code"] == code
        ][0]

        range_mag = (
            DFT_sims[code]["structure"]["supercell"][
                2,
                2,
            ]
            / 2
        )
        range_lims = [-range_mag / 2, range_mag / 2]

        margins = {
            "l": 30,
            "r": 10,
            "t": 10,
            "b": 35,
        }
        plot_width = (pixels_per_unit * range_mag) + margins["l"] + margins["r"]
        com_axis = {
            "linecolor": "black",
            "linewidth": 0.7,
            "ticks": "inside",
            "tickwidth": 1,
            "mirror": "ticks",
            "gridwidth": 1,
            "showgrid": False,
            "zeroline": False,
            "tickformat": ".0f",
        }

        layout = {
            "template": "none",
            "width": plot_width,
            "showlegend": False,
            "height": 145,
            "margin": margins,
            "xaxis": {
                "range": range_lims,
                "dtick": 5,
                "tickangle": 0,
                "title": {
                    "text": r"\CCXLab{}",
                    "font": {
                        "size": 10,
                    },
                },
                **com_axis,
            },
            "yaxis": {
                "range": [9.5, 15.5],
                **com_axis,
                "title": {
                    "text": r"\CCYLab{}",
                    "font": {
                        "size": 10,
                    },
                },
            },
        }

        width_vol_lns_com = {
            "type": "scatter",
            "x": range_lims,
            "mode": "lines",
            "line": {
                "color": "gray",
                "dash": "dot",
                "width": 1,
            },
        }
        plot_data = [
            {
                "x": range_lims,
                "y": [BULK_ALPHA_ZR_COORD] * 2,
                "mode": "lines",
                "line": {
                    "width": 1.5,
                    "color": "gray",
                },
            },
            {
                "x": coord_change_DFT_i["int_dist"],
                "y": coord_change_DFT_i["coord"],
                "name": f"{code} DFT",
                "mode": "markers",
                "marker": {
                    "symbol": "circle-open",
                    "color": "black",
                    "size": 4,
                },
            },
            {
                "x": coord_change_EAM_i["int_dist"],
                "y": coord_change_EAM_i["coord"],
                "name": f"{code} EAM",
                "mode": "markers",
                "marker": {
                    "symbol": "circle",
                    "color": DEFAULT_PLOTLY_COLORS[1],
                    "size": 2.5,
                },
                "line": {
                    "width": 1.5,
                    "dash": "1px 1px",
                    "color": DEFAULT_PLOTLY_COLORS[1],
                },
            },
        ]
        fig = graph_objects.FigureWidget(
            data=plot_data,
            layout=layout,
        )
        fig.write_image(f"coord_change_{code}.svg")
        figs.append(fig)

    return figs


def write_manuscript_figures_volume_change(
    DFT_sims, vol_change_DFT, vol_change_EAM, width_vol_change_lims=None
):

    pixels_per_unit = 4.1
    figs = []
    width_vol_change_lims = width_vol_change_lims or DEFAULT_WIDTH_VOLUME_CHANGE_LIMITS
    gb_widths_DFT, gb_widths_EAM = get_predicted_GB_widths(
        DFT_sims, vol_change_DFT, vol_change_EAM, width_vol_change_lims
    )

    for vol_change_DFT_i in vol_change_DFT:

        code = vol_change_DFT_i["structure_code"]
        vol_change_EAM_i = [i for i in vol_change_EAM if i["structure_code"] == code][0]

        range_mag = (
            DFT_sims[code]["structure"]["supercell"][
                2,
                2,
            ]
            / 2
        )
        range_lims = [-range_mag / 2, range_mag / 2]

        margins = {
            "l": 30,
            "r": 10,
            "t": 10,
            "b": 35,
        }
        plot_width = (pixels_per_unit * range_mag) + margins["l"] + margins["r"]

        _, gb_width_lower_DFT, gb_width_upper_DFT = gb_widths_DFT[code]
        _, gb_width_lower_EAM, gb_width_upper_EAM = gb_widths_EAM[code]

        com_axis = {
            "linecolor": "black",
            "linewidth": 0.7,
            "ticks": "inside",
            "tickwidth": 1,
            "mirror": "ticks",
            "gridwidth": 1,
            "showgrid": False,
            "zeroline": False,
            "tickformat": ".0f",
        }

        layout = {
            "template": "none",
            "width": plot_width,
            "showlegend": False,
            "height": 145,
            "margin": margins,
            "xaxis": {
                "dtick": 5,
                "tickangle": 0,
                "range": range_lims,
                "title": {
                    "text": r"\volChangeXLab{}",
                    "font": {
                        #'size': 12,
                    },
                },
                **com_axis,
            },
            "yaxis": {
                "range": [-15, 20],
                **com_axis,
                "title": {
                    "text": r"\volChangeYLab{}",
                    "font": {
                        #'size': 12,
                    },
                },
            },
        }
        width_vol_lns_com = {
            "type": "scatter",
            "x": range_lims,
            "mode": "lines",
            "line": {
                "color": "gray",
                "dash": "dot",
                "width": 1,
            },
        }
        plot_data = [
            {
                "y": [
                    width_vol_change_lims[0],
                ]
                * 2,
                **width_vol_lns_com,
            },
            {
                "y": [
                    width_vol_change_lims[1],
                ]
                * 2,
                **width_vol_lns_com,
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [
                    gb_width_lower_DFT,
                ]
                * 2,
                "y": layout["yaxis"]["range"],
                "line": {
                    "color": "black",
                    "dash": "solid",
                    "width": 0.8,
                },
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [
                    gb_width_upper_DFT,
                ]
                * 2,
                "y": layout["yaxis"]["range"],
                "line": {
                    "color": "black",
                    "dash": "solid",
                    "width": 0.8,
                },
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [
                    gb_width_lower_EAM,
                ]
                * 2,
                "y": layout["yaxis"]["range"],
                "line": {
                    "color": DEFAULT_PLOTLY_COLORS[1],
                    "dash": "dot",
                    "width": 0.8,
                },
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": [
                    gb_width_upper_EAM,
                ]
                * 2,
                "y": layout["yaxis"]["range"],
                "line": {
                    "color": DEFAULT_PLOTLY_COLORS[1],
                    "dash": "dot",
                    "width": 0.8,
                },
            },
            {
                "x": vol_change_DFT_i["int_dist"],
                "y": vol_change_DFT_i["vol_change"],
                "name": f"{code} DFT",
                "mode": "markers",
                "marker": {
                    "symbol": "circle-open",
                    "color": "black",
                    "size": 4,
                },
            },
            {
                "x": vol_change_EAM_i["int_dist"],
                "y": vol_change_EAM_i["vol_change"],
                "name": f"{code} EAM",
                "mode": "markers",
                "marker": {
                    "symbol": "circle",
                    "color": DEFAULT_PLOTLY_COLORS[1],
                    "size": 2.5,
                },
                "line": {
                    "width": 1.5,
                    "dash": "1px 1px",
                    "color": DEFAULT_PLOTLY_COLORS[1],
                },
            },
        ]
        fig = graph_objects.FigureWidget(
            data=plot_data,
            layout=layout,
        )
        fig.write_image(f"vol_change_{code}.svg")
        figs.append(fig)

    return figs


def write_manuscript_figures_interplanar_spacing(
    DFT_sims, int_spacing_data_DFT, int_spacing_data_EAM
):

    pixels_per_unit = 4.1
    figs = []
    for int_space_DFT_i in int_spacing_data_DFT:
        code = int_space_DFT_i["structure_code"]
        xrange = int_space_DFT_i["xrange"]
        bulk_val = int_space_DFT_i["bulk_val"]
        y_DFT = int_space_DFT_i["y"]

        int_space_EAM_i = [
            i for i in int_spacing_data_EAM if i["structure_code"] == code
        ][0]
        y_EAM = int_space_EAM_i["y"]

        range_mag = (
            DFT_sims[code]["structure"]["supercell"][
                2,
                2,
            ]
            / 2
        )

        margins = {
            "l": 45 if "tw" in code else 38,
            "r": 10,
            "t": 10,
            "b": 35,
        }
        plot_width = (pixels_per_unit * range_mag) + margins["l"] + margins["r"]

        com_axis = {
            "linecolor": "black",
            "linewidth": 0.7,
            "ticks": "inside",
            "tickwidth": 1,
            "mirror": "ticks",
            "gridwidth": 1,
            "showgrid": False,
            "zeroline": False,
            #         'tickformat': '.0f',
        }
        yrange = [0, 1.5] if "tl" in code else [2.53, 2.77]
        ytickfmt = ".1f" if "tl" in code else ".2f"
        ydtick = 0.5 if "tl" in code else 0.1
        ytick0 = 0.0 if "tl" in code else 2.55
        layout = {
            "template": "none",
            "width": plot_width,
            "showlegend": False,
            "height": 145,
            "margin": margins,
            "xaxis": {
                "tickangle": 0,
                "title": {
                    "text": r"\IDXLab{}",
                    "font": {
                        #'size': 12,
                    },
                },
                **com_axis,
            },
            "yaxis": {
                "range": yrange,
                "dtick": ydtick,
                "tick0": ytick0,
                **com_axis,
                "tickformat": ytickfmt,
                "title": {
                    "text": r"\IDYLab{}",
                    "font": {
                        #'size': 12,
                    },
                },
            },
        }
        plot_data = [
            {
                "x": xrange,
                "y": [bulk_val] * 2,
                "cliponaxis": False,
                "mode": "lines",
                "line": {
                    "width": 1,
                    "color": "gray",
                },
                "name": "Bulk ({})".format(code),
            },
            {
                "y": y_DFT,
                "name": "{}".format(code),
                "mode": "lines",
                "line": {
                    "width": 1,
                    "color": "black",
                },
            },
            {
                "y": y_EAM,
                "name": "{}".format(code),
                "mode": "lines",
                "line": {"width": 1.5, "color": "orange", "dash": "dot"},
            },
        ]
        fig = graph_objects.FigureWidget(
            data=plot_data,
            layout=layout,
        )
        fig.write_image(f"int_spacing_{code}.svg")
        figs.append(fig)

    return figs


def write_manuscript_figure_E_GB_DFT_vs_EAM(E_GB_EAM, E_GB_DFT, latex_labels=False):

    plt_x_DFT_tlA = []
    plt_y_EAM_tlA = []
    plt_x_DFT_tlB = []
    plt_y_EAM_tlB = []
    plt_x_DFT_tw = []
    plt_y_EAM_tw = []

    annots = []

    annot_offsets = {
        "s7-tw": [-20, 0],
        "s13-tw": [0, 30],
        "s19-tw": [0, -30],
        "s7-tlA": [0, 0],
        "s13-tlA": [0, 0],
        "s19-tlA": [0, 40],
        "s31-tlA": [20, 0],
        "s7-tlB": [0, -30],
    }
    annots_show_line = {
        "s7-tw": True,
        "s13-tw": True,
        "s19-tw": True,
        "s7-tlA": False,
        "s13-tlA": False,
        "s19-tlA": False,
        "s31-tlA": False,
        "s7-tlB": False,
    }
    annots_shift = {
        "s7-tw": [0, 0],
        "s13-tw": [0, 0],
        "s19-tw": [0, 0],
        "s7-tlA": [0, -15],
        "s13-tlA": [0, 13],
        "s19-tlA": [0, -15],
        "s31-tlA": [16, -1],
        "s7-tlB": [-4, 16],
    }

    for idx, (key, GB_energy) in enumerate(E_GB_DFT.items()):
        x = GB_energy
        y = E_GB_EAM[key]

        key_short = key[:-3]
        label = key_short

        if "-tlA" in key_short:
            plt_x_DFT_tlA.append(x)
            plt_y_EAM_tlA.append(y)
        elif "tlB" in key_short:
            plt_x_DFT_tlB.append(x)
            plt_y_EAM_tlB.append(y)
        elif "-tw" in key_short:
            plt_x_DFT_tw.append(x)
            plt_y_EAM_tw.append(y)

        sigma = key_short.split("-")[0].lstrip("s")

        annot_i = {
            "text": key_short if not latex_labels else f"\\sig{{{sigma}}}",
            "arrowhead": 0,
            "standoff": 2,
            "arrowwidth": 0.5,
            "showarrow": annots_show_line[key_short],
            "font": {
                "size": 7,
            },
            "xshift": annots_shift[key_short][0],
            "yshift": annots_shift[key_short][1],
            "x": x,
            "y": y,
            "ax": annot_offsets[key_short][0],
            "ay": annot_offsets[key_short][1],
        }
        if key_short == "s13-tw":
            annot_i["font"]["size"] = 9
        annots.append(annot_i)

    plt_data = [
        {
            "type": "scatter",
            "x": plt_x_DFT_tlA,
            "y": plt_y_EAM_tlA,
            "mode": "markers+text",
            "showlegend": False,
            "marker": {
                "color": "black",
                "symbol": "cross",
                "size": 6,
            },
        },
        {
            "type": "scatter",
            "x": plt_x_DFT_tlB,
            "y": plt_y_EAM_tlB,
            "mode": "markers+text",
            "showlegend": False,
            "marker": {
                "color": "black",
                "symbol": "cross-open",
                "size": 6,
            },
        },
        {
            "type": "scatter",
            "x": plt_x_DFT_tw,
            "y": plt_y_EAM_tw,
            "mode": "markers+text",
            "showlegend": False,
            "marker": {
                "color": "navy",
                "symbol": "circle",
                "size": 6,
            },
        },
    ]

    MIN_RANGE = 0
    MAX_RANGE = 0.9
    com_axis = {
        "range": [MIN_RANGE, MAX_RANGE],
        "dtick": 0.2,
        "linecolor": "black",
        "linewidth": 0.7,
        "ticks": "inside",
        "tickwidth": 1,
        "mirror": "ticks",
        "gridwidth": 1,
        "showgrid": False,
        "zeroline": False,
        "tickformat": ".1f",
    }
    layout = {
        "width": 180,
        "height": 180,
        "template": "none",
        "yaxis": {
            "title": {
                "text": "E_GB (EAM) / Jm^-2"
                if not latex_labels
                else r"\en{GB} \text{(EAM)}/ \jpermsq{}",
                "font": {
                    "size": 8,
                },
            },
            "scaleanchor": "x",
            **com_axis,
        },
        "xaxis": {
            "title": {
                "text": "E_GB (DFT-PBE) / Jm^-2"
                if not latex_labels
                else r"\en{GB} \text{(DFT-PBE)}/ \jpermsq{}",
                "font": {
                    "size": 8,
                },
            },
            **com_axis,
        },
        "shapes": [
            {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": MIN_RANGE,
                "y0": MIN_RANGE,
                "x1": MAX_RANGE,
                "y1": MAX_RANGE,
                "line": {"color": "gray"},
                "layer": "below",
            },
        ],
        "annotations": annots,
        "margin": {
            "t": 10,
            "r": 10,
            "b": 40,
            "l": 40,
        },
    }
    fig = graph_objects.FigureWidget(data=plt_data, layout=layout)
    fig.write_image("E_GB_EAM_vs_DFT.svg")

    return fig


def get_binned_structures(
    structure_codes, densities_base_path=None, den_file=None, decompress=True
):
    if not densities_base_path:
        densities_base_path = "data/dft_sims"
    if not den_file:
        den_file = "sim.den_fmt" if not decompress else "sim.den_fmt.xz"

    out = {}
    for i in structure_codes:

        bc = get_simulated_bicrystal(i, opt_idx=-1)
        bc.set_voronoi_tessellation()
        bc.set_bond_regions(method="polyhedron")

        den_fmt_path = Path(densities_base_path).joinpath(i, den_file)
        if decompress:
            den_fmt_path = decompress_file(den_fmt_path)

        density_dat = read_den_fmt(den_fmt_path)

        print('Adding volumetric data: "charge_density"...')
        bc.add_volumetric_data(
            "charge_density", density_dat["grid_size"], density_dat["density"]
        )

        # print('Binning volumetric data "charge_density" using method "grid"...')
        # bc.bin_volumetric_data(name='charge_density', method='grid', grid_size=[1, 1, bc.num_atoms])
        # bc.bin_volumetric_data(name='charge_density', method='grid', grid_size=[1, 1, 1])

        print(
            'Binning volumetric data "charge_density" using method "atoms_voronoi"...'
        )
        bc.bin_volumetric_data(name="charge_density", method="atoms_voronoi")

        # print('Binning volumetric data "charge_density" using method "bonds_polyhedron"...')
        # bc.bin_volumetric_data(name='charge_density', method='bonds_polyhedron')

        out.update({i: bc})

    return out


def get_atoms_num_electrons_by_voronoi(bicrystal, crystal_idx=0):
    """By summing the number of electrons found in atom Voronoi region, count
    the number of electrons associated with each atom in a bicrystal crystal.

    """

    crystal_atoms = np.where(bicrystal.atoms.crystal_idx == crystal_idx)[0]
    num_grid_points = np.product(
        bicrystal.volumetric_data["charge_density"]["grid_size"]
    )
    num_electrons_by_atom = {
        atom_idx: (binned_data["bin_value"] / num_grid_points)
        for atom_idx, binned_data in enumerate(
            bicrystal.tessellation.binned_volumetric_data["charge_density"]
        )
        if atom_idx in crystal_atoms
    }

    return num_electrons_by_atom
