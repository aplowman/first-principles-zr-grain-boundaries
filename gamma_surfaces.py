from pathlib import Path
from subprocess import run, PIPE

from plotly import graph_objects
from ruamel.yaml import YAML
from atomistic.bicrystal import GammaSurface
from lammps_parse import write_lammps_inputs, read_lammps_output
from tqdm import tqdm

from utilities import make_structure, get_lammps_parameters
from utilities import DEFAULT_DIRS_LIST_PATH, DEF_PARAMS_FILE

LAMMPS_EXECUTABLE = 'lmp_serial.exe'
DEF_GS_GRID_FILE = 'parameters/gamma_surfaces.yml'
UNIT_CONV = 16.02176565  # For conversion from eV / Ang^2 --> J / m^2


def get_gamma_surface(structure_code, method='eam'):

    bicrystal = make_structure(
        structure_code + '-gb',
        configuration='sized',
        method=method,
        parameters_file=DEF_PARAMS_FILE,
        dirs_list_path=DEFAULT_DIRS_LIST_PATH,
    )
    gs_path = Path('data/processed/gamma_surfaces/{}/{}.json'.format(
        method, structure_code))

    gs = GammaSurface.from_json_file(bicrystal, gs_path)

    return gs


def show_master_gamma_surface(gamma_surface, data_name='energy'):

    master_plot_data = gamma_surface.get_fitted_surface_plot_data(
        data_name, xy_as_grid=False)
    grid_dat = gamma_surface.get_xy_plot_data()
    fig = graph_objects.FigureWidget(
        data=[
            {
                'type': 'contour',
                'colorscale': 'viridis',
                'colorbar': {
                    'title': data_name,
                },
                **master_plot_data,
            },
            {
                **grid_dat,
                'mode': 'markers',
                'marker': {
                    'size': 2,
                },
                'showlegend': True,
            },
        ],
        layout={
            'xaxis': {
                'scaleanchor': 'y',
            }
        }
    )
    return fig


def show_gamma_surface_fit(gamma_surface, shift, data_name='energy'):

    fit_plot_dat = gamma_surface.get_fit_plot_data(data_name, shift)
    fig = graph_objects.FigureWidget(
        data=[
            {
                **fit_plot_dat['fitted_data'],
                'name': 'Fit',
            },
            {
                **fit_plot_dat['data'],
                'name': data_name,
            },
            {
                **fit_plot_dat['minimum'],
                'name': 'Fit min.',
            },
        ],
        layout={
            'xaxis': {
                'title': 'Expansion',
            },
            'yaxis': {
                'title': data_name,
            },
            'width': 400,
            'height': 400,
        }
    )
    return fig


def compute_master_gamma_surface(structure_code, sims_dir, grid_size_file=DEF_GS_GRID_FILE):
    'Construct inputs, run Lammps sims, collate outputs, fit and save to a JSON file.'

    bicrystal = make_structure(
        f'{structure_code}-gb',
        method='eam',
        configuration='sized',
    )
    bulk = make_structure(
        f'{structure_code}-b',
        method='eam',
        configuration='sized',
    )

    with Path(DEF_GS_GRID_FILE).open() as handle:
        gs_params = YAML().load(handle)[f'{structure_code}-gb']

    gamma_surface = GammaSurface.from_grid(bicrystal, **gs_params)

    sims_dir = Path(sims_dir)
    pot_base_dir = Path('.').resolve()
    GB_input_paths = []

    common_lammps_params = get_lammps_parameters(base_path=pot_base_dir)

    print('Writing simulation input files...')
    with tqdm(total=len(gamma_surface)) as pbar:

        for idx, i in enumerate(gamma_surface.all_coordinates()):

            # Make a directory for this sim:
            sim_path = sims_dir.joinpath(f'{structure_code}-gb', i.coordinate_fmt)
            sim_path.mkdir(parents=True)

            # Write simulation inputs:
            lammps_params_gb_i = {
                'supercell': i.structure.supercell,
                'atom_sites': i.structure.atoms.coords,
                'species': i.structure.atoms.labels['species'].unique_values,
                'species_idx': i.structure.atoms.labels['species'].values_idx,
                'dir_path': sim_path,
                **common_lammps_params,
            }
            GB_input_paths.append(write_lammps_inputs(**lammps_params_gb_i))
            pbar.update()

    # Make a directory for the bulk sim:
    sim_path = sims_dir.joinpath(f'{structure_code}-b')
    sim_path.mkdir()

    # Write bulk simulation inputs:
    lammps_params_bulk = {
        'supercell': bulk.supercell,
        'atom_sites': bulk.atoms.coords,
        'species': bulk.atoms.labels['species'].unique_values,
        'species_idx': bulk.atoms.labels['species'].values_idx,
        'dir_path': sim_path,
        **common_lammps_params,
    }
    bulk_input_path = write_lammps_inputs(**lammps_params_bulk)

    print('Running simulations...')
    # Run sims:
    with tqdm(total=len(GB_input_paths + [bulk_input_path])) as pbar:
        for idx, i in enumerate(GB_input_paths + [bulk_input_path]):
            cmd = '{} < {}'.format(LAMMPS_EXECUTABLE, i.name)
            proc = run(cmd, shell=True, cwd=i.parent, stdout=PIPE, stderr=PIPE)
            pbar.update()

    print('Collating simulation outputs...')
    # Collate simulation outputs
    simulated_gamma_surface_params = {
        'shifts': [],
        'expansions': [],
        'data': {
            'energy': [],
            'grain_boundary_energy': [],
        },
        'metadata': {},
    }
    # First get the bulk sim data:
    lammps_out_bulk = read_lammps_output(dir_path=bulk_input_path.parent)
    E_tot_bulk = lammps_out_bulk['final_energy'][-1]
    simulated_gamma_surface_params['metadata'].update({
        'E_tot_bulk': E_tot_bulk,
        'grain_boundary_area': bicrystal.boundary_area,
        'num_atoms_bulk': bulk.num_atoms,
        'num_atoms_grain_boundary': bicrystal.num_atoms,
    })

    # Now iterate over GB sims:
    with tqdm(total=len(GB_input_paths)) as pbar:
        for idx, i in enumerate(GB_input_paths):

            shift_str, exp_str = i.parent.name.split('__')
            shift = []
            for j in shift_str.split('_'):
                num, denom = j.split('.')
                shift.append(int(num) / int(denom))

            simulated_gamma_surface_params['shifts'].append(shift)
            simulated_gamma_surface_params['expansions'].append(float(exp_str))

            lammps_out_GB_i = read_lammps_output(dir_path=i.parent)
            E_tot_GB_i = lammps_out_GB_i['final_energy'][-1]
            simulated_gamma_surface_params['data']['energy'].append(E_tot_GB_i)

            E_GB_i = (1 / (2 * bicrystal.boundary_area)) * (
                E_tot_GB_i - (bicrystal.num_atoms / bulk.num_atoms) * E_tot_bulk
            ) * UNIT_CONV
            simulated_gamma_surface_params['data']['grain_boundary_energy'].append(E_GB_i)
            pbar.update()

    # Generate a new GammaSurface:
    simulated_gamma_surface = GammaSurface(bicrystal, **simulated_gamma_surface_params)

    # Fit:
    simulated_gamma_surface.add_fit('energy', 3)
    simulated_gamma_surface.add_fit('grain_boundary_energy', 3)

    # Save results as a JSON:
    json_path = simulated_gamma_surface.to_json_file(
        'gamma_surface_data_eam_{}.json'.format(structure_code))

    print(f'γ-surface saved as a JSON file: {json_path}')

    return simulated_gamma_surface
