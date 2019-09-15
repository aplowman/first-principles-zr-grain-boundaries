from pathlib import Path
from subprocess import run, PIPE

from plotly import graph_objects
from ruamel.yaml import YAML
from atomistic.bicrystal import GammaSurface
from lammps_parse import write_lammps_inputs, read_lammps_output

from utilities import make_structure, get_lammps_parameters
from utilities import DEFAULT_DIRS_LIST_PATH, DEF_PARAMS_FILE

LAMMPS_EXECUTABLE = 'lmp_serial.exe'
DEF_GS_GRID_FILE = 'parameters/gamma_surfaces.yml'


def get_gamma_surface(structure_code, lattice_parameters='eam'):

    bicrystal = make_structure(
        structure_code,
        configuration='sized',
        lattice_parameters=lattice_parameters,
        parameters_file=DEF_PARAMS_FILE,
        dirs_list_path=DEFAULT_DIRS_LIST_PATH,
    )
    gs_path = Path('data/processed/gamma_surfaces/{}/{}.json'.format(
        lattice_parameters, structure_code))

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
                **master_plot_data,
            },
            {
                **grid_dat,
                'mode': 'markers',
                'marker': {
                    'size': 2,
                },
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
        structure_code,
        lattice_parameters='eam',
        configuration='sized'
    )

    with Path(DEF_GS_GRID_FILE).open() as handle:
        gs_params = YAML().load(handle)[structure_code]

    gamma_surface = GammaSurface.from_grid(bicrystal, **gs_params)

    sims_dir = Path(sims_dir)
    pot_base_dir = Path('.').resolve()
    input_paths = []

    common_lammps_params = get_lammps_parameters(base_path=pot_base_dir)

    # for i in [gamma_surface.get_coordinate(0)]:
    for i in gamma_surface.all_coordinates():

        # Make a directory for this sim:
        sim_path = sims_dir.joinpath('{}__{}'.format(i.shift_fmt, i.expansion))
        sim_path.mkdir()

        # Write simulation inputs:
        lammps_params = {
            'supercell': i.structure.supercell,
            'atom_sites': i.structure.atoms.coords,
            'species': i.structure.atoms.labels['species'].unique_values,
            'species_idx': i.structure.atoms.labels['species'].values_idx,
            'dir_path': sim_path,
            **common_lammps_params,
        }
        input_paths.append(write_lammps_inputs(**lammps_params))

    # Run sims:
    for i in input_paths:
        cmd = 'lmp_serial < {}'.format(i.name)
        _ = run(cmd, shell=True, cwd=i.parent, stdout=PIPE, stderr=PIPE)

    # Collate simulation outputs
    simulated_gamma_surface_params = {
        'shifts': [],
        'expansions': [],
        'data': {
            'energy': []
        },
    }
    for i in input_paths:

        shift_str, exp_str = i.parent.name.split('__')
        shift = []
        for j in shift_str.split('_'):
            num, denom = j.split('.')
            shift.append(int(num) / int(denom))

        simulated_gamma_surface_params['shifts'].append(shift)
        simulated_gamma_surface_params['expansions'].append(float(exp_str))

        lammps_out = read_lammps_output(dir_path=i.parent)
        simulated_gamma_surface_params['data']['energy'].append(
            lammps_out['final_energy'][-1])

    # Generate a new GammaSurface:
    simulated_gamma_surface = GammaSurface(bicrystal, **simulated_gamma_surface_params)

    # Fit:
    simulated_gamma_surface.add_fit('energy', 3)

    # Save results as a JSON:
    json_path = simulated_gamma_surface.to_json_file(
        'gamma_surface_data_eam_{}.json'.format(structure_code))

    return simulated_gamma_surface
