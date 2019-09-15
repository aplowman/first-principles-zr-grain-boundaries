from pathlib import Path

from atomistic.tensile_test import AtomisticTensileTest
from plotly import graph_objects

from utilities import get_simulated_bicrystal


def get_tensile_test(structure_code):

    bicrystal = get_simulated_bicrystal(structure_code)

    tt_path = Path('data/processed/ab_initio_tensile_tests/rgs-{}.json'.format(
        structure_code))

    tt = AtomisticTensileTest.from_json_file(bicrystal, tt_path)

    return tt


def show_traction_separation(tensile_test, ts_data_name='ts_energy'):

    ts_plot_data = tensile_test.get_traction_separation_plot_data(ts_data_name)
    fig = graph_objects.FigureWidget(
        data=[
            {
                **ts_plot_data,
                'mode': 'markers+lines',
                'marker': {},
                'line': {
                    'width': 0.5,
                }
            },
        ],
        layout={
            'width': 400,
            'height': 400,
            'xaxis': {
                'title': 'Separation distance /Ang'
            },
            'yaxis': {
                'title': 'TS energy / Jm<sup>-2</sup>',
            }
        }
    )

    return fig


def show_multiple_traction_separation(tensile_tests, ts_data_name='ts_energy'):
    'Overlay multiple traction separation curves.'

    ts_plot_data = {name: tt.get_traction_separation_plot_data(ts_data_name)
                    for name, tt in tensile_tests.items()}

    fig = graph_objects.FigureWidget(
        data=[
            {
                **i,
                'name': name,
                'mode': 'markers+lines',
                'marker': {},
                'line': {'width': 0.5}
            }
            for name, i in ts_plot_data.items()
        ],
        layout={
            'width': 400,
            'height': 400,
            'xaxis': {
                'title': 'Separation distance /Ang'
            },
            'yaxis': {
                'title': 'TS energy / Jm<sup>-2</sup>',
            }
        }
    )

    return fig
