from collections import namedtuple
from pathlib import Path

import numpy as np
from atomistic.tensile_test import AtomisticTensileTest
from pandas import DataFrame
from plotly import graph_objects
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS

from utilities import get_simulated_bicrystal, load_structures, get_all_wsep


def get_tensile_test(structure_code):

    bicrystal = get_simulated_bicrystal(structure_code)

    tt_path = Path(
        "data/processed/ab_initio_tensile_tests/rgs-{}.json".format(structure_code)
    )

    tt = AtomisticTensileTest.from_json_file(bicrystal, tt_path)

    return tt


def show_traction_separation(tensile_test, ts_data_name="ts_energy"):

    ts_plot_data = tensile_test.get_traction_separation_plot_data(ts_data_name)
    fig = graph_objects.FigureWidget(
        data=[
            {
                **ts_plot_data,
                "mode": "markers+lines",
                "marker": {},
                "line": {
                    "width": 0.5,
                },
            },
        ],
        layout={
            "width": 400,
            "height": 400,
            "xaxis": {"title": "Separation distance /Ang"},
            "yaxis": {
                "title": "TS energy / Jm<sup>-2</sup>",
            },
        },
    )

    return fig


def show_multiple_traction_separation(
    tensile_test_keys,
    ts_data_name="ts_energy",
    show_stress=False,
    latex_labels=False,
    structures_json_path="data/processed/DFT_sims.json",
    traction_threshold=20,
    positive_only=True,
    split_legend=False,
    show_unfiltered_stress=False,
    subplot_width=400,
    subplot_height=400,
    max_stress_limit=20,
):
    "Overlay multiple traction separation curves."

    DFT_sims = load_structures(structures_json_path)
    w_sep_all = get_all_wsep(DFT_sims)

    rows = 2 if show_stress else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    markers = {
        "Pristine": {"symbol": "circle-open", "color": "black", "size": 4},
        "Iodine": {"symbol": "diamond", "color": "navy", "size": 4},
    }
    lines = {
        "Pristine": {"width": 0.5, "color": "black"},
        "Iodine": {"width": 0.5, "dash": "5px 5px", "color": "navy"},
    }

    for name, key in tensile_test_keys.items():

        tt = get_tensile_test(key)
        trac_sep_data = tt.get_traction_separation_plot_data()
        stress_data = tt.get_stress_data(
            w_sep_all[key], stress_threshold=traction_threshold
        )

        ts_x = trac_sep_data["x"]
        ts_y = trac_sep_data["y"]
        stress_x = stress_data["stress_x"]
        stress_y = stress_data["stress_y"]
        stress_x_filtered = stress_data["stress_x_filtered"]
        stress_y_filtered = stress_data["stress_y_filtered"]

        if positive_only:

            pos_idx = np.where(trac_sep_data["x"] >= 0)[0]
            ts_x = ts_x[pos_idx]
            ts_y = ts_y[pos_idx]
            stress_x = stress_x[pos_idx]
            stress_y = stress_y[pos_idx]

            filt_pos_idx = np.where(stress_x_filtered >= 0)[0]
            stress_x_filtered = stress_x_filtered[filt_pos_idx]
            stress_y_filtered = stress_y_filtered[filt_pos_idx]

        trace_name = name
        if split_legend:
            trace_name += " - traction-separation curve"

        fig.add_scatter(
            x=ts_x,
            y=ts_y,
            mode="markers+lines",
            marker=markers[name],
            line=lines[name],
            row=1,
            col=1,
            name=trace_name,
            legendgroup=trace_name,
        )

        if show_stress:

            if show_unfiltered_stress:
                trace_name = name
                if split_legend:
                    trace_name += " - central diff."
                fig.add_scatter(
                    x=stress_x,
                    y=stress_y,
                    mode="markers+lines",
                    marker={
                        **markers[name],
                        "opacity": 0.5,
                    },
                    line=lines[name],
                    row=2,
                    col=1,
                    name=trace_name,
                    legendgroup=trace_name,
                    showlegend=False if not split_legend else True,
                )

            trace_name = name
            if split_legend:
                trace_name += " - central diff. (filtered)"
            fig.add_scatter(
                x=stress_x_filtered,
                y=stress_y_filtered,
                mode="markers",
                marker={
                    **markers[name],
                    "opacity": 0.3,
                },
                row=2,
                col=1,
                name=trace_name,
                legendgroup=trace_name,
                showlegend=False if not split_legend else True,
            )

            trace_name = name
            if split_legend:
                trace_name += " - traction triangle"
            fig.add_scatter(
                x=[
                    0,
                    stress_data["separation_critical"],
                    stress_data["separation_final"],
                ],
                y=[0, stress_data["stress_critical"], 0],
                mode="lines",
                line={
                    **lines[name],
                    "width": 2,
                },
                row=2,
                col=1,
                name=trace_name,
                legendgroup=trace_name,
                showlegend=False if not split_legend else True,
            )

    com_ax_props = {
        "linecolor": "black",
        "linewidth": 0.7,
        "ticks": "inside",
        "tickwidth": 1,
        "mirror": "ticks",
        "gridwidth": 1,
        "showgrid": False,
        "tickfont": {
            "size": 12,
        },
    }

    if latex_labels:
        x_lab = r"Separation distance, d /\angs{}"
        y1_lab = r"\eint{} /\jpermsq{}"
        y2_lab = r"Traction /\SI[]{}{\giga\pascal}"
    else:
        x_lab = "Separation distance /Ang"
        y1_lab = "TS energy / Jm<sup>-2</sup>"
        y2_lab = "Traction / GPa"

    axis_label_font = {
        "font": {
            "size": 10,
        }
    }

    fig.update_layout(
        template=None,
        height=subplot_height * 1.8 if show_stress else subplot_height,
        width=subplot_width,
        xaxis={
            **com_ax_props,
            "title": {
                "text": x_lab,
                **axis_label_font,
            },
            "range": [0, 6.4],
        },
        yaxis={
            **com_ax_props,
            "title": {
                "text": y1_lab,
                **axis_label_font,
            },
            "range": [-3.8, 0.25],
        },
        legend={
            "x": 0.99,
            "y": 0.51,
            "xanchor": "right",
            "yanchor": "bottom",
            "tracegroupgap": 1,
            "bgcolor": "rgba(255,255,255,0)",
        },
        # paper_bgcolor='pink',
        margin={
            "t": 25,
            "l": 50,
            "r": 10,
            "b": 50,
        },
    )

    if show_stress:

        fig.update_layout(
            xaxis={"title": None},
            xaxis2={
                **com_ax_props,
                "title": {
                    "text": x_lab,
                    **axis_label_font,
                },
                "range": [0, 6.4],
            },
            yaxis2={
                **com_ax_props,
                "title": {
                    "text": y2_lab,
                    **axis_label_font,
                },
                "range": [0, max_stress_limit],
            },
        )

    return fig


def fit_poly(x, y, degree=5):
    "Fit using an n-degree polynomial"
    p1d = np.poly1d(np.polyfit(x, y, degree))
    yfit = p1d(x)
    pder = np.polyder(p1d)
    yder = pder(x)
    out = {
        "fit_y": yfit,
        "fit_dy_dx": yder,
    }

    return out


def fit_quadratic(x, y):
    return fit_poly(x, y, degree=2)


TractionSeparationTriangle = namedtuple(
    "TractionSeparationTriangle",
    ["critical_traction", "critical_separation", "final_separation"],
)

TensileTestFit = namedtuple(
    "TensileTestFit",
    [
        "tensile_test",
        "fit_limit",
        "fit_x",
        "fit_y",
        "fit_dy_dx",
        "traction_separation_triangle",
        "traction_separation_energy",
    ],
)


def get_trac_sep_triangle(traction, separation, w_sep):

    sep_srt_idx = np.argsort(separation)
    sep_srt = separation[sep_srt_idx]
    traction_srt = traction[sep_srt_idx]

    threshold = 4
    traction_thrsh = traction[separation < threshold]
    separation_thrsh = separation[separation < threshold]

    max_stress_idx = np.argmax(traction_thrsh)
    max_stress_x = separation_thrsh[max_stress_idx]
    max_stress_y = traction_thrsh[max_stress_idx]

    traction_critical = max_stress_y
    sep_critical = max_stress_x
    sep_final = (2 * w_sep) / (traction_critical / 10)

    return TractionSeparationTriangle(traction_critical, sep_critical, sep_final)


def fit_tensile_test(structure_code, W_sep, fit_limit):

    tt = get_tensile_test(structure_code)
    en_sep = tt.get_traction_separation_plot_data()
    fit_filter = en_sep["x"] <= fit_limit
    fit_x = en_sep["x"][fit_filter]
    fit_y = en_sep["y"][fit_filter]
    fit = fit_quadratic(fit_x, fit_y)
    tri = get_trac_sep_triangle(fit["fit_dy_dx"], fit_x, W_sep[structure_code])
    return TensileTestFit(
        tt, fit_limit, fit_x, fit["fit_y"], fit["fit_dy_dx"], tri, en_sep
    )


def show_fitted_tensile_test(
    fitted_tensile_tests, fig_name=None, styles=None, layout=None
):

    if not styles:
        styles = [
            {
                "mode": "markers",
                "marker": {"symbol": "circle-open", "color": "black", "size": 4},
            },
            {
                "mode": "lines",
                "line": {
                    "width": 1,
                    "color": "black",
                },
            },
            {
                "mode": "markers",
                "marker": {"symbol": "diamond", "color": "navy", "size": 4},
            },
            {
                "mode": "lines",
                "line": {
                    "width": 1,
                    "color": "black",
                },
            },
        ]
    data = []
    for idx, (k, v) in enumerate(fitted_tensile_tests.items()):
        fit_name = f"Fit"
        ts_name = f"{k.title()}"
        data.extend(
            [
                {
                    "x": v.fit_x,
                    "y": v.fit_y,
                    "name": fit_name,
                    "legendgroup": fit_name,
                    "showlegend": not bool(data),
                    **styles[2 * idx + 1],
                },
                {
                    "x": v.traction_separation_energy["x"],
                    "y": v.traction_separation_energy["y"],
                    "name": ts_name,
                    "legendgroup": "ts_data",
                    "mode": "markers",
                    **styles[2 * idx],
                },
            ]
        )

    x_lab = "Separation distance /Ang"
    y1_lab = "TS energy / Jm<sup>-2</sup>"

    layout = {
        "width": 500,
        "legend": {
            "xanchor": "right",
            "traceorder": "grouped",
            "x": 0.99,
            "y": 0.1,
            "yanchor": "bottom",
            "tracegroupgap": 1,
            "bgcolor": "rgba(255,255,255,0)",
        },
        "xaxis": {
            "range": [0, 6.4],
            "title": {
                "text": x_lab,
            },
        },
        "yaxis": {
            "title": "Interface energy / J/m^2",
            "tickformat": ".1f",
            "range": [-3.8, 0.25],
            "title": {
                "text": y1_lab,
            },
        },
        **(layout or {}),
    }
    if fig_name:
        layout["title"] = fig_name
    fig = graph_objects.FigureWidget(
        data=data,
        layout=layout,
    )
    return fig


def tabulate_TS_results(fits, W_sep):

    codes = [
        "s7-tlA-gb",
        "s7-tlA-b",
        "s7-tw-gb",
        "s7-tw-b",
    ]

    dat = {}
    for code in codes:
        def_code = code + "-a1"
        fit_prs = fits[code]
        fit_def = fits[def_code]

        sig_c_change = (fit_def.fit_dy_dx[-1] * 10) - (fit_prs.fit_dy_dx[-1] * 10)
        sig_c_change_pc = (sig_c_change / (fit_prs.fit_dy_dx[-1] * 10)) * 100

        final_sep_change = (
            fit_def.traction_separation_triangle.final_separation / 10
            - fit_prs.traction_separation_triangle.final_separation / 10
        )
        final_sep_change_pc = (
            final_sep_change
            / fit_prs.traction_separation_triangle.final_separation
            / 10
        ) * 100

        w_prs = W_sep[code]
        w_def = W_sep[def_code]
        w_change = w_def - w_prs
        w_change_pc = 100 * w_change / w_prs

        dat_i = {
            "σ_c[pris.] (GPa)": fit_prs.fit_dy_dx[-1] * 10,
            "σ_c[def.] (GPa)": fit_def.fit_dy_dx[-1] * 10,
            "Δσ_c (GPa)": sig_c_change,
            "Δσ_c (%)": sig_c_change_pc,
            "Δd_f (Ang.)": final_sep_change,
            "Δd_f (%)": final_sep_change_pc,
            "W[pris.] (J/m^2)": w_prs,
            "W[def.] (J/m^2)": w_def,
            "ΔW (J/m^2)": w_change,
            "ΔW (%)": w_change_pc,
        }
        dat[code] = dat_i

    df = DataFrame.from_dict(dat).T.round(2)
    return df


def write_manuscript_figures_tensile_tests(fitted_tensile_tests):
    ax_com = {
        "linecolor": "black",
        "linewidth": 0.7,
        "ticks": "inside",
        "tickwidth": 1,
        "mirror": "ticks",
        "gridwidth": 1,
        "showgrid": False,
        "tickfont": {
            "size": 12,
        },
    }

    axis_label_font = {
        "font": {
            "size": 10,
        }
    }
    layout = {
        "margin": {
            "t": 25,
            "l": 50,
            "r": 10,
            "b": 50,
        },
        "width": 220,
        "height": 280,
        "template": "none",
        "xaxis": {
            **ax_com,
            "title": {
                "text": r"Separation distance, d /\angs{}",
                **axis_label_font,
            },
            "range": [0, 6.4],
        },
        "yaxis": {
            **ax_com,
            "title": "Interface energy / J/m^2",
            "tickformat": ".1f",
            "range": [-3.8, 0.25],
            "title": {
                "text": r"\eint{} /\jpermsq{}",
                **axis_label_font,
            },
        },
    }
    codes = [
        "s7-tlA",
        "s7-tw",
    ]

    styles = [
        {
            "mode": "markers",
            "marker": {"symbol": "diamond", "color": "navy", "size": 4},
        },
        {
            "mode": "lines",
            "line": {
                "width": 1,
                "color": "black",
            },
        },
        {
            "mode": "markers",
            "marker": {"symbol": "circle-open", "color": "black", "size": 4},
        },
        {
            "mode": "lines",
            "line": {
                "width": 1,
                "color": "black",
            },
        },
    ]

    figs = []
    for fig_idx, code in enumerate(codes):

        gb_code = code + "-gb"
        b_code = code + "-b"
        gb_def_code = gb_code + "-a1"
        b_def_code = b_code + "-a1"
        gb_fit_tt_prs = fitted_tensile_tests[gb_code]
        gb_fit_tt_def = fitted_tensile_tests[gb_def_code]
        b_fit_tt_prs = fitted_tensile_tests[b_code]
        b_fit_tt_def = fitted_tensile_tests[b_def_code]

        gb_fig = show_fitted_tensile_test(
            fitted_tensile_tests={
                "def.     ": gb_fit_tt_def,
                "prist.     ": gb_fit_tt_prs,
            },
            styles=styles,
            layout=layout,
        )
        b_fig = show_fitted_tensile_test(
            fitted_tensile_tests={
                "def.     ": b_fit_tt_def,
                "prist.     ": b_fit_tt_prs,
            },
            styles=styles,
            layout=layout,
        )
        gb_dat = list(gb_fig.data)
        b_dat = []
        for i in b_fig.data:
            i.xaxis = "x2"
            i.yaxis = "y2"
            i.showlegend = False
            b_dat.append(i)

        data = gb_dat + b_dat

        margin_t = 50
        height = 260
        leg_space = 40
        if fig_idx > 0:
            for i_idx, i in enumerate(data):
                data[i_idx].showlegend = False
                _margin_t = margin_t - leg_space
                _height = height - leg_space
            margin_t = _margin_t
            height = _height

        annots = [
            {
                "text": r"Separation distance, d /\angs{}",
                "x": 0.5,
                "y": -0.22,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
            },
            {
                "text": "GB",
                "x": 3,
                "y": -3.5,
                "yanchor": "bottom",
                "xref": "x1",
                "yref": "y1",
                "showarrow": False,
            },
            {
                "text": "Bulk",
                "x": 3,
                "y": -3.5,
                "yanchor": "bottom",
                "xref": "x2",
                "yref": "y2",
                "showarrow": False,
            },
        ]
        domain_sep = 0.04
        yticks = [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5]
        layout = {
            "margin": {
                "t": margin_t,
                "l": 40,
                "r": 10,
                "b": 50,
            },
            "width": 290,
            "height": height,
            "template": "none",
            "annotations": annots,
            # "paper_bgcolor": "green",
            # "plot_bgcolor": "pink",
            "legend": {
                "xanchor": "center",
                "traceorder": "reversed",
                "itemwidth": 30,
                "x": 0.5,
                "y": 1.05,
                "yanchor": "bottom",
                # 'tracegroupgap': 1,
                # "bgcolor": "rgba(255,255,255,0)",
                "orientation": "h",
            },
            "xaxis": {
                "domain": [0, 0.5 - domain_sep / 2],
                "range": [0, 5.5],
                "constrain": "domain",
                **ax_com,
                "dtick": 1,
            },
            "xaxis2": {
                "domain": [0.5 + domain_sep / 2, 1],
                "scaleanchor": "x1",
                "constrain": "domain",
                "matches": "x1",
                **ax_com,
                "dtick": 1,
            },
            "yaxis": {
                "anchor": "x1",
                "range": [-3.6, 0.25],
                "title": {
                    "text": r"\eint{} /\jpermsq{}",
                    "font": {
                        "size": 8,
                    },
                },
                **ax_com,
                "dtick": 0.5,
                "tickmode": "array",
                "ticktext": [str(i) if (i % 1 == 0) else "" for i in yticks],
                "tickvals": yticks,
            },
            "yaxis2": {
                "anchor": "x2",
                "scaleanchor": "y1",
                "matches": "y1",
                **ax_com,
                "ticktext": [str(i) if (i % 1 == 0) else "" for i in yticks],
                "tickvals": yticks,
                "showticklabels": False,
            },
        }

        fig = graph_objects.Figure(
            data=data,
            layout=layout,
        )

        fig.write_image(f"AITT_{code}.svg")
        figs.append(fig)

    return figs


def show_min_vac_dist_change(code):

    def_code = code + "-a1"
    tt_prs = get_tensile_test(code)
    tt_def = get_tensile_test(def_code)

    srt_idx_prs = np.argsort(tt_prs.expansions)
    srt_idx_def = np.argsort(tt_def.expansions)

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
        subplot_titles=[
            "Pristine system",
            "Defective system",
        ],
        x_title="Distance / Ang.",
    )

    fig.add_trace(
        graph_objects.Scatter(
            **{
                "x": tt_prs.expansions[srt_idx_prs],
                "y": tt_prs.data["min_vac_dist_change_#1"][srt_idx_prs],
                "name": "Interface #1",
                "legendgroup": "Interface #1",
                "line": {
                    "color": DEFAULT_PLOTLY_COLORS[0],
                },
            },
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        graph_objects.Scatter(
            **{
                "x": tt_prs.expansions[srt_idx_prs],
                "y": tt_prs.data["min_vac_dist_change_#2"][srt_idx_prs],
                "name": "Interface #2",
                "legendgroup": "Interface #2",
                "line": {
                    "color": DEFAULT_PLOTLY_COLORS[1],
                },
            },
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        graph_objects.Scatter(
            **{
                "x": tt_def.expansions[srt_idx_def],
                "y": tt_def.data["min_vac_dist_change_#1"][srt_idx_def],
                "name": "Interface #1",
                "legendgroup": "Interface #1",
                "showlegend": False,
                "line": {
                    "color": DEFAULT_PLOTLY_COLORS[0],
                },
            },
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        graph_objects.Scatter(
            **{
                "x": tt_def.expansions[srt_idx_def],
                "y": tt_def.data["min_vac_dist_change_#2"][srt_idx_def],
                "name": "Interface #2",
                "legendgroup": "Interface #2",
                "line": {
                    "color": DEFAULT_PLOTLY_COLORS[1],
                },
                "showlegend": False,
            },
        ),
        row=1,
        col=2,
    )

    ax_com = {
        "linecolor": "black",
        "linewidth": 0.7,
        "ticks": "inside",
        "tickwidth": 1,
        "mirror": "ticks",
        "gridwidth": 1,
        "showgrid": False,
    }

    layout = {
        "width": 600,
        "height": 350,
        "template": "none",
        "title": code,
        "xaxis": {
            **ax_com,
            "range": [0, 6],
        },
        "xaxis2": {
            **ax_com,
        },
        "yaxis": {
            **ax_com,
            # "range": [-2.1, 0.3],
            # "tickformat": ".1f",
            "title": "Change in min. vac. thickness /Ang",
        },
        "yaxis2": {
            **ax_com,
            "matches": "y",
            "showticklabels": False,
        },
    }

    fig.layout.update(layout)

    return fig
