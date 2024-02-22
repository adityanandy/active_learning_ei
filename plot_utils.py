import sys, os
import pandas as pd
import colorlover as cl
import numpy as np
from _plotly_future_ import v4_subplots
import plotly.graph_objs as go
import plotly.io as pio
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff

plotly.io.orca.config.executable = '/anaconda2/envs/pytorch/bin/orca'
init_notebook_mode(connected=True)

glob_layout = go.Layout(
    font=dict(family='Helvetica', size=24, color='black'),
    margin=dict(l=100, r=10, t=10, b=100),
    xaxis=dict(showgrid=False, zeroline=False, ticks="inside", showline=True,
               tickwidth=3, linewidth=3, ticklen=10,
               mirror="allticks", color="black"),
    yaxis=dict(showgrid=False, zeroline=False, ticks="inside", showline=True,
               tickwidth=3, linewidth=3, ticklen=10,
               mirror="allticks", color="black"),
    legend_orientation="v",
)


def get_front_line_points(pareto_points):
    pp = []
    for i, _p in enumerate(pareto_points[:-1]):
        _ps = pareto_points[i + 1]
        pp.append(_p)
        pp.append(np.array([_ps[0], _p[1]]))
    pp.append(pareto_points[-1])
    return np.array(pp)


def plot_ei_pi(df, next_points,
               pareto_points, val,
               y1l='y1', y2l='y2',
               figname=False, show=True,
               range1=False, range2=False):
    cs_dict = {"ei": "jet", "pi": "Picnic"}
    trace0 = go.Scatter(
        x=df[y1l].values,
        y=df[y2l].values,
        mode='markers',
        opacity=0.8,
        marker=dict(
            size=5,
            color=df[val].values / np.max(df[val].values),
            colorscale=cs_dict[val],
            cmin=0,
            cmax=1,
        ),
    )
    trace1 = go.Scatter(
        x=next_points[y1l].values,
        y=next_points[y2l].values,
        mode='markers',
        opacity=1,
        marker=dict(
            size=5,
            color='black',
            symbol='circle-open',
        ),
    )
    pp = get_front_line_points(pareto_points)
    trace2 = go.Scatter(
        x=pp[:, 0],
        y=pp[:, 1],
        mode='lines',
        opacity=1,
        line=dict(color='green', width=2, ),
    )
    data = [trace0, trace1, trace2]
    layout = go.Layout()
    layout.update(glob_layout)
    layout["xaxis"].update({'title': y1l})
    layout["yaxis"].update({'title': y2l})
    layout.legend.update(x=0, y=1.0, bgcolor='rgba(0,0,0,0)')
    layout.update(xaxis=dict(), yaxis=dict())
    layout.update(height=500, width=500, showlegend=False)
    if range1:
        layout["xaxis"].update({'range': range1})
    if range2:
        layout["yaxis"].update({'range': range2})
    fig = dict(data=data, layout=layout)

    if show:
        iplot(fig)
    if figname:
        path = '/'.join(figname.split('/')[:-1])
        if not os.path.isdir(path):
            os.makedirs(path)
        pio.write_image(fig, figname)


def plot_pareto_front(df, pareto_points,
                      y1l='y1', y2l='y2',
                      global_fronts=False,
                      next_fronts=False,
                      next_points=False,
                      figname=False, show=True,
                      range1=False, range2=False):
    """
    Generates a plot of the Pareto front.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame containing all of the data points.
    pareto_points : np.array
        The Pareto points out of all of the data in df.
    y1l : str
        The name of the first property of interest, as recorded in df as a column.
    y2l : str
        The name of the second property of interest, as recorded in df as a column.
    global_fronts : np.array
        The global Pareto points. IF False, not plotted.
    next_fronts : np.array
        The Pareto points that have so far been discovered. If False, not plotted.
    next_points : Pandas DataFrame
        Newly discovered points, to be plotted in blue. If False, not plotted.
    figname : str
        The name of the figure to be saved. If False, no figure is saved.
    show : bool
        Whether or not to display the Pareto plot.
    range1 : list
        A length 2 list indicating the lower and upper bound of the x axis. If False, default bounds are used.
    range2 : list
        A length 2 list indicating the lower and upper bound of the y axis. If False, default bounds are used.

    Returns
    -------
    This function does not return anything.

    """      

    trace0 = go.Scatter(
        x=df[y1l].values,
        y=df[y2l].values,
        mode='markers',
        opacity=0.5,
        marker=dict(
            size=5,
            color='black',
        ),
    )
    trace1 = go.Scatter(
        x=pareto_points[:, 0],
        y=pareto_points[:, 1],
        mode='markers',
        opacity=1,
        marker=dict(
            size=5,
            color='red',
        ),
    )
    pp = get_front_line_points(pareto_points)
    trace2 = go.Scatter(
        x=pp[:, 0],
        y=pp[:, 1],
        mode='lines',
        opacity=1,
        line=dict(color='red', width=2, ),
    )
    data = [trace0, trace1, trace2]
    if isinstance(global_fronts, np.ndarray):
        trace1 = go.Scatter(
            x=global_fronts[:, 0],
            y=global_fronts[:, 1],
            mode='markers',
            opacity=1,
            marker=dict(
                size=5,
                color='green',
            ),
        )
        pp = get_front_line_points(global_fronts)
        trace2 = go.Scatter(
            x=pp[:, 0],
            y=pp[:, 1],
            mode='lines',
            opacity=1,
            line=dict(color='green', width=2, ),
        )
        data += [trace1, trace2]
    if isinstance(next_fronts, np.ndarray):
        trace1 = go.Scatter(
            x=next_fronts[:, 0],
            y=next_fronts[:, 1],
            mode='markers',
            opacity=1,
            marker=dict(
                size=5,
                color='orange',
            ),
        )
        pp = get_front_line_points(next_fronts)
        trace2 = go.Scatter(
            x=pp[:, 0],
            y=pp[:, 1],
            mode='lines',
            opacity=1,
            line=dict(color='orange', width=2, dash='dash'),
        )
        data += [trace1, trace2]
    if isinstance(next_points, pd.DataFrame):
        trace1 = go.Scatter(
            x=next_points[y1l].values,
            y=next_points[y2l].values,
            mode='markers',
            opacity=1,
            marker=dict(
                size=5,
                color='blue',
            ),
        )
        data += [trace1]
    layout = go.Layout()
    layout.update(glob_layout)
    layout["xaxis"].update({'title': y1l})
    layout["yaxis"].update({'title': y2l})
    layout.legend.update(x=0, y=1.0, bgcolor='rgba(0,0,0,0)')
    layout.update(xaxis=dict(), yaxis=dict())
    layout.update(height=500, width=500, showlegend=False)
    if range1:
        layout["xaxis"].update({'range': range1})
    if range2:
        layout["yaxis"].update({'range': range2})
    fig = dict(data=data, layout=layout)

    if show:
        iplot(fig)
    if figname:
        path = '/'.join(figname.split('/')[:-1])
        if not os.path.isdir(path):
            os.makedirs(path)
        pio.write_image(fig, figname)


def plot_model(df, known_points,
               figname=False, show=True):
    trace0 = go.Scatter(
        x=df['x'].values,
        y=df['y1'].values,
        mode='lines',
        opacity=1,
        name='y1',
        line=dict(color='black', width=3, dash='solid'),
    )
    trace1 = go.Scatter(
        x=df['x'].values,
        y=df['hat_y1'] + df['sigma_y1'],
        mode='lines',
        opacity=0.5,
        name='y1_gp',
        marker=dict(color="#444"),
        line=dict(color='blue', width=0, dash='dash'),
        fillcolor='rgba(0, 0, 255, 0.3)',
        fill='tonexty'
    )
    trace2 = go.Scatter(
        x=df['x'].values,
        y=df['hat_y1'] - df['sigma_y1'],
        mode='lines',
        opacity=0.5,
        name='y1_gp',
        marker=dict(color="#444"),
        line=dict(color='blue', width=0, dash='dash'),
        fillcolor='rgba(0, 0, 255, 0.3)',
        fill='tonexty'
    )
    trace3 = go.Scatter(
        x=df['x'].values,
        y=df['hat_y1'].values,
        mode='lines',
        opacity=1,
        name='y1',
        line=dict(color='blue', width=3, dash='solid'),
    )
    trace4 = go.Scatter(
        x=known_points['x'].values,
        y=known_points['y1'].values,
        mode='markers',
        opacity=1,
        name='y1',
        marker=dict(size=8, color='blue', ),
    )
    trace0_1 = go.Scatter(
        x=df['x'].values,
        y=df['y2'].values,
        mode='lines',
        opacity=1,
        name='y2',
        line=dict(color='black', width=3, dash='solid'),
    )
    trace1_1 = go.Scatter(
        x=df['x'].values,
        y=df['hat_y2'] + df['sigma_y2'],
        mode='lines',
        opacity=0.5,
        name='y2_gp',
        marker=dict(color="#444"),
        line=dict(color='red', width=0, dash='dash'),
        fillcolor='rgba(255, 0, 0, 0.3)',
        fill='tonexty'
    )
    trace2_1 = go.Scatter(
        x=df['x'].values,
        y=df['hat_y2'] - df['sigma_y2'],
        mode='lines',
        opacity=0.5,
        name='y2_gp',
        marker=dict(color="#444"),
        line=dict(color='red', width=0, dash='dash'),
        fillcolor='rgba(255, 0, 0, 0.3)',
        fill='tonexty'
    )
    trace3_1 = go.Scatter(
        x=df['x'].values,
        y=df['hat_y2'].values,
        mode='lines',
        opacity=1,
        name='y2',
        line=dict(color='red', width=3, dash='solid'),
    )
    trace4_1 = go.Scatter(
        x=known_points['x'].values,
        y=known_points['y2'].values,
        mode='markers',
        opacity=1,
        name='y1',
        marker=dict(size=8, color='red', ),
    )
    data = [trace0, trace3, trace1, trace2, trace4,
            trace0_1, trace3_1, trace1_1, trace2_1, trace4_1]
    layout = go.Layout()
    layout.update(glob_layout)
    layout["xaxis"].update({'title': 'x'})
    layout["yaxis"].update({'title': 'y'})
    layout.legend.update(x=0, y=1.0, bgcolor='rgba(0,0,0,0)')
    layout.update(xaxis=dict(), yaxis=dict())
    layout.update(height=500, width=500, showlegend=False)

    fig = dict(data=data, layout=layout)
    if show:
        iplot(fig)
    if figname:
        path = '/'.join(figname.split('/')[:-1])
        if not os.path.isdir(path):
            os.makedirs(path)
        pio.write_image(fig, figname)


def plot_known_and_new(df, known_points, new_x,
                       figname=False, show=True):
    trace0 = go.Scatter(
        x=df[['y1', 'y2']].values[:, 0],
        y=df[['y1', 'y2']].values[:, 1],
        mode='markers',
        opacity=0.5,
        marker=dict(
            size=5,
            color='black',
        ),
        line=dict(color='black', width=2, dash='dash'),
    )
    trace1 = go.Scatter(
        x=known_points[['y1', 'y2']].values[:, 0],
        y=known_points[['y1', 'y2']].values[:, 1],
        mode='markers',
        opacity=1,
        marker=dict(
            size=8,
            color='blue',
        ),
        line=dict(color='black', width=2, dash='dash'),
    )
    trace2 = go.Scatter(
        x=[new_x['y1']],
        y=[new_x['y2']],
        mode='markers',
        opacity=1,
        marker=dict(
            size=8,
            color='green',
        ),
    )

    data = [trace0, trace1, trace2]
    layout = go.Layout()
    layout.update(glob_layout)
    layout["xaxis"].update({'title': 'c-1'})
    layout["yaxis"].update({'title': 'c-2'})
    layout.legend.update(x=0, y=1.0, bgcolor='rgba(0,0,0,0)')
    layout.update(xaxis=dict(), yaxis=dict())
    layout.update(height=500, width=500, showlegend=False)

    fig = dict(data=data, layout=layout)
    if show:
        iplot(fig)
    if figname:
        path = '/'.join(figname.split('/')[:-1])
    if not os.path.isdir(path):
        os.makedirs(path)
    pio.write_image(fig, figname)


def plot_sigma_dist(df, y1l='y1', y2l='y2'):
    hist_data = [df['sigma_y1'].values, df['sigma_y2'].values]
    group_labels = [y1l, y2l]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.01)
    fig.layout.update(glob_layout)
    fig.layout.xaxis.update({'title': 'sigma'})
    fig.layout.yaxis.update({'title': 'frequency'})
    fig.layout.legend.update(x=0.05, y=1.0, bgcolor='rgba(0,0,0,0)')
    fig.layout.update(width=500, height=500)
    iplot(fig)


def plot_model_violin(distribution_list, ylable,
                      fillcolor='blue', figname=False, show=True):
    tot_gen = len(distribution_list)
    mae_y1_calibrate = [np.mean(np.abs(distribution_list[gen])) for gen in range(tot_gen)]
    data = []
    for gen in range(tot_gen):
        trace = go.Violin(x=[gen for _ in distribution_list[gen]], y=np.abs(distribution_list[gen]),
                          box_visible=True, meanline_visible=True,
                          line_color='black', fillcolor=fillcolor, opacity=0.6, x0='%d' % gen)
        data += [trace]
    trace = go.Scatter(x=[gen for gen in range(tot_gen)], y=mae_y1_calibrate,
                       mode='markers+lines', opacity=1,
                       marker=dict(size=5, color='red', symbol='square'),
                       line=dict(color='red', width=2, dash='dash'), )
    data += [trace]
    layout = go.Layout()
    layout.update(glob_layout)
    layout.legend.update(x=0, y=1.0, bgcolor='rgba(0,0,0,0)')
    layout["xaxis"].update({'title': "gen", })
    layout["yaxis"].update({'title': ylable, })
    layout.update(height=500, width=1000, showlegend=False)

    fig = dict(data=data, layout=layout)
    if show:
        iplot(fig)
    if figname:
        path = '/'.join(figname.split('/')[:-1])
        if not os.path.isdir(path):
            os.makedirs(path)
        pio.write_image(fig, figname)


def plot_pareto_area(coverage_list, pareto_areas,
                     figname=False, show=True):
    trace = go.Scatter(x=coverage_list, y=pareto_areas,
                       mode='markers+lines', opacity=1,
                       marker=dict(size=8, color='black', symbol='square'),
                       line=dict(color='black', width=2, dash='dash'), )
    data = [trace]
    layout = go.Layout()
    layout.update(glob_layout)
    layout.legend.update(x=0, y=1.0, bgcolor='rgba(0,0,0,0)')
    layout.update(height=500, width=500, showlegend=False)
    layout["yaxis"].update({'title': 'pareto area', 'type': 'linear'})
    layout["xaxis"].update({'title': 'coverage', 'type': 'linear'})
    fig = dict(data=data, layout=layout)
    if show:
        iplot(fig)
    if figname:
        path = '/'.join(figname.split('/')[:-1])
        if not os.path.isdir(path):
            os.makedirs(path)
        pio.write_image(fig, figname)


def plot_pred_actual_front(pareto_points_pred, pareto_points,
                           pareto_pred_actual_space, pareto_pred_actual_space_2nd,
                           y1l='y1', y2l='y2', figname=False, show=True, ):
    trace1 = go.Scatter(
        x=pareto_points_pred[:, 0],
        y=pareto_points_pred[:, 1],
        mode='markers',
        opacity=1,
        marker=dict(
            size=5,
            color='red',
        ),
    )
    pp = get_front_line_points(pareto_points_pred)
    trace2 = go.Scatter(
        x=pp[:, 0],
        y=pp[:, 1],
        mode='lines',
        opacity=1,
        line=dict(color='red', width=2, dash='dash'),
    )
    trace3 = go.Scatter(
        x=pareto_points[:, 0],
        y=pareto_points[:, 1],
        mode='markers',
        opacity=1,
        marker=dict(
            size=5,
            color='green',
        ),
    )
    pp = get_front_line_points(pareto_points)
    trace4 = go.Scatter(
        x=pp[:, 0],
        y=pp[:, 1],
        mode='lines',
        opacity=1,
        line=dict(color='green', width=2),
    )
    trace5 = go.Scatter(
        x=pareto_pred_actual_space[:, 0],
        y=pareto_pred_actual_space[:, 1],
        mode='markers',
        opacity=1,
        marker=dict(
            size=7,
            color='orange',
            symbol='square-open',
            line=dict(width=2, color='orange')
        ),
    )
    trace6 = go.Scatter(
        x=pareto_pred_actual_space_2nd[:, 0],
        y=pareto_pred_actual_space_2nd[:, 1],
        mode='markers',
        opacity=1,
        marker=dict(
            size=7,
            color='blue',
            symbol='square-open',
            line=dict(width=2, color='blue')
        ),
    )
    data = [trace1, trace2, trace3, trace4, trace5, trace6]
    layout = go.Layout()
    layout.update(glob_layout)
    layout["xaxis"].update({'title': y1l})
    layout["yaxis"].update({'title': y2l})
    layout.legend.update(x=0, y=1.0, bgcolor='rgba(0,0,0,0)')
    layout.update(xaxis=dict(), yaxis=dict())
    layout.update(height=500, width=500, showlegend=False)
    fig = dict(data=data, layout=layout)
    if show:
        iplot(fig)
    if figname:
        path = '/'.join(figname.split('/')[:-1])
        if not os.path.isdir(path):
            os.makedirs(path)
        pio.write_image(fig, figname)
