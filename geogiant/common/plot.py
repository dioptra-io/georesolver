
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll
import matplotlib.colors as mpl_colors
import scipy
# import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.markers as mmarkers
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


font = {#'family' : 'normal',
    'weight' : 'bold',
    'size'   : 16
}
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontsize_axis = 17
font_size_alone = 14
matplotlib.rc('font', **font)

markers = ["o", "s", "v", "^"]
linestyles = ["-", "--", "-.", ":"]

colors_blind = [
    ["blue", (0,114.0/255,178.0/255)],
    ["orange", (230.0/255, 159.0/255,0)],
    ["rebeccapurple", (204.0 / 255, 121.0 / 255, 167.0 / 255)],
    ["black", (0, 0, 0)],
    ["lightseagreen", (0, 158.0/255, 115.0/255)],
    ["skyblue", (86.0/255, 180.0/255,233.0/255)],
    # ["vermillon", (213.0/255, 94.0/255, 0) ],
    ["yellow", (240.0 / 255, 228.0 / 255, 66.0 / 255)],
]

colors = [
    'blue',
'green',
'red',
'cyan',
'magenta',
'yellow',
'black',
]


def plot_cdf_from_pdf(X, Y, inf_born, xtick_interval, sup_born, title, ofile):
    Y = np.array(Y)
    fig, ax = plt.subplots()


    ax.set_xlabel(title, fontsize=fontsize_axis)
    title = title + " CDF"
    #plt.title("CDF", fontsize=fontsize_axis)

    x_ticks = [inf_born]
    x_ticks.extend(np.arange(inf_born, sup_born + 1, xtick_interval))
    ax.set_xticks(x_ticks)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])
    # ax.set_xticklabels(xtickNames, rotation=45)
    # ax.set_xticklabels(xtickNames)

    ax.grid(linestyle = "dotted")
    # ax.set_ylabel("", fontsize=fontsize_axis)
    r = ax.plot(X[:, 0], Y, color = "black", linewidth = 1.35)
    # patches[0].set_xy(patches[0].get_xy()[:-1])
    # plt.xscale("log")
    # plt.yscale("log")
    ax.set_xlim(left=inf_born,right = sup_born)
    ax.set_ylim(bottom=0, top=1.05)
    # Normalize the data to a proper PDF
    plt.tight_layout()
    plt.savefig(r"resources/figures/" + ofile + ".pdf")

    plt.show()


def plot_cdf(Y, n_bins, inf_born, xtick_interval, sup_born, xlabel, ylabel, ofile, xlog=False, ylog=False, cumulative = True):
    Y = np.array(Y)
    fig, ax = plt.subplots()


    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    # title = title + " CDF"
    #plt.title("CDF", fontsize=fontsize_axis)

    # x_ticks = [inf_born]
    # x_ticks.extend(np.arange(inf_born, sup_born, xtick_interval))
    # ax.set_xticks(x_ticks)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])
    # ax.set_xticklabels(xtickNames, rotation=45)
    # ax.set_xticklabels(xtickNames)

    ax.grid(linestyle = "dotted")
    # ax.set_ylabel("", fontsize=fontsize_axis)
    n, bins, patches = ax.hist(Y, density=True, histtype='step', bins = n_bins,
                               cumulative=cumulative,color = "black", linewidth = 1.35)
    patches[0].set_xy(patches[0].get_xy()[:-1])
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlim(left=inf_born,right = sup_born)
    ax.set_ylim(bottom=0, top=1)
    # Normalize the data to a proper PDF
    return fig, ax

def plot_scatter(X, Y, xmin, xmax, ymin, ymax, xscale, yscale, xlabel, ylabel):
    '''
    Data is a map of occurences
    '''

    fig, ax = plt.subplots()


    # ax.set_xlabel(title, fontsize=fontsize_axis)
    #plt.title("CDF", fontsize=fontsize_axis)

    # x_ticks = [inf_born]
    # x_ticks.extend(np.arange(inf_born, sup_born, xtick_interval))
    # ax.set_xticks(x_ticks)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])
    # ax.set_xticklabels(xtickNames, rotation=45)
    # ax.set_xticklabels(xtickNames)

    ax.grid(linestyle = "dotted")
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    ax.scatter(X, Y, color = "black", marker="+", s=0.01)#, markersize=10, markeredgewidth=2)
    # patches[0].set_xy(patches[0].get_xy()[:-1])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_xlim(left=xmin,right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    return fig, ax

def plot_multiple_subplots(Xs, Ys, xmin, xmax, ymin, ymax, xscale, yscale, xlabel, ylabel,
                           markers, marker_colors, marker_size):

    if len(Ys) > 10:
        fig, axs = plt.subplots(len(Ys), figsize=(8 * len(Xs), 8 * len(Xs)), sharex=True)
    else:
        fig, axs = plt.subplots(len(Ys), sharex=True)


    for i  in  range(len(Ys)):
        if len(Ys) > 1:
            ax = axs[i]
        else:
            ax = axs
        ax.grid(linestyle = "dotted")
        ax.set_xlabel(xlabel, fontsize=fontsize_axis)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)
        X = Xs[i]
        Y = Ys[i]
        ax.plot(X, Y, color = marker_colors[i][0], marker=markers[i], markersize=marker_size[i], linewidth=0.1)#, markersize=10, markeredgewidth=2)
        # ax.plot(X, Y)
        # patches[0].set_xy(patches[0].get_xy()[:-1])
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        ax.set_xlim(left=xmin,right=xmax)
        ax.set_ylim(bottom=ymin[i], top=ymax[i])

    return fig, axs



def plot_scatter_multiple(Xs, Ys, xmin, xmax, ymin, ymax, xscale, yscale, xlabel, ylabel,
                          markers, marker_colors, marker_size):
    fig, ax = plt.subplots()


    # ax.set_xlabel(title, fontsize=fontsize_axis)
    #plt.title("CDF", fontsize=fontsize_axis)

    # x_ticks = [inf_born]
    # x_ticks.extend(np.arange(inf_born, sup_born, xtick_interval))
    # ax.set_xticks(x_ticks)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])
    # ax.set_xticklabels(xtickNames, rotation=45)
    # ax.set_xticklabels(xtickNames)
    from matplotlib.dates import DateFormatter, HourLocator

    ax.grid(linestyle = "dotted")
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)

    for i in range(0, len(Xs)):
        X = Xs[i]
        Y = Ys[i]

        ax.scatter(X, Y, c = marker_colors[i], marker=markers[i], s=marker_size[i])#, markersize=10, markeredgewidth=2)
        # ax.plot(X, Y)
        # patches[0].set_xy(patches[0].get_xy()[:-1])
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_xlim(left=xmin,right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)


    return fig, ax


def plot_quantile_per_ttl(Y, title, ofile):

    fig, axs = plt.subplots(2, 1, sharex=True, sharey="none")



    axs[0].grid(linestyle="dotted")

    # ax.set_ylabel("", fontsize=fontsize_axis)
    axs[0].boxplot(Y, whis="range")
    # patches[0].set_xy(patches[0].get_xy()[:-1])
    # plt.xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_yticks([10 ** i for i in range(0, 9)])


    # ax.set_xlim(left=inf_born, right=sup_born)

    x = [i for i in range(0, len(Y))]
    y = [len(Y[i]) for i in x]
    lines = []
    for i in range(0, len(x)):
        pair = [(x[i], 0), (x[i], y[i])]
        lines.append(pair)

    linecoll = matcoll.LineCollection(lines)
    axs[1].add_collection(linecoll)
    axs[1].set_yscale("log")
    axs[1].scatter(x, y)
    axs[1].set_xlabel(title, fontsize=fontsize_axis)
    axs[1].set_yticks([10 ** i for i in range(0, 6)])
    # axs[1].set_xticks(x)
    xticks = axs[1].xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i % 5 != 0:
            xticks[i].label1.set_visible(False)

    # Normalize the data to a proper PDF
    plt.tight_layout()
    plt.savefig(r"resources/figures/" + ofile + ".pdf")

    plt.show()


def plot_hexbins_joint_distribution(X, Y, xmin, xmax, ymin, ymax, xscale, yscale, xlabel, ylabel, norm, gridsize, title):
    x = np.asarray(X)
    y = np.asarray(Y)

    fig, ax = plt.subplots()
    # x_ticks = np.arange(xmin, xmax + 0.05, step=0.2)
    # ax.set_xticks(x_ticks)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])
    # ax.set_xticklabels(xtickNames)


    # y_ticks = np.arange(ymin, ymax + 0.05, step=0.2)
    # ax.set_yticks(y_ticks)
    # ytickNames = plt.setp(ax, yticklabels=["{0:.1f}".format(r) for r in y_ticks])
    # ax.set_yticklabels(ytickNames)
    if norm is not None:
        # norm = mpl_colors.LogNorm()
        hb = ax.hexbin(x, y, xscale=xscale, yscale=yscale,
                       cmap='Spectral_r', norm=norm,
                       gridsize=gridsize, mincnt=1,
                       )
    else:
        hb = ax.hexbin(x, y, xscale=xscale, yscale=yscale,
                       cmap='Spectral_r',
                       gridsize=gridsize, mincnt=1,
                       )
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    ax.set_title(title)
    # ax.set_yscale("log")
    # plt.title(title)
    cb = plt.colorbar(hb)
    cb.set_label("Count")

    return fig, ax


def plot_multiple_hexbins_joint_distribution(Xs, Ys, xmin, xmax, ymin, ymax, xscale, yscale, xlabel, ylabel, legend):

    assert (len(Xs) == len(Ys))
    fig, axs = plt.subplots(sharex='all', sharey='all', ncols=len(Xs))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

    for i in range(len(Xs)):
        ax = axs[i]
        X = Xs[i]
        Y = Ys[i]


        x = np.asarray(X)
        y = np.asarray(Y)


        # x_ticks = np.arange(xmin, xmax + 0.05, step=0.2)
        # ax.set_xticks(x_ticks)
        # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])
        # ax.set_xticklabels(xtickNames)


        # y_ticks = np.arange(ymin, ymax + 0.05, step=0.2)
        # ax.set_yticks(y_ticks)
        # ytickNames = plt.setp(ax, yticklabels=["{0:.1f}".format(r) for r in y_ticks])
        # ax.set_yticklabels(ytickNames)



        # fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        hb = ax.hexbin(x, y, xscale=xscale, yscale=yscale, cmap='Spectral_r', norm=mpl_colors.LogNorm(), gridsize=25)
        ax.set_xlim(left=xmin, right=xmax)
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.set_xlabel(xlabel, fontsize=fontsize_axis)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)
        ax.set_title(legend[i])
        # ax.set_yscale("log")
        # plt.title(title)
        cb = plt.colorbar(hb)
        cb.set_label("Count")

    return fig, axs


def hexbins_ratio(X_pos, Y_pos, X_neg, Y_neg, xmin, xmax, ymin, ymax, xlabel, ylabel, xscale, yscale, title):



    x_neg = np.asarray(X_neg)
    y_neg = np.asarray(Y_neg)

    n = 2000
    if xscale == "log":
        x_bins = np.logspace(np.log10(0.1), np.log10(xmax), int(np.sqrt(n)))
    else:
        x_bins = np.linspace(xmin, xmax, int(np.sqrt(n)))
    if yscale == "log":
        y_bins = np.logspace(np.log10(0.1), np.log10(ymax), int(np.sqrt(n)))
    else:
        y_bins = np.linspace(ymin, ymax, int(np.sqrt(n)))


    H, xedges, yedges = np.histogram2d(x_neg, y_neg, range=[[xmin, xmax], [ymin, ymax]], bins=[x_bins, y_bins])

    x_pos = np.asarray(X_pos)
    y_pos = np.asarray(Y_pos)
    H_num, xedges, yedges = np.histogram2d(x_pos, y_pos, range=[[xmin, xmax], [ymin, ymax]], bins=(xedges, yedges))

    H_denom = H_num + H
    H_ratio = H_num / H_denom

    fig, ax = plt.subplots()
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    X, Y = np.meshgrid(xedges, yedges)
    hb = ax.pcolormesh(X, Y, H_ratio, cmap="Spectral_r")
    cb = plt.colorbar(hb)
    cb.set_label("Hit rate")
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    ax.set_title(title)

    return fig, ax



def plot_multiple_cdf(Ys, n_bins,
                      xmin, xmax,
                      xlabel, ylabel,
                      legend,
                      ymin=0,
                      ymax=1.05,
                      xticks = None,
                      xticks_labels = None,
                      xscale="linear", yscale="linear",
                      cumulative=True,
                      figure=None, axes=None,
                      offset=0,
                      colors_arg = None,
                      linestyles_arg = None):

    if figure is not None and axes is not None:
        fig = figure
        ax = axes
    else:
        subplots = plt.subplots()
        fig, ax = subplots
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    # title = title + " CDF"
    # plt.title("CDF", fontsize=fontsize_axis)


    ax.grid(linestyle="dotted")
    if len(Ys) == 1:
        i = 0
        Y = Ys[i]
        if colors_arg is not None:
            color = colors_arg[i][1]
        else:
            color = colors_blind[(i + offset) % len(colors_blind)][1]

        if linestyles_arg is not None:
            linestyle = linestyles[i]
        else:
            linestyle = linestyles[(i + offset) % len(linestyles)]

        n, bins, patches = ax.hist(Y, density=True, histtype='step', bins=n_bins,
                                   cumulative=cumulative, linewidth=1.35,
                                   color=color,
                                   linestyle=linestyle)
        patches[0].set_xy(patches[0].get_xy()[1:-1])
    else:
        for i in range(0, len(Ys)):
            Y = Ys[i]
            if colors_arg is not None:
                color = colors_arg[i][1]
            else:
                color = colors_blind[(i + offset) % len(colors_blind)][1]

            if linestyles_arg is not None:
                linestyle = linestyles_arg[i]
            else:
                linestyle = linestyles[(i + offset) % len(linestyles)]

            n, bins, patches = ax.hist(Y, density=True, histtype='step', bins = n_bins,
                                       cumulative=cumulative, linewidth = 1.35, label = legend[i],
                                       color=color,
                                       linestyle=linestyle)
            patches[0].set_xy(patches[0].get_xy()[1:-1])

    # plt.xscale("symlog")
    # xticks = ax.xaxis.get_major_ticks()
    # xticks[1].label1.set_visible(False)
    # # xticks[2].label1.set_visible(False)
    # xticks[-2].label1.set_visible(False)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xmin, right = xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    if xticks is not None:
        ax.set_xticks(xticks)
    # xtickNames = plt.setp(ax, xticklabels=[f"{r}" for r in x_ticks])
    if xticks_labels is not None:
        ax.set_xticklabels(xticks_labels)

    # Normalize the data to a proper PDF
    # plt.tight_layout()
    # plt.savefig(r"resources/figures/" + ofile + ".pdf")
    return fig, ax

def plot_multiple_cdf_from_pickle(fig, ax,
                                  xmin=None, xmax=None,
                                  xlabel=None, ylabel=None,
                                  xlog=False, ylog=False,
                                  legend_loc = None):
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    # title = title + " CDF"
    # plt.title("CDF", fontsize=fontsize_axis)

    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])
    # ax.set_xticklabels(xtickNames, rotation=45)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if xmin is not None:
        ax.set_xlim(left=xmin)
    if xmax is not None:
        ax.set_xlim(right = xmax)
    ax.set_ylim(bottom=0, top=1.05)


    assert(legend_loc is not None)
    homogenize_legend(ax, legend_loc)
    # if legend_loc is not None:
    #     ax.legend(loc=legend_loc)

    return fig, ax
    # Normalize the data to a proper PDF
    # plt.tight_layout()



def plot_stacked_bars_and_cumulative_total(Xs, Ys, Yerrs, xlabel, ylabel, title, legend, ofile):
    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel, fontsize=fontsize_axis)

    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    ax.set_title(title, fontsize=fontsize_axis)
    plt.grid(linestyle="dotted")

    # Last one is total cumulative
    for i in range(len(Ys)):
        X = Xs[i]
        Y = Ys[i]


        if i == len(Ys) - 1:
            Yerr = Yerrs[0]
            p = ax.bar(X, Y, 0.35, yerr=Yerr, bottom=Ys[i-1], label=legend[i], color=colors_blind[i][1], align="center")
        elif i > 0:
            p = ax.bar(X, Y, 0.35, bottom=Ys[i-1], label=legend[i], color=colors_blind[i][1], align="center")
        else:
            p = ax.bar(X, Y, 0.35, label=legend[i], color=colors_blind[i][1], align="center")

    # Y_total     = Ys[len(Ys) - 1]
    #
    # Y_err_total = Yerrs[len(Yerrs) - 1]
    # X_total     = Xs[len(Xs) - 1]
    #
    # ax.errorbar(X_total, Y_total, Y_err_total, label=legend[len(Ys) - 1], linewidth=0.5,
    #             marker=markers[(len(Ys) - 1) % len(markers)], markersize=10, markeredgewidth=2,
    #             capsize=2)

    plt.legend(loc="upper right", prop={'size': 16}) #, bbox_to_anchor=(1, 0.65),
    # ncol=1, fancybox=True, shadow=True,  prop={'size': 18})
    # ax.set_yscale("log")
    ax.set_xlim(left=0.8, right=10.8)
    ax.set_ylim(bottom=0)#, top=3500000)
    plt.tight_layout()
    plt.savefig(r"resources/figures/" + ofile + ".pdf")
    plt.show()

def plot_multiple_error_bars(X, Ys, Yerrs,
                             xmin, xmax, ymin, ymax,
                             xlabel, ylabel,
                             xscale, yscale,
                             labels):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)

    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    ax.grid(linestyle="dotted")

    # x_ticks = [inf_born+1]
    for i in range(len(Ys)):

        Y = Ys[i]
        Yerr = Yerrs[i]
        lns1 = ax.errorbar(X, Y, Yerr,  label=labels[i], linewidth=0.5,
                           marker=markers[i % len(markers)], markersize=1, markeredgewidth=1,
                           capsize=2)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    return fig, ax



def plot_aggregate_data_in_beanplot(data_to_boxplot,
                                    xticks,
                                    ofile,
                                    ymin,
                                    ymax,
                                    title='',
                                    xlabel='', ylabel='', pickling=False, type_of_boxplot=0,
                                    jitter=True, fontsize=25, ticksize_x=None, ticksize_y=None,
                                    ax=None, xscale='linear', yscale='linear'):
    # rcParams.update({'font.size': fontsize})
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot!
    data_to_boxplot = pd.Series([pd.Series(g) for g in data_to_boxplot])
    if not jitter:
        plot_opts = {'violin_fc': (0.8, 0.8, 0.8), 'cutoff': True,
                     'bean_color': '#FF6F00', 'bean_mean_color': '#009D91',
                     'bean_median_color': 'b', 'bean_median_marker': '+'}
    else:
        plot_opts = {'violin_fc': (0.8, 0.8, 0.8), 'cutoff': True,
                     'jitter_marker': '.', 'jitter_marker_size': 1,
                     'bean_color': '#FF6F00',
                     'bean_mean_size' : 1.2, 'bean_mean_color': '#009D91',
                     'bean_median_color': 'b', 'bean_median_marker': '+'}
    to_remove = []
    for i, g in enumerate(data_to_boxplot):
        if len(g) <= 1:
            to_remove.append(i)
    for i in sorted(to_remove, reverse=True):
        data_to_boxplot.pop(i)
        positions.pop(i)
    if len(data_to_boxplot) == 0:
        return
    sm.graphics.beanplot(data_to_boxplot, ax=ax, jitter=jitter, plot_opts=plot_opts)

    total_experiments = sum(len(a_data_set) for a_data_set in data_to_boxplot)
    # draw annotations
    # annotate = False
    # if annotate:
    #     counter = 1
    #     for a_data_set in data_to_boxplot:
    #         datalen = len(a_data_set)
    #         fraction = float(datalen) * 100 / total_experiments
    #         pos_x = counter + 0.1
    #         pos_y = scipy.stats.scoreatpercentile(a_data_set, 75)
    #
    #         ax.annotate('%s (%.1f%%)' % (datalen, fraction), xy=(pos_x, pos_y),
    #                     xytext=(pos_x + 0.2, pos_y + 1),
    #                     arrowprops=dict(facecolor='black', shrink=0.05),
    #                     )
    #         counter += 1
    xtickNames = plt.setp(ax, xticklabels=[str(r) for r in xticks])

    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xtickNames, rotation=45)
    # ax.tick_params(axis='x', which='major', labelsize=10)
    if ticksize_x:
        ax.tick_params(axis='x', which='major', labelsize=ticksize_x)
    if ticksize_y:
        ax.tick_params(axis='y', which='major', labelsize=ticksize_y)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_title(title)
    fig.tight_layout()
    fig.show()
    plt.savefig(r"resources/figures/" + ofile + ".pdf")


def plot_dist_from_dict(distributions, cumulative, xlabel, ylabel, labels, ofile):

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    # ax.set_xlim(left=0, 32)
    plt.grid(linestyle="dotted")

    for i in range(len(distributions)):
        d = distributions[i][1]
        d = {int(k): v for k, v in d.items()} # dict keys to int
        d = dict(sorted(d.items())) # order dict
        values = list(d.values())

        # ax.set_title(title, fontsize=fontsize_axis)
        if cumulative:
            total = sum(values)
            d = dict(zip(d.keys(), (100*subtotal/total for subtotal in np.cumsum(values))))
            # ax.plot(list(d_cumsum.keys()), list(d_cumsum.values()))
        ax.plot(list(d.keys()), list(d.values()),
                linewidth=0.5,
                marker="s", markersize=1, markeredgewidth=2,
                label=labels[i])

    plt.tight_layout()
    plt.legend(loc="upper left", prop={'size': 8})
    plt.savefig(r"resources/figures/" + ofile + ".pdf")


def plot_dist_from_dict_v2(fig, ax, distributions, n_bins, xlabel, ylabel, xscale, yscale,
                           xtick_interval,
                           xmin, xmax, ymin, ymax,
                           labels, cumulative):

    # Make bins from xmin and xmax

    # Divide the xmax and xmin in n_bins
    if xmax == 1.05:
        xmax_value = 1
        interval_bin_size = (xmax_value - xmin) / n_bins
    else:
        xmax_value = xmax
        interval_bin_size = (xmax_value - xmin) / n_bins
    intervals = [((i * interval_bin_size) + xmin, (i + 1) * interval_bin_size + xmin) for i in range(0, n_bins)]

    assert(intervals[-1][1] == xmax_value)

    # fig, ax = plt.subplots()

    # ax.set_xlim(left=0, 32)


    for i in range(len(distributions)):
        d = distributions[i]
        # d = {int(k): v for k, v in d.items()} # dict keys to int
        d = dict(sorted(d.items())) # order dict
        values = list(d.values())
        # ax.set_title(title, fontsize=fontsize_axis)
        X = []
        Y = []
        if cumulative:
            total = sum(values)

            y = 0

            values_per_interval = {interval : 0 for interval in intervals}
            current_index_interval = 0
            for v, occ in d.items():
                # Assume d keys are sorted
                if v < intervals[0][0]:
                    values_per_interval[intervals[0]] += occ
                    continue
                for k in range(len(intervals[current_index_interval:])):
                    start_interval, end_interval = intervals[current_index_interval + k]
                    # Assume intervals are sorted
                    if end_interval != xmax_value:
                        if start_interval <= v < end_interval:
                            values_per_interval[intervals[current_index_interval + k]] += occ
                            current_index_interval = current_index_interval + k
                            break
                    else:
                        if start_interval <= v <= end_interval:
                            values_per_interval[intervals[current_index_interval + k]] += occ
                            current_index_interval = current_index_interval + k
                            break

            # Sort values_per_intervals
            values_per_interval = sorted(list(values_per_interval.items()), key=lambda x: x[0][0])



            # for start_interval, end_interval in intervals:
            #     print(start_interval, end_interval)
            #     # Each interval is two points with the same y
            #     for v, occ in list(d.items()):
            #         if v < intervals[0][0]:
            #             y += occ
            #             del d[v]
            #             continue

            y = 0
            for k in range(len(values_per_interval)):
                start_interval, end_interval = values_per_interval[k][0]
                value = values_per_interval[k][1]
                X.append(start_interval)
                X.append(end_interval)
                y += value
                Y.append(y)
                Y.append(y)
            assert(y == total)
            Y = [y/total for y in Y]

            # d = dict(zip(d.keys(), (subtotal/total for subtotal in np.cumsum(values))))
            # ax.plot(list(d_cumsum.keys()), list(d_cumsum.values()))
        ax.plot(X, Y,
                linewidth=1.35,
                # marker="s", markersize=1, markeredgewidth=2,
                label=labels[i],
                color=colors_blind[i % len(colors_blind)][1],
                linestyle=linestyles[i % len(linestyles)])

    # xmax = 1.05
    x_ticks = [xmin]
    x_ticks.extend(np.arange(xmin, xmax, xtick_interval))
    ax.set_xticks(x_ticks)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    plt.grid(linestyle="dotted")
    # ax.legend(loc="upper left", prop={'size': 14})
    return fig, ax

    # plt.tight_layout()
    #
    # plt.savefig(r"resources/figures/" + ofile + ".pdf")
    # plt.show()


def plot_dist_from_dict_subplots(distributions_list, labels_list, scale_list,
                                 xtick_interval_list, ytick_interval_list, x_bounds_list, y_bounds_list,
                                 legend_list, cumulative, ofile):

    fig, axs = plt.subplots(2, 2)

    # ax.set_xlim(left=0, 32)

    for i in range(2):
        for j in range(2):
            ax = axs[i, j]

            distributions = distributions_list[i]
            xlabel, ylabel = labels_list[i]
            xscale, yscale = scale_list[i]
            xtick_interval = xtick_interval_list[i]
            ytick_interval = ytick_interval_list[i]
            xmin, xmax = x_bounds_list[i]
            ymin, ymax = y_bounds_list[i]
            labels = legend_list[i]

            for i in range(len(distributions)):
                d = distributions[i]
                # d = {int(k): v for k, v in d.items()} # dict keys to int
                d = dict(sorted(d.items())) # order dict
                values = list(d.values())

                if cumulative:
                    total = sum(values)
                    d = dict(zip(d.keys(), (subtotal/total for subtotal in np.cumsum(values))))
                ax.plot(list(d.keys()), list(d.values()),
                        linewidth=1.35,
                        # marker="s", markersize=1, markeredgewidth=2,
                        label=labels[i],
                        color=colors_blind[i % len(colors_blind)][1],
                        linestyle=linestyles[i % len(linestyles)])

        x_ticks = [xmin]
        x_ticks.extend(np.arange(xmin, xmax, xtick_interval))
        ax.set_xticks(x_ticks)

        y_ticks = [ymin]
        y_ticks.extend(np.arange(ymin, ymax, ytick_interval))
        ax.set_yticks(y_ticks)

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(xlabel, fontsize=fontsize_axis)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)

        ax.grid(linestyle="dotted")
        ax.legend(loc="best")

    # plt.tight_layout()
    plt.savefig(r"resources/figures/" + ofile + ".pdf")
    plt.show()


def plot(X, Y, xmin, xmax, ymin, ymax, xticks_labels, xlabel, ylabel):
    fig, ax = plt.subplots()


    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)

    if xticks_labels is not None:
        x_ticks = [x for x in range(len(xticks_labels))]
        ax.set_xticks(x_ticks)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])

    # ax.set_xticklabels(xticks_labels, rotation=90)

    # xticks = ax.xaxis.get_major_ticks()
    # for i in range(len(xticks)):
    #     if i % 12 != 0:
    #         xticks[i].label1.set_visible(False)
    # ax.set_xticklabels(xtickNames)

    ax.grid(linestyle = "dotted")
    ax.plot(X, Y)
    # ax.set_yscale("log")
    ax.set_xlim(left=xmin,right = xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    # Normalize the data to a proper PDF
    return fig, ax

def plot_multiple(Xs, Ys, xmin, xmax, ymin, ymax, xscale, yscale, xticks_labels, xlabel, ylabel, labels, colors=None):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    # plt.title("CDF", fontsize=fontsize_axis)

    if xticks_labels is not None:
        x_ticks = [x for x in range(len(xticks_labels))]
        ax.set_xticks(x_ticks)

    ax.grid(linestyle="dotted")
    for i in range(0, len(Ys)):
        Y = Ys[i]
        X = Xs[i]
        if colors is None:
            lines = ax.plot(X, Y, linewidth = 1.35, label = labels[i], linestyle=linestyles[i % len(linestyles)])
        else:
            lines = ax.plot(X, Y, linewidth=1.35, label=labels[i], linestyle=linestyles[i % len(linestyles)], color=colors[i])
        # patches[0].set_xy(patches[0].get_xy()[:-1])

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xmin,right = xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    return fig, ax

def plot_add_twin(fig, ax, X, Y,
                  xmin, xmax, ymin, ymax,
                  xscale, yscale,
                  xlabel, ylabel,
                  label,
                  xticks_labels,
                  ofile):

    ax_twin = ax.twiny()

    ax_twin.set_xlabel(xlabel, fontsize=fontsize_axis)

    ax_twin.plot(X, Y, label=label, color="black")
    ax_twin.set_xlim(left=xmin, right=xmax)
    ax_twin.set_xscale(xscale)
    xticks = ax_twin.get_xticks()
    if xscale == "linear":
        # Only select the xtick_labels corresponding to the number of xticks
        ax_twin.set_xticklabels([xticks_labels[int(i)] for i in xticks[:-1]], rotation=0)
    # ax_twin.set_xticks([i for i in range(len(xticks_labels))])


    ax_twin.legend(loc="center right", prop={"size": 12})
    # ax.set_ylim(bottom=ymin, top=ymax)
    ax_twin_xlabels = ax_twin.get_xticklabels()
    # plt.setp(ax_twin_xlabels, visible=True)

    return fig, ax, ax_twin


def plot_add_twin_from_pickle(ax_twin,
                              # xmin, xmax, ymin, ymax,
                              xscale, yscale,
                              xlabel, ylabel, legend_loc=None):

    ax_twin.set_xlabel(xlabel, fontsize=fontsize_axis)
    # ax_twin.set_xlim(left=xmin, right=xmax)
    # ax_twin.set_xscale(xscale)
    # ax_twin.set_xticks([i for i in range(len(xticks_labels))])


    if legend_loc is not None:
        ax_twin.legend(loc=legend_loc, prop={"size": 12})
    # ax.set_ylim(bottom=ymin, top=ymax)
    # plt.setp(ax_twin_xlabels, visible=True)

    return ax_twin


def plot_multiple_bar(X, Ys,
                      xmin, xmax, ymin, ymax,
                      xlabel, ylabel,
                      labels,
                      xticks_names = None,
                      xscale="linear", yscale="linear",
                      style="stacked"):
    fig, ax = plt.subplots()


    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    # title = title + " CDF"
    #plt.title("CDF", fontsize=fontsize_axis)

    # x_ticks = [x for x in range(len(xticks_labels))]
    # ax.set_xticks(x_ticks)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])

    # ax.set_xticklabels(X, rotation=45)

    # xticks = ax.xaxis.get_major_ticks()
    # for i in range(1, len(xticks)+1):
    #     if i % 3 != 0:
    #         xticks[i].label1.set_visible(False)
    if xticks_names is not None:
        ax.set_xticklabels(xticks_names)

    # ax.set_xticklabels([xticks_labels[int(i)] for i in x_ticks[:-1]], rotation=0)

    ax.grid(linestyle="dotted")
    width = 0.20
    X = np.array(X)
    for i in range(0, len(Ys)):
        Y = Ys[i]

        if style == "stacked":
            lines = ax.bar(X , Y, label = labels[i])
        elif style == "near":
            lines = ax.bar(X + i * width, Y, width=width, label=labels[i])

        # patches[0].set_xy(patches[0].get_xy()[:-1])

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xmin,right = xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    return fig, ax

def plot_bar(Y,
             xmin, xmax, ymin, ymax,
             xlabel, ylabel,
             xticks_names = None,
             xscale="linear", yscale="linear", label_size=8,
             ):
    fig, ax = plt.subplots()


    ax.set_xlabel(xlabel, fontsize=fontsize_axis)
    ax.set_ylabel(ylabel, fontsize=fontsize_axis)
    # title = title + " CDF"
    #plt.title("CDF", fontsize=fontsize_axis)
    # xtickNames = plt.setp(ax, xticklabels=["{0:.1f}".format(r) for r in x_ticks])

    # ax.set_xticklabels(X, rotation=45)

    # xticks = ax.xaxis.get_major_ticks()
    # for i in range(1, len(xticks)+1):
    #     if i % 3 != 0:
    #         xticks[i].label1.set_visible(False)

    # ax.set_xticklabels([xticks_labels[int(i)] for i in x_ticks[:-1]], rotation=0)

    ax.grid(linestyle="dotted")
    width = 0.20
    X = np.arange(len(Y))
    lines = ax.bar(X , Y)
    # patches[0].set_xy(patches[0].get_xy()[:-1])

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(left=xmin,right = xmax)
    ax.set_ylim(bottom=ymin, top=ymax)
    if xticks_names is not None:
        x_ticks = [x for x in range(len(Y))]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(xticks_names, rotation=90)
        ax.tick_params(axis='x', which='major', labelsize=label_size)

    return fig, ax

def plot_save(ofile, is_tight_layout):
    if is_tight_layout:
        plt.tight_layout()
    # plt.show()
    plt.savefig(ofile)

    # plt.clf()

def homogenize_legend(ax, legend_location, legend_size=14):
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if isinstance(h, Line2D):
            new_handles.append(h)
        elif isinstance(h, Polygon):
            new_handles.append(Line2D([], [], linestyle=h.get_linestyle(), color=h.get_edgecolor()))
    ax.legend(loc=legend_location, prop={"size": legend_size}, handles=new_handles, labels=labels)


if __name__ == "__main__":
    import json
    ixp_name = "AMS-IX"
    with open(f"resources/evaluation/distance_{ixp_name}_validated_peering_link.cdf") as f:
        data = json.load(f)
        Xs, Ys = data

    hexbins_ratio(Xs[0], Ys[0], Xs[1], Ys[1],
                  xmin=0, xmax=max(max(x) for x in Xs),
                  ymin=0, ymax=max(max(x) for x in Ys),
                  xscale="log", yscale="log",
                  xlabel=f"S -- {ixp_name}",
                  ylabel=f"T -- {ixp_name}",
                  title="Ratio of validated links"
                  )

    plot_save("resources/evaluation/distance_AMS-IX_validated_peering_link.pdf", is_tight_layout=True)