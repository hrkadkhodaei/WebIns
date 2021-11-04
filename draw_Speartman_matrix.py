import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2

dic1 = {'link_change_rate': 'LCR', 'link_change_rate_ET': 'LCR-ET', 'related_link_change_rate': 'LCR-R',
        'content_change_rate': 'CCR', 'num_new_outlinks_avg_history': 'NNL-AV',
        'num_new_outlinks_ET': 'NNL-ET', 'num_new_outlinks_week_10': 'NNL',
        'prob_of_new_link': 'PNL', 'NGB_preds': 'NNL-NGB', 'NGB_gener': 'ngb-gener',
        'new_link_using_ET': 'NL-ET', 'new_link_in_week_10': 'NL', 'related_num_new_outlinks': 'RNNL',
        'LCR': 'LCR', 'LCRET-SPSNDPRate': 'LCRET-SP_SN_RLCR',
        'LCRET-SPSNDP2DPRate': 'LCRET-SP_SN_DP2_RLCR', 'LCRET-SPSN': 'LCRET-SP_SN',
        'random': 'random'}

dic1 = {'link_change_rate': 'LCR', 'link_change_rate_ET': 'LCR-ET', 'related_link_change_rate': 'LCR-R',
        'content_change_rate': 'CCR', 'num_new_outlinks_avg_history': 'NNL-Av',
        'num_new_outlinks_ET': 'NNL-ET', 'num_new_outlinks_week_10': 'NNL',
        'prob_of_new_link': 'PNL', 'NGB_preds': 'NNL-NGB', 'NGB_gener': 'ngb-gener',
        'new_link_using_ET': 'NL-ET', 'new_link_in_week_10': 'NL', 'related_num_new_outlinks': 'NNL-R',
        'LCR': 'LCR', 'LCRET-SPSNDPRate': r'LCR-ET_{SP,SN,DP0}',
        'LCRET-SPSNDP2DPRate': 'LCR-ET_{SP,SN,DP2,RLCR}', 'LCRET-SPSN': 'LCR-ET', 'NNL-DB': 'NNL-Pr',
        'random': 'random'}

dic_attributes = {
    'link_change_rate':
        ['related_link_change_rate', 'link_change_rate_ET', 'LCRET-SPSNDP2DPRate', 'LCRET-SPSNDPRate',
         'prob_of_new_link', 'NGB_preds', 'content_change_rate',
         # 'LCRET-SPSNDPRate', 'LCRET-SPSNDP2DPRate', 'LCRET-SPSN'],
         ],
    'num_new_outlinks_week_10':
    # related_num_new_outlinks
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'NGB_preds'],
    'new_link_in_week_10':
        ['new_link_using_ET',  # 'num_new_outlinks_ET',
         'related_link_change_rate', 'link_change_rate_ET',
         'content_change_rate']
}

dic_attributes = {
    'link_change_rate':
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'prob_of_new_link',
         'NGB_preds', 'content_change_rate', 'new_link_using_ET', 'NNL-DB'],
    'new_link_in_week_10':
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'prob_of_new_link',
         'NGB_preds', 'content_change_rate', 'new_link_using_ET', 'NNL-DB'],
    'num_new_outlinks_week_10':
    # related_num_new_outlinks
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'prob_of_new_link',
         'NGB_preds', 'content_change_rate', 'new_link_using_ET', 'NNL-DB'],
}


def int_to_categorical(x):
    if round(x) > 0.5:
        #     if x > 0:
        return 1
    else:
        return 0


def bar_corr(data2, title='', ylabel='', filename=''):
    plt.rcParams.update({'font.size': 8})  # label size on axis ticks
    fig, ax = plt.subplots(figsize=(4, 2))
    idx = 0
    for data in data2:
        order_names = list(s for s in data.axes[0][:-1])
        test = [0.5 for i in range(len(order_names))]
        spman_corr = list(data[:-1])
        # norm = plt.Normalize(min(spman_corr), max(spman_corr))
        norm = plt.Normalize(0.0, 1.2)
        #     colors = plt.cm.RdYlGn(norm(spman_corr))
        # colors = plt.cm.copper_r(norm(spman_corr))
        # colors = plt.cm.RdYlGn(norm(spman_corr))
        colors = plt.cm.Greens(norm(spman_corr))
        # fig, ax = plt.subplots(figsize=(2.5, 0.58 + len(spman_corr) / 5))
        if idx == 1:
            colors = plt.cm.YlOrBr(norm(spman_corr))
        ax.barh(order_names, spman_corr, height=0.8, color=colors, zorder=3)
        idx += 1
        # ax.barh(order_names, test, height=0.8, color=colors2, zorder=3)
    #     plt.yticks(spman_corr, order_names)
    #     plt.tick_params(left=False)

    # for i in range(len(order_names)):
    #     plt.text(spman_corr[i]/2, i-0.1, round(spman_corr[i], 2), ha='center', fontsize=8,
    #              # )
    #              Bbox=dict(facecolor='white', alpha=.5, pad=1))
    plt.xlim((0, 1))
    plt.xlabel(ylabel, fontsize=8)
    # from matplotlib.cm import ScalarMappable
    # sm = ScalarMappable(cmap='copper_r', norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm)
    # cbar.set_label('Color', rotation=90, labelpad=10)
    #
    # # ticklabels = cbar.ax.get_ymajorticklabels()
    # # ticks = list(cbar.get_ticks())
    #
    # # Append the ticks (and their labels) for minimum and the maximum value
    # cbar.set_ticks([0, 1.3])
    # cbar.set_ticklabels(['aaaaa', 'bbbbb'])

    plt.ylabel("order approach")
    ax.grid(axis='x', color='grey', linestyle='--', linewidth=0.5, zorder=0)
    # plt.grid(axis='x', color='grey', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if len(filename) > 3:
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    # plt.show()
    plt.close()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=14)
    ax.set_yticklabels(row_labels, fontsize=16)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)

    # return im, cbar
    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


InExs = ['External', 'Internal']
fig_path = 'figures_spearman_correlation_orders-tmp/'
path = r'orders\\'
for InEx in InExs:
    fn = path + InEx + '_orders-NGB.csv'
    df_orig = pd.read_csv(fn)
    # df = df[df['num_new_outlinks_week_10'] > 0]
    df_orig = df_orig.apply(lambda x: pd.Series.round(x, 3))
    df_orig['prob_of_new_link2'] = df_orig['prob_of_new_link'].apply(int_to_categorical)
    filter_zeros = [False]
    # filter_zeros = [True]
    idx = 0
    fig, ax = plt.subplots(3, figsize=(10, 8))
    for target in dic_attributes.keys():
        data = []
        for filter_zero in filter_zeros:
            df = df_orig.copy()
            if filter_zero:
                df = df_orig[df[target] > 0]
            # f1 = df.sort_values(target)
            # f1[target + '_index1'] = [i for i in range(1, len(f1) + 1)]
            ranked_df = pd.DataFrame()
            ranked_df[dic1[target]] = df[target]
            for val in dic_attributes[target]:
                ranked_df[dic1[val]] = df[val]
            cr1 = ranked_df.corr(method='spearman')[dic1[target]].sort_values(ascending=True)
            print("*" * 8 + InEx + " " + target + "*" * 8 + "\n", cr1)
            data.append(cr1)
        # del cr1[dic1[target]]
        cols = ['NNL-ET', 'PNL', 'NL-ET', 'NNL-NGB', 'LCR-ET', 'NNL-Pr', 'NNL-Av', 'CCR']
        data2 = data[0][cols]
        a = np.array([data2])
        im = heatmap(a, [dic1[target]], cols if idx == 0 else [], ax=ax[idx], cmap="YlGn")
        texts = annotate_heatmap(im, valfmt="{x:.2f}", fontsize=14)

        idx += 1
    fig.tight_layout(w_pad=-5, h_pad=-28)
    # fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([1,2,3,4])
    # cbar_ax = fig.colorbar(1)
    # cbar_ax.ax.tick_params(size=0)
    # cbar_ax.set_ticks([])
    if InEx in ['External']:
        cbar_ax = fig.add_axes([0.82, 0.30, 0.05, 0.4])
        cb = fig.colorbar(im, cax=cbar_ax)
        cb.set_ticks([])
        cb.ax.set_title('Max', fontsize=14, pad=-6)
        cb.ax.set_xlabel('Min', fontsize=14)
    # fig.colorbar(im, ax=ax.ravel().tolist())
    plt.savefig(r'd:\a\SS\\' + 'Spearman' + '_' + InEx + '.png', dpi=500, bbox_inches='tight')
    # plt.show()
    print(InEx + ' finished')
    print('-' * 80)

# targets = ['LCR', 'NL', 'NNL']
# prediction_targets = ['w', 'x']
#
# data1 = np.array([[1, 2], [1, 2], [1, 6]])
# data1 = np.array([[1, 2]])
# fig, ax = plt.subplots(3)
#
# idx = 0
# for t in targets:
#     a = [t]
#
#     # im, cbar = heatmap(data1, a, prediction_targets, ax=ax[idx], cmap="YlGn", cbarlabel="harvest [t/year]")
#     # im, cbar = heatmap(data1, a, prediction_targets, ax=ax[idx], cmap="YlGn")
#     # im = heatmap(data1, a, prediction_targets, ax=ax[idx], cmap="YlGn")
#     im = heatmap(data1, a, prediction_targets if idx == 0 else [], ax=ax[idx], cmap="YlGn")
#     texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=20)
#
#     fig.tight_layout()
#     idx += 1
# plt.show()
"""

# fig, ax = plt.subplots()
# im = ax.imshow(data1)

# We want to show all ticks...
# ax.set_xticks(np.arange(len(prediction_targets)))
# ax.set_yticks(np.arange(len(targets)))
# # ... and label them with the respective list entries
# ax.set_xticklabels(prediction_targets)
# ax.set_yticklabels(targets)
# fig.tight_layout()
# plt.show()

plt.imshow(data1)
plt.colorbar()
plt.show()
"""
