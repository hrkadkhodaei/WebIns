import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2

dic1 = {'link_change_rate': 'LCR', 'link_change_rate_ET': 'LCRET', 'related_link_change_rate': 'RLCR',
        'content_change_rate': 'CCR', 'num_new_outlinks_avg_history': 'NNLAH',
        'num_new_outlinks_ET': 'NNLET', 'num_new_outlinks_week_10': 'NNLW10',
        'prob_of_new_link': 'PNL', 'prob_of_new_link2': 'PNL2', 'NGB_preds': 'NGB', 'NGB_gener': 'ngb-gener',
        'new_link_using_ET': 'NLET',
        'new_link_in_week_10': 'NLW10'}

dic_attributes = {
    'num_new_outlinks_week_10':
    # related_num_new_outlinks
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET', 'link_change_rate',
         'link_change_rate_ET', 'content_change_rate'],
    'link_change_rate':
        ['related_link_change_rate', 'link_change_rate_ET',
         'prob_of_new_link', 'NGB_preds', 'content_change_rate'],
    'new_link_in_week_10':
        ['link_change_rate', 'new_link_using_ET',
         'related_link_change_rate', 'link_change_rate_ET', 'content_change_rate'
         ]
}


def int_to_categorical(x):
    if round(x) > 0.5:
        #     if x > 0:
        return 1
    else:
        return 0


def bar_corr(data, title='', ylabel='', filename=''):
    order_names = list(s for s in data.axes[0][:-1])
    spman_corr = list(data[:-1])
    # norm = plt.Normalize(min(spman_corr), max(spman_corr))
    norm = plt.Normalize(0.0, 1.3)
    #     colors = plt.cm.RdYlGn(norm(spman_corr))
    colors = plt.cm.copper_r(norm(spman_corr))
    # colors = plt.cm.RdYlGn(norm(spman_corr))
    # colors = plt.cm.Greens(norm(spman_corr))
    # fig, ax = plt.subplots(figsize=(2.5, 0.58 + len(spman_corr) / 5))
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.barh(order_names, spman_corr, height=0.8, color=colors, zorder=3)
    #     plt.yticks(spman_corr, order_names)
    #     plt.tick_params(left=False)

    # for i in range(len(order_names)):
    #     plt.text(spman_corr[i]/2, i-0.1, round(spman_corr[i], 2), ha='center', fontsize=8,
    #              # )
    #              Bbox=dict(facecolor='white', alpha=.5, pad=1))

    plt.xlim((0, 1))
    plt.xlabel(ylabel)
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


InExs = ['External', 'Internal']
fig_path = 'figures_spearman_correlation_orders-tmp/'
path = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\orders\\'
for InEx in InExs:
    fn = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\\' + InEx + '_orders.csv'
    fn = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\orders\\' + InEx + '_orders_SP_SN_DN8_DP8_DPRate.csv'
    fn = r'd:/a/all_orders_tmp.csv'
    fn = path + InEx + '_orders-NGB.csv'
    df = pd.read_csv(fn)
    df = df.apply(lambda x: pd.Series.round(x, 2))
    df['prob_of_new_link2'] = df['prob_of_new_link'].apply(int_to_categorical)

    for target in dic_attributes.keys():
        f1 = df.sort_values(target)
        f1[target + '_index1'] = [i for i in range(1, len(f1) + 1)]
        ranked_df = pd.DataFrame()
        ranked_df[dic1[target]] = f1[target + '_index1']
        for val in dic_attributes[target]:
            f1 = f1.sort_values([val, target + '_index1'])
            f1[val + '-' + target + '_index2'] = [i for i in range(1, len(f1) + 1)]
            f1 = f1.sort_values(target + '_index1')
            ranked_df[dic1[val]] = f1[val + '-' + target + '_index2']
        cr1 = ranked_df.corr(method='spearman')[dic1[target]].sort_values(ascending=True)
        print(cr1)
        # del cr1[dic1[target]]
        bar_corr(data=cr1, ylabel='Spearman correlation with ' + dic1[target],
                 filename=path + fig_path + InEx + '_' + dic1[target] + '.png')
        ranked_df.to_csv(path + r'corr\\' + InEx + '_' + dic1[target] + '.csv', index=False, header=True)
        print('ending ' + target)
    print(InEx + ' finished')
    print('-' * 80)
