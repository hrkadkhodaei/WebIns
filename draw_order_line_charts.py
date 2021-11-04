import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import random

a = ['num_new_outlinks_avg_history', 'num_new_outlinks_ET', 'link_change_rate',
     'related_link_change_rate', 'link_change_rate_ET', 'content_change_rate',
     'prob_of_new_link', 'new_link_using_ET', 'new_link_in_week_10',
     'num_new_outlinks_week_10']

dic_attributes = {
    'num_new_outlinks_week_10':
    # related_num_new_outlinks
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'NGB_preds'],
    'link_change_rate':
        ['related_link_change_rate', 'link_change_rate_ET',  # 'LCRET-SPSNDPRate',
         'prob_of_new_link', 'NGB_preds', 'content_change_rate'
         # 'LCRET-SPSN', 'LCRET-SPSNDP2DPRate',
         # 'LCRET-SPSNDPRate', 'LCRET-SPSNDP2DPRate', 'LCRET-SPSN'],
         ],
    'new_link_in_week_10':
        ['new_link_using_ET', 'prob_of_new_link',  # 'num_new_outlinks_ET',
         'related_link_change_rate', 'link_change_rate_ET',
         'content_change_rate']
}

dic_attributes = {
    'num_new_outlinks_week_10':
    # related_num_new_outlinks
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'prob_of_new_link',
         'NGB_preds', 'content_change_rate', 'new_link_using_ET', 'NNL-DB'],
    'link_change_rate':
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'prob_of_new_link',
         'NGB_preds', 'content_change_rate', 'new_link_using_ET', 'NNL-DB'],
    'new_link_in_week_10':
        ['num_new_outlinks_avg_history', 'num_new_outlinks_ET',
         'link_change_rate_ET', 'prob_of_new_link',
         'NGB_preds', 'content_change_rate', 'new_link_using_ET', 'NNL-DB'],
}

remove_zero_values = False
# remove_zero_values = True
InExs = ['External', 'Internal']
# InExs = ['Internal']
targets = ['num_new_outlinks_week_10', 'link_change_rate', 'new_link_in_week_10']
# targets = ['link_change_rate']
round_decimal = 8
num_iter_avg = 5
# gap_between_records = [500, 1000, 1500, 2000, 2500, 3000]
separator_in_filename = '-'

markers = ['o', '*', 'v', '.', 'p', '+', 'd', 'x', 'X', 'P']
dic_labels = {'link_change_rate': 'LCR', 'link_change_rate_ET': 'LCR-ET_{ALL}', 'related_link_change_rate': 'LCR-R',
              'content_change_rate': 'CCR', 'num_new_outlinks_avg_history': 'NNL-AV',
              'num_new_outlinks_ET': 'NNL-ET_{ALL}', 'num_new_outlinks_week_10': 'NNL',
              'prob_of_new_link': 'PNL', 'NGB_preds': 'NNL-NGB_{ALL}', 'NGB_gener': 'ngb-gener',
              'new_link_using_ET': 'NL-ET_{ALL}', 'new_link_in_week_10': 'NL', 'related_num_new_outlinks': 'RNNL',
              'LCR': 'LCR', 'LCRET-SPSNDPRate': r'LCR-ET_{SP,SN,DN0}',
              'LCRET-SPSNDP2DPRate': 'LCR-ET_{SP,SN,DN1,DP1}', 'LCRET-SPSN': 'LCR-ET_{SP,SN}', 'NNL-DB': 'NNL-DB',
              'random': 'random'}

dic_labels = {'link_change_rate': 'LCR', 'link_change_rate_ET': 'LCR-ET', 'related_link_change_rate': 'LCR-R',
              'content_change_rate': 'CCR', 'num_new_outlinks_avg_history': 'NNL-Av',
              'num_new_outlinks_ET': 'NNL-ET', 'num_new_outlinks_week_10': 'NNL',
              'prob_of_new_link': 'PNL-ET', 'NGB_preds': 'NNL-NGB', 'NGB_gener': 'ngb-gener',
              'new_link_using_ET': 'NL-ET', 'new_link_in_week_10': 'NL', 'related_num_new_outlinks': 'NNL-R',
              'LCR': 'LCR', 'LCRET-SPSNDPRate': r'LCR-ET_{SP,SN,DP0}',
              'LCRET-SPSNDP2DPRate': 'LCR-ET_{SP,SN,DP2,RLCR}', 'LCRET-SPSN': 'LCR-ET', 'NNL-DB': 'NNL-Pr',
              'random': 'random'}


def calc_area(a, b):
    a1 = np.array(a)
    b1 = np.array(b)
    area = 0
    for i in range(len(a1) - 1):
        mn = min(b1[i + 1], b1[i])
        # mx = max(b1[i + 1], b1[i])
        w = (a1[i + 1] - a1[i]) / 100
        area += (mn * w) + (w * abs(b1[i + 1] - b1[i]) / 2)
    return area


def plot(dff, file_name, target):
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    cols = dff.columns.values
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [i * 100 / len(dff[cols[0]]) for i in range(1, len(dff[cols[0]]) + 1)]
    z = []
    for col in cols:
        z.append((col, calc_area(x, dff[col])))
    z.sort(key=lambda element: element[1], reverse=True)
    means = dff.mean()
    spman_corr = [val for col, val in z if col != 'random']
    norm = plt.Normalize(min(spman_corr) * 0.8, max(spman_corr) * 1.1)
    # norm = plt.Normalize(0.2, 0.6)
    colors = plt.cm.Greens(norm(spman_corr))
    colors = np.append(colors, np.array([[0.75, 0.75, 0.75, 1]]), axis=0)  # for random color: grey
    # means = dff.mean()
    # spman_corr = list(means[:-1])
    # norm = plt.Normalize(min(spman_corr) * 0.8, max(spman_corr) * 1.1)
    # # norm = plt.Normalize(0.2, 0.6)
    # colors = plt.cm.Greens(norm(spman_corr))
    # colors = np.append(colors, np.array([[0.75, 0.75, 0.75, 1]]), axis=0)  # for random color: grey

    idx = 0
    for e in z:
        # ar = calc_area(x, dff[col])
        col = e[0]
        ar = e[1]
        # lbl = '$' + dic_labels[col] + '$' + ': area=' + str(round(ar, 3))
        lbl = dic_labels[col]
        idx_ = lbl.find('_')
        if idx_ >= 0:
            lbl = lbl[:idx_] + r'$' + lbl[idx_:] + '$'
            # lbl = lbl[:idx_] + r'$\mathregular' + lbl[idx_+1:] + '$'
        lbl = lbl + ': area=' + str(round(ar, 3))
        z = 3
        if 'ET' in col:
            z = 50
        elif col == 'NNL-DB':
            z = 40
        elif col == 'random':
            lbl = 'random: area=0.5'
            z = -1
        c = colors[idx] if dic_labels[col] not in ['NNL-Pr', 'NNL-Av', 'CCR'] else 'orange'
        ax.plot(x, dff[col] * 100, '--', color=c, linewidth=1, label=lbl,
                marker=markers[idx], markersize=4,
                zorder=z)
        # ax.plot(x, dff[col] * 100, '--', color=colors[idx], linewidth=1, label=dic_labels[col], marker=markers[idx], markersize=4)
        idx += 1
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())  # put unit % to x-axis
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # put unit % to y-axis
    plt.xlabel('% of hot pages selected from ' + 'each ranking')
    plt.ylabel('% of hot pages exist in the ' + target + ' ranking')

    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap='Greens', norm=norm)
    ## sm.set_array([])
    ## cbar = plt.colorbar(sm, orientation="horizontal")
    # cbar = plt.colorbar(sm)
    ## cbar.set_label('Color', rotation=270, labelpad=-30, horizontalalignment='right')
    ## ticklabels = cbar.ax.get_ymajorticklabels()
    ## ticks = list(cbar.get_ticks())

    ## Append the ticks (and their labels) for minimum and the maximum value
    ## cbar.set_ticks([min(spman_corr), max(spman_corr)])
    # cbar.set_ticks([])
    ## cbar.set_ticklabels(['The worst', 'The best'])
    # cbar.ax.set_title('The best', fontsize=7, pad=-10)
    # cbar.ax.set_xlabel('The worst', fontsize=7)

    plt.grid(color='0.95', linestyle='--')
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(file_name + ".png", dpi=500, bbox_inches='tight')
    plt.close()
    ## plt.show()


output_path = ('nonZero' if remove_zero_values else 'all') + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

for InEx in InExs:
    fn = InEx + '_orders-NGB.csv'
    df_total = pd.read_csv(fn)
    df_total = df_total.apply(lambda x: pd.Series.round(x, round_decimal))
    for target in targets:
        df_target = df_total.copy()
        if 'num_new' in target:
            df_target = df_total.apply(lambda x: pd.Series.round(x, 0))
        attributes = dic_attributes[target]
        dic_results = {}
        # for gap in gap_between_records:

        if remove_zero_values:
            df_target = df_target[df_target[target] > 0]
        gap = len(df_target) // 30  # 30 is the number of points in the chart
        print("\npreparing for gap: " + str(gap) + ' --> ' + InEx + ' ' + dic_labels[target] + '\n', '-' * 80)

        fn_write = output_path + InEx + separator_in_filename + 'precisionAtK_ranking' + dic_labels[
            target] + separator_in_filename + ('nonZero' if remove_zero_values else 'all')
        for att in attributes:
            iter_list = []
            for seed in range(num_iter_avg):
                random.seed(seed)
                df = df_target[[att, target]]
                df_sort_target = df.sort_values([target])
                df_sort_target['index1'] = [(i + 1) for i in range(len(df))]
                # in case of tied data, apply the random selection for more reliablity
                df_sort_target['index_random'] = random.sample(range(len(df)), len(df))
                df_sort_target['att_round'] = round(df_sort_target[att], round_decimal)
                a = df_sort_target.sort_values(['att_round', 'index_random'])
                # if some samples have the same rank, make a random selection between them
                a['index2'] = [(i + 1) for i in range(len(a))]
                a = a.sort_values(['index1'])
                lst = []
                for i in range(len(a) - gap, 0, -gap):
                    if (i - gap) < 0:
                        i = 0
                    d1 = a[a['index1'] >= i]
                    n1 = len(d1)
                    n2 = len(d1[d1['index2'] >= i])
                    lst.append(n2 / n1)
                iter_list.append(lst)
            average_list = []
            for i in range(len(iter_list[0])):
                ss = 0
                for row in iter_list:
                    ss += row[i]
                ss /= num_iter_avg
                average_list.append(ss)
            dic_results[att] = average_list
            print("writing " + att + " ok")

        total_data_cnt = len(dic_results[attributes[0]])
        dic_results['random'] = [i / total_data_cnt for i in range(1, total_data_cnt + 1)]
        s = ''
        for i in range(total_data_cnt):
            k = ''
            for key in dic_results.keys():
                k = k + str(dic_results[key][i]) + ','
            s = s + k[:-1] + '\n'
        s = ','.join(attributes) + ',random\n' + s
        with open(fn_write + '.csv', 'w') as f:
            f.write(s)
            f.close()

        # later added
        df = pd.read_csv(fn_write + '.csv')
        plot(df, fn_write, dic_labels[target])

# import glob
# import shutil
# for file in glob.glob(output_path + '/*4000*.png'):
#     print(file)
#     shutil.copy(file, output_path + 'final/')

print("Finished")
