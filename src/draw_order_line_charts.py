import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

a = ['num_new_outlinks_avg_history', 'num_new_outlinks_ET', 'link_change_rate',
     'related_link_change_rate', 'link_change_rate_ET', 'content_change_rate',
     'prob_of_new_link', 'new_link_using_ET', 'new_link_in_week_10',
     'num_new_outlinks_week_10']

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
         'related_link_change_rate', 'link_change_rate_ET',
         'content_change_rate']
}

remove_zero_values = False
# remove_zero_values = True
InExs = ['External', 'Internal']
targets = ['num_new_outlinks_week_10', 'link_change_rate', 'new_link_in_week_10']
# targets = ['num_new_outlinks_week_10']
round_decimal = 2
# gap_between_records = [500, 1000, 1500, 2000, 2500, 3000]
gap_between_records = [1500, 2000, 2500, 3000, 4000, 5000]
gap_between_records = [2500, 3000, 4000]
gap_between_records = [4000]
separator = '-'

markers = ['*', 'o', 'v', '.', 'p', '+', 'd', 'x']
dic_labels = {'link_change_rate': 'LCR', 'link_change_rate_ET': 'LCRET', 'related_link_change_rate': 'RLCR',
              'content_change_rate': 'CCR', 'num_new_outlinks_avg_history': 'NNLAH',
              'num_new_outlinks_ET': 'NNLET', 'num_new_outlinks_week_10': 'NNLW10',
              'prob_of_new_link': 'PNL', 'NGB_preds': 'NGB', 'NGB_gener': 'ngb-gener',
              'new_link_using_ET': 'NLET', 'new_link_in_week_10': 'NLW10', 'related_num_new_outlinks': 'RNNL',
              'random': 'random'}


def plot(dff, file_name, target):
    cols = dff.columns.values
    fig, ax = plt.subplots(figsize=(6, 4))
    xx = dff.mean()
    spman_corr = list(xx[:-1])
    norm = plt.Normalize(min(spman_corr), max(spman_corr))
    norm = plt.Normalize(min(spman_corr), 1.1)
    norm = plt.Normalize(0.5, 1.1)
    #     colors = plt.cm.RdYlGn(norm(spman_corr))
    colors = plt.cm.copper_r(norm(spman_corr))
    # colors = plt.cm.RdYlGn(norm(spman_corr))
    # colors = plt.cm.Greens(norm(spman_corr))
    colors = np.append(colors, np.array([[0.75, 0.75, 0.75, 1]]), axis=0)  # for random color: grey
    x = [i * 100 / len(dff[cols[0]]) for i in range(1, len(dff[cols[0]]) + 1)]
    idx = 0
    for col in cols:
        ax.plot(x, dff[col] * 100, '--', color=colors[idx], linewidth=1, label=dic_labels[col], marker=markers[idx],
                markersize=4)
        idx += 1
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())  # put unit % to x-axis
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())  # put unit % to y-axis
    plt.xlabel('% of hot pages selected from ' + 'different orders')
    plt.ylabel('% of hot pages exists in the ' + target + ' order')

    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap='copper_r', norm=norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm, orientation="horizontal")
    cbar = plt.colorbar(sm)
    # cbar.set_label('Color', horizontalalignment='right')
    # cbar.set_label('Color', rotation=270, labelpad=-30, horizontalalignment='right')
    # cbar.set

    # ticklabels = cbar.ax.get_ymajorticklabels()
    # ticks = list(cbar.get_ticks())

    # Append the ticks (and their labels) for minimum and the maximum value
    # cbar.set_ticks([min(spman_corr), max(spman_corr)])
    cbar.set_ticks([])
    # cbar.set_ticklabels(['The worst', 'The best'])
    cbar.ax.set_title('The best', fontsize=7, pad=-10)
    cbar.ax.set_xlabel('The worst', fontsize=7)
    # ax2.text(0.45, .94, 'The best', rotation=0)

    plt.grid(color='0.95', linestyle='--')
    plt.legend(loc='best', fontsize=11)
    plt.savefig(file_name + ".png", dpi=500, bbox_inches='tight')
    plt.close()
    #     plt.title('test')
    # plt.show()


output_path = r'd:/a/outputs/11/' + ('nonZero' if remove_zero_values else 'all') + '/'
if not os.path.exists(output_path):
    os.mkdir(output_path)
for InEx in InExs:
    fn = r'G:\WebInsight Datasets2\1M\1M pickle dataset 384323 instances doina\orders\\' + InEx + '_orders-NGB2.csv'
    df_total = pd.read_csv(fn)
    df_total = df_total.apply(lambda x: pd.Series.round(x, round_decimal))
    for target in targets:
        attributes = dic_attributes[target]
        dic_results = {}
        for gap in gap_between_records:
            print("\npreparing for gap: " + str(gap) + ' --> ' + InEx + ' ' + target + '\n', '-' * 80)

            fn_write = output_path + InEx + separator + dic_labels[target] + separator + 'round_' + str(
                round_decimal) + separator + 'gap_' + str(gap) + separator + (
                           'nonZero' if remove_zero_values else 'all')
            for att in attributes:
                df = df_total[[att, target]]
                if remove_zero_values:
                    df = df[df[target] > 0]
                df_sort_target = df.sort_values([target])
                df_sort_target['index1'] = [(i + 1) for i in range(len(df))]
                # df_sort_target_att = df_sort_target.sort_values(by=[att])
                # df_sort_target_att['index2'] = [(i + 1) for i in range(len(df))]
                # df_sort_target['rank_target'] = rankdata(df_sort_target[target], method='dense')
                # df_sort_target['rank_att'] = rankdata(round(df_sort_target[att], round_decimal), method='dense')
                df_sort_target['att_round'] = round(df_sort_target[att], round_decimal)
                a = df_sort_target.sort_values(['att_round', 'index1'])
                a['index2'] = [(i + 1) for i in range(len(a))]
                a = a.sort_values(['index1'])
                lst = []
                # for i in range(500, len(a), gap):
                #     d1 = a[a['index1'] <= i]
                #     n1 = len(d1)
                #     n2 = len(d1[d1['index2'] <= i])
                #     lst.append(n2 / n1)

                for i in range(len(a) - gap, 0, -gap):
                    d1 = a[a['index1'] >= i]
                    n1 = len(d1)
                    n2 = len(d1[d1['index2'] >= i])
                    lst.append(n2 / n1)

                dic_results[att] = lst
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
