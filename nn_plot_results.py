import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib	import Path
import tabulate
from pprint import pprint
from util import DATADIR, setup_mplt, PLOTDIR

import scipy.stats as st


if __name__ == "__main__":
    setup_mplt()
    resutls_path = DATADIR / 'results.pkl'
    df_results = pd.read_pickle(resutls_path)

    # Index(['experiment_name', 'train_actual', 'train_preditions', 'train_r2',
    #        'train_mae', 'test_actual', 'test_predittions', 'test_r2', 'test_mae',
    #        'test_r2_avg', 'test_mae_avg', 'train_r2_avg', 'train_mae_avg'],
    #print(df.columns)

    print(tabulate.tabulate(df_results[['experiment_name',
                   'train_r2_avg', 'test_r2_avg',
                   'train_mae_avg', 'test_mae_avg',

                    ]], headers='keys', tablefmt='psql', floatfmt=".4f"))


    target_names = ['Caff', 'CGA', 'TDS']

    #exp = 'CoffeeNetOX-NOISE20-0.005-nobins-256-3-1000'\
    #exp = 'CoffeeNetOX-NOISE100-0.001-nobins-256-3-1000'
    #exp = 'CoffeeNetOX-nobins-3-300'
    #exp = 'CoffeeNetOX-NOISE20-0.005-nobins-256-3-1000'

    #exp = 'CoffeeNetOX-NOISE100-0.05-nobins-256-3-1000'

    #exp = 'CoffeeNetOX-NOISE100-0.05-nobins-256-3-1000'
    #exp = 'CoffeeNetOX-NOISE30-0.5-nobins-256-3-1000'
    #exp = df_best['experiment_name']
    #df_best = df.iloc[0]
    #df_best = df[df['experiment_name'] == exp]

    #exp = 'CoffeeNetOX-nobins-1024-128-64-128-3-2000'


    for exp in ['CoffeeNetOX-NOISE100-0.05-nobins-256-3-1000',
                'CoffeeNetOX-NOISE30-0.5-nobins-256-3-1000',
                'CoffeeNetOX-NOISE10-0.001-nobins-256-3-1000',
                'CoffeeNetOX-nobins-256-3-500']:

        df_best = df_results.loc[df_results['experiment_name'] == exp].iloc[0]


        print ('Using model', exp)


        #print(df_best)



        results = {
            'test_r2': df_best['test_r2'],
            'test_mae': df_best['test_mae'],
            'train_r2': df_best['train_r2'],
            'train_mae': df_best['train_mae'],
        }
        for run in ['train', 'test']:
            for mode in ['actual', 'predictions']:
                cols = np.array(df_best[f'{run}_{mode}'].tolist()).T
                for name, col in zip(target_names, cols):
                    if name == 'TDS':
                        col *= 10000
                    results[f'{run}_{name}_{mode}']= col

        #pprint(results)

        # not actually a dataframe, kept in to match the old code.
        df = results



        # df['test_Caff_err'] = df['test_Caff_actual'] -df['test_Caff_predictions']
        # df['test_CGA_err'] = df['test_CGA_actual'] -df['test_CGA_predictions']

        # df['test_TDS_actual_ppm']= df['test_TDS_actual'] * 10000
        # df['test_TDS_predictions_ppm'] = df['test_TDS_predictions'] * 10000
        # df['test_TDS_err_ppm'] = df['test_TDS_actual_ppm'] -df['test_TDS_predictions_ppm']

        # df['train_TDS_actual_ppm']= df['train_TDS_actual'] * 10000
        # df['train_TDS_predictions_ppm'] = df['train_TDS_predictions'] * 10000
        # df['train_TDS_err_ppm'] = df['train_TDS_actual_ppm'] -df['train_TDS_predictions_ppm']



        # train_actual = combine(df_best, ['train_HPLC_Caff_actual', 'train_HPLC_CGA_actual', 'train_TDS_actual_ppm'])
        # train_predictions = combine(df_best, ['train_HPLC_Caff_predictions', 'train_HPLC_CGA_predictions', 'train_TDS_predictions_ppm'])


        train_actual = np.array( [df[f'train_{name}_actual'] for name in target_names]).T
        train_predictions = np.array( [df[f'train_{name}_predictions'] for name in target_names]).T

        actual = np.array( [df[f'test_{name}_actual'] for name in target_names]).T
        predictions = np.array( [df[f'test_{name}_predictions'] for name in target_names]).T



        # actual = combine(df_best, ['test_HPLC_Caff_actual', 'test_HPLC_CGA_actual', 'test_TDS_actual_ppm'])
        # predictions = combine(df_best, ['test_HPLC_Caff_predictions', 'test_HPLC_CGA_predictions', 'test_TDS_predictions_ppm'])


        #print(predictions.shape)
        #print(actual.shape)



        err_ppm = ((predictions-actual))
        err_pct = 100.0 * (predictions - actual) / actual


        # print(df_best[['test_HPLC_Caff_mae', 'test_HPLC_CGA_mae', 'test_TDS_mae']])
        # print(df_best[['test_HPLC_Caff_mae', 'test_HPLC_CGA_mae', 'test_TDS_mae']].agg('mean').values)
        # exit()

        #r2 = df_best[['test_HPLC_Caff_r2', 'test_HPLC_CGA_r2', 'test_TDS_r2']].agg('mean').values
        #mae = df_best[['test_HPLC_Caff_mae', 'test_HPLC_CGA_mae', 'test_TDS_mae']].agg('mean').values
        r2 = df['test_r2']
        mae = df['test_mae']



        #convert TDS mae from percent to ppm
        mae[2] *= 10000


        #train_r2 =df_best[['train_HPLC_Caff_r2', 'train_HPLC_CGA_r2', 'train_TDS_r2']].agg('mean').values
        #train_mae = df_best[['train_HPLC_Caff_mae', 'train_HPLC_CGA_mae', 'train_TDS_mae']].agg('mean').values

        train_r2 = df['train_r2']
        train_mae = df['train_mae']


        train_mae[2] *= 10000


        target_names = ['Caffeine', 'CGA', 'TDS']


        # compute confidence interval on errors
        mean_err = np.mean(err_ppm, axis=0)
        std_err = np.std(err_ppm, axis=0, ddof=1)

        n = err_ppm.shape[0]

        confidence_level = 0.95
        t_value = st.t.ppf((1 + confidence_level) / 2, n - 1)

        valrange = t_value * std_err / np.sqrt(n)
        lower_bound = mean_err - valrange
        upper_bound = mean_err + valrange


        q1_err = np.percentile(err_ppm, 25, axis=0)
        q3_err = np.percentile(err_ppm, 75, axis=0)
        iqr_err = q3_err - q1_err

        r = []
        for i, name in enumerate(target_names):
            r.append({
                'Param': name,
                'Mean': mean_err[i],
                'Std': std_err[i],
                'Lower': lower_bound[i],
                'Upper': upper_bound[i],
                'IQR': iqr_err[i],
                'max': np.max(err_ppm[:,i]),
                'min': np.min(err_ppm[:,i]),
                f'{confidence_level}%range': valrange[i],
                f'{confidence_level}%ci': f"{mean_err[i]:.1f} +/- {valrange[i]:.1f}"
            })

        print(tabulate.tabulate(r, floatfmt=".1f", headers='keys', tablefmt='psql'))

        # this creates the scatter plots of data
        if 1:

            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            #fig.suptitle(f'Model Evaluation on Test Data ({MODEL_NAME})', fontsize=16)

            #fig.suptitle("Title this", fontsize=16)
            for i, name in enumerate(target_names):

                axes[i].scatter(train_actual[:, i], train_predictions[:, i], marker='o',
                                edgecolors='gray',
                                color = 'lightblue', alpha=0.9, label='Train')

                axes[i].scatter(actual[:, i], predictions[:, i], marker='o',
                                edgecolors='gray',
                                color = 'coral', alpha=0.7, label='Test')

                axes[i].plot([actual[:, i].min(), actual[:, i].max()],
                            [actual[:, i].min(), actual[:, i].max()], 'r--', label='Ideal')

                axes[i].set_xlabel(f"Actual {name} (ppm)")
                axes[i].set_ylabel(f"Predicted {name} (ppm)")

                axes[i].set_title(f"Actual vs. Predicted {name}\nTest R2: {r2[i]:.4f}, Test MAE: {mae[i]:.4f}\nTrain R2: {train_r2[i]:.4f}, Train MAE: {train_mae[i]:.4f}",)
                axes[i].grid(True)
                axes[i].legend()

            #plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plt.tight_layout()
            plt.savefig(PLOTDIR / f"{exp}_scatter.pdf", bbox_inches='tight', dpi=600)
            #plt.show()

        # the box plot of errors / r2
        if 1:



            fig_violin, ax_violin_main = plt.subplots(figsize=(9, 5)) # Adjusted figure size
            ax_violin_main.violinplot(err_ppm[:,:2], positions=[1, 2], showmeans=True, showmedians=False, widths=0.8)


            #ax_violin_main.violinplot(err_pct, positions=[1, 2, 3], showmeans=True, showmedians=False, widths=0.8)
            #ax_violin_main.violinplot(err_ppm[:,:2], positions=[1, 2], showmeans=True, showmedians=False, widths=0.8)


            # allrows = np.array([])
            # for i, row in enumerate(df_best.loc[:,'test_Caff_err'].values):
            #     print(row)
            #     ax_violin_main.violinplot(row, positions=[i+1], showmeans=True, showmedians=False, widths=0.8)
            #     allrows = np.append(allrows, row)

            # ax_violin_main.violinplot(allrows, positions=[i+2], showmeans=True, showmedians=False, widths=0.8)
            # ax_violin_main.violinplot(err_ppm[:,0], positions=[i+4], showmeans=True, showmedians=False, widths=0.8)

            #exit()
            #ax_violin_main.violinplot(, positions=[1, 2], showmeans=True, showmedians=False, widths=0.8)

            ax_violin_second = ax_violin_main.twinx()
            ax_violin_second.violinplot(err_ppm[:,2], positions=[3], showmeans=True, showmedians=False, widths=0.8)
            ax_violin_second.set_ylim(-6600,6600)
            ax_violin_second.set_ylabel('ppm (TDS)')

            ax_violin_main.set_ylabel('ppm (Caffeine, CGA)')
            ax_violin_main.tick_params(axis='y',)

            # set minor ticks to 100
            ax_violin_main.yaxis.set_minor_locator(plt.MultipleLocator(100))
            ax_violin_main.set_ylim(-660,660)
            ax_violin_main.set_xticks(np.arange(1, len(target_names) + 1))
            ax_violin_main.set_xticklabels(target_names)
            ax_violin_main.set_title('Distribution of Prediction Error')
            #ax_violin_main.grid(True, linestyle='--', alpha=0.7, axis='x') # Grid for x-axis from main
            ax_violin_main.grid(True, linestyle='--', alpha=0.7)


            # for i, name in enumerate(target_names):

            #     print(f"{name} Mean error   {np.mean(err_ppm[:, i]):.4f}")
            #     print(f"{name} Median error {np.median(err_ppm[:, i]):.4f}")
            #     print(f"{name} Max error    {np.max(err_ppm[:, i]):.4f}")
            #     print(f"{name} Min error    {np.min(err_ppm[:, i]):.4f}")
            #     print(f"{name} std error    {np.std(err_ppm[:, i]):.4f}")

            plt.tight_layout()
            plt.savefig(PLOTDIR / f"{exp}_error_violin.pdf", bbox_inches='tight', dpi=600)

        # plot distributions of data
        if 1:

            fig_data, ax_data = plt.subplots(figsize=(9, 5))

            ax_data.set_title('Distribution of Data')

            act_plot = ax_data.violinplot(
                actual[:,:2],
                positions=[0.8,1.8],
                widths=[0.5,0.5],
                showmeans=True,
                showmedians=False,

            )

            pred_plot = ax_data.violinplot(
                predictions[:,:2],
                positions=[1.2, 2.2],
                widths=[0.5, 0.5],
                showmeans=True,
                showmedians=False,

            )

            ax_data.set_ylabel('ppm (Caffeine, CGA)')
            ax_data.tick_params(axis='y',)

            ax_data.set_ylim(-300,2200)
            ax_data.set_xticks(np.arange(1, len(target_names) + 1))


            ax_data.legend([act_plot['bodies'][0], pred_plot['bodies'][0]],
                        ['Actual', 'Predicted'],
                        loc='lower right')
            ax_2 = ax_data.twinx()

            ax_2.violinplot(
                actual[:,2],
                positions=[2.8],
                showmeans=True,
                showmedians=False
            )

            ax_2.violinplot(
                predictions[:,2],
                positions=[3.2],
                showmeans=True,
                showmedians=False
            )

            ax_2.set_ylim(-3000,22000)
            ax_2.set_ylabel('ppm (TDS)')

            ax_data.set_xticklabels(target_names)
            ax_data.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(PLOTDIR / f"{exp}_data_distribution.pdf", bbox_inches='tight', dpi=600)
        #plt.show()

