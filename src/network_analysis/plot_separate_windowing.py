import pandas as pd
from pathlib import Path
import sys
from tabulate import tabulate


def analyze_best_windows(file_path):
    """
    Loads the voltage window search results and finds the best average
    test MAE score and the corresponding voltage window for each target.

    Best = lowest average MAE across folds.
    """
    try:
        results_path = Path(file_path)

        if not results_path.exists():
            print(f"Error: Results file not found at {results_path.resolve()}")
            print("Please ensure the path to 'voltage_window_search_results.pkl' is correct.")
            sys.exit(1)

        print(f"--- Analyzing results from: {results_path.name} ---\n")
        df_results = pd.read_pickle(results_path)

        targets = ['HPLC_Caff', 'HPLC_CGA', 'TDS']
        best_results = []

        for target_name in targets:
            r2_column = f'test_{target_name}_r2'
            mae_column = f'test_{target_name}_mae'

            # Filter results for this target
            df_target = df_results[df_results['target'] == target_name].copy()

            # Group mean metrics per window
            df_avg = df_target.groupby(['v_start', 'v_end', 'v_window_size']).agg({
                r2_column: 'mean',
                mae_column: 'mean'
            }).reset_index()

            # 3. Find best window = lowest MAE
            if not df_avg.empty:
                best_row = df_avg.loc[df_avg[mae_column].idxmin()]

                best_results.append({
                    'Target': target_name,
                    'V_Start': f"{best_row['v_start']:.2f} V",
                    'V_End': f"{best_row['v_end']:.2f} V",
                    'Window_Size': f"{best_row['v_window_size']:.2f} V",
                    'Lowest_Avg_Test_MAE': best_row[mae_column],
                    'Avg_Test_R2': best_row[r2_column]
                })
            else:
                best_results.append({
                    'Target': target_name,
                    'V_Start': 'N/A',
                    'V_End': 'N/A',
                    'Window_Size': 'N/A',
                    'Lowest_Avg_Test_MAE': float('inf'),
                    'Avg_Test_R2': 'N/A'
                })

        # Print summary
        print("=" * 80)
        print("Summary of Best Performing Voltage Windows (Based on Lowest Average Test MAE)")
        print("=" * 80)

        df_summary = pd.DataFrame(best_results)

        # Sort by LOWEST MAE (ascending)
        df_summary.sort_values(by='Lowest_Avg_Test_MAE', ascending=True, inplace=True)

        print(
            tabulate(df_summary,
                     headers='keys',
                     tablefmt='psql',
                     showindex=False,
                     floatfmt=(".4f", ".4f"),
                     numalign="right")
        )

        print(
            "\nNote: MAE values represent the average MAE across all 'Leave-One-Coffee-Out' folds for that voltage window."
        )

    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")
        print("Check that the file is not corrupted and the column names match the expected structure.")


if __name__ == "__main__":
    DATADIR = Path('../data')
    results_file = 'voltage_window_search_results.pkl'
    results_path = DATADIR / results_file

    analyze_best_windows(results_path)
