import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Set font to support English only
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# File path
file_path = r'C:\Users\沐阳\Desktop\城市得分结果_pro\Structural change of Chongqing.xlsx'

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    print("Please check if the file path is correct.")
else:
    print(f"Reading file: {file_path}")

    # Read Excel file
    try:
        # Read data, assume first column is Year
        df = pd.read_excel(file_path)

        # Check data
        print(f"Data read successfully, shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Preview first few rows
        print("\nPreview first 5 rows:")
        print(df.head())

        # Check if 'Year' column exists
        if 'Year' not in df.columns:
            print("\nWarning: 'Year' column not found.")
            print("Attempting to use the first column as Year...")
            # Try renaming the first column to 'Year'
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'Year'})
            print(f"Renamed column '{first_col}' to 'Year'")

        # Set Year as index
        df.set_index('Year', inplace=True)

        # Get numeric columns (exclude possible non-numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # If no numeric columns, try converting all columns to numeric
        if not numeric_columns:
            print("Attempting to convert all columns (except Year) to numeric...")
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        print(f"\nNumber of numeric columns (features): {len(numeric_columns)}")

        if not numeric_columns:
            print("Error: No numeric columns found for plotting.")
        else:
            # Use the last year (assume index sorted by year)
            df = df.sort_index()  # ensure sorted by Year
            last_year = df.index.max()
            print(f"\nLast year: {last_year}")

            # Get data for the last year
            last_year_data = df.loc[last_year]

            # Sort features by descending value for the last year and take top 10
            last_year_data_sorted = last_year_data[numeric_columns].sort_values(ascending=False)

            # Top N features
            top_n = 10
            last_year_data_top10 = last_year_data_sorted.head(top_n)
            sorted_features = last_year_data_top10.index.tolist()

            print(f"\nTop {top_n} features in {last_year}:")
            for i, feature in enumerate(sorted_features):
                print(f"{i + 1}. {feature}: {last_year_data_top10[feature]:.6f}")

            # Compute mean for each selected feature
            feature_means = df[sorted_features].mean()

            # Create figure - fixed for up to 10 features
            n_features = min(top_n, len(sorted_features))
            fig = plt.figure(figsize=(12, 2 * n_features + 2))

            # Grid layout
            gs = fig.add_gridspec(n_features, 2, width_ratios=[0.5, 0.5],
                                  hspace=0.4, left=0.1, right=0.95, top=0.95, bottom=0.1)

            # Colors using viridis colormap
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_features))

            # 1. Bar chart (left)
            ax_bar = fig.add_subplot(gs[:, 0])
            bars = ax_bar.barh(sorted_features, last_year_data_top10.values,
                               color=colors, edgecolor='black', linewidth=0.5)

            # Make y-axis feature labels bold (safer approach compatible with matplotlib versions)
            ax_bar.set_yticks(range(len(sorted_features)))
            ax_bar.set_yticklabels(sorted_features, fontsize=14)
            for label in ax_bar.get_yticklabels():
                label.set_fontweight('bold')

            ax_bar.set_xlabel('Rate', fontsize=16, fontweight='bold')
            ax_bar.invert_yaxis()
            ax_bar.grid(True, axis='x', linestyle='--', alpha=0.5)
            ax_bar.set_axisbelow(True)

            # Hide bar chart spines
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['left'].set_visible(False)
            ax_bar.spines['bottom'].set_visible(False)

            # Add value labels on bars
            max_value = last_year_data_top10.max()
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width < max_value * 0.1:
                    ha = 'left'
                    xpos = width + max_value * 0.01
                else:
                    ha = 'right'
                    xpos = width - max_value * 0.01

                if abs(width) < 0.001:
                    formatted_value = f'{width:.2e}'
                else:
                    formatted_value = f'{width:.4f}'

                ax_bar.text(xpos, bar.get_y() + bar.get_height() / 2,
                            formatted_value, ha=ha, va='center', fontsize=12, fontweight='bold')

            # 2. Trend plots for each feature (right)
            for i, feature in enumerate(sorted_features):
                ax_trend = fig.add_subplot(gs[i, 1])

                # Get trend data
                trend_data = df[feature].values

                # Plot trend line
                ax_trend.plot(df.index, trend_data,
                              linestyle='-',
                              linewidth=2,
                              color='black',
                              marker='o',
                              markersize=4,
                              markerfacecolor=colors[i],
                              markeredgecolor='black')

                # Plot mean line (dashed)
                mean_value = feature_means[feature]
                ax_trend.axhline(y=mean_value, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

                # Mean label
                if abs(mean_value) < 0.001:
                    mean_formatted = f'{mean_value:.2e}'
                else:
                    mean_formatted = f'{mean_value:.4f}'

                ax_trend.text(df.index[-1] + (df.index[-1] - df.index[0]) * 0.05,
                              mean_value, f'      {mean_formatted}',
                              fontsize=12, va='center', ha='left',fontweight="bold")

                # X-axis limits
                ax_trend.set_xlim(df.index[0] - 0.5, df.index[-1] + (df.index[-1] - df.index[0]) * 0.1)

                # Hide spines
                ax_trend.spines['top'].set_visible(False)
                ax_trend.spines['right'].set_visible(False)
                ax_trend.spines['left'].set_visible(False)
                ax_trend.spines['bottom'].set_visible(False)

                # Hide y-tick labels and tick lines
                ax_trend.set_yticklabels([])
                ax_trend.tick_params(axis='y', which='both', length=0)

                # Grid and background
                ax_trend.grid(True, linestyle='--', alpha=0.3)
                ax_trend.set_facecolor('#f5f5f5')

                # Hide x-axis tick lines
                ax_trend.tick_params(axis='x', which='both', length=0)

                # Determine x-tick step based on number of years
                years = df.index.tolist()
                if len(years) > 12:
                    tick_step = 4
                elif len(years) > 8:
                    tick_step = 3
                elif len(years) > 4:
                    tick_step = 2
                else:
                    tick_step = 1

                tick_positions = years[::tick_step]
                if tick_positions[-1] != years[-1]:
                    tick_positions.append(years[-1])

                ax_trend.set_xticks(tick_positions)
                ax_trend.set_xticklabels([])

                # Only the bottom trend plot shows x-axis labels
                if i == n_features - 1:
                    ax_trend.set_xlabel('Year', fontsize=16, fontweight="bold")
                    ax_trend.set_xticklabels([str(int(year)) for year in tick_positions])
                else:
                    ax_trend.set_xlabel('')

            # Overall title
            plt.suptitle(f'Top {top_n} Features in {last_year} - {os.path.basename(file_path).split(".")[0]}',
                         fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(wspace=0.02)
            # Save figure
            output_path = file_path.replace('.xlsx', f'_top{top_n}_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {output_path}")

            plt.show()

            # Print detailed stats for top features
            print(f"\n{'=' * 60}")
            print(f"Detailed statistics for top {top_n} features:")
            print('=' * 60)
            for i, feature in enumerate(sorted_features):
                feature_data = df[feature]
                mean_val = feature_means[feature]
                max_val = feature_data.max()
                min_val = feature_data.min()
                max_year = feature_data.idxmax()
                min_year = feature_data.idxmin()
                last_val = last_year_data_top10[feature]

                print(f"\n{i + 1}. {feature}:")
                print(f"   Last year ({last_year}) value: {last_val:.6f}")
                print(f"   Historical mean: {mean_val:.6f}")
                print(f"   Historical max: {max_val:.6f} (Year: {max_year})")
                print(f"   Historical min: {min_val:.6f} (Year: {min_year})")
                print(f"   Trend: {'increasing' if last_val > feature_data.iloc[0] else 'decreasing'} "
                      f"(from {feature_data.iloc[0]:.6f} to {last_val:.6f})")

    except Exception as e:
        print(f"Error reading file: {e}")
        print("Please check the Excel file format.")