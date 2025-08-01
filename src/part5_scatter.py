'''
PART 5: SCATTER PLOTS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part5_plots`
'''

import seaborn as sns
import matplotlib.pyplot as plt
import src.part1_etl as part1

# 1. Using lmplot, create a scatter plot where the x-axis is the prediction for felony and the y-axis the is prediction for a nonfelony, and hue this by whether the current charge is a felony.

def scatterplot_felony_nonfelony_scatter(merged_charges):
    """
    Creates a scatter plot with predictions for felony and nonfelony rearrest and hues by whether the current charge is a felony.

    Parameters:
    - merged_charges: DataFrame containing the merged prediction data and felony charge status.

    Returns:
    - A scatter plot with regression line, showing the relationship between felony and nonfelony rearrest predictions.
    """
    plt.clf()  # clears the current figure

    # creates the scatter plot using lmplot
    scatter_plot = sns.lmplot(
        data=merged_charges,
        x='prediction_felony', 
        y='prediction_nonfelony',  
        hue='has_felony_charge', 
        markers=['o', 'x'], 
        height=5,
        aspect=1.5
    )

    # sets up labels
    scatter_plot.set_axis_labels('Prediction for Felony Rearrest', 'Prediction for Nonfelony Rearrest')
    scatter_plot.fig.suptitle('Scatter Plot: Felony vs Nonfelony Rearrest Predictions')

    plt.savefig('data/part5_plots/felony_nonfelony_scatter.png', bbox_inches='tight')
    print("Part 5 Answers:")
    print("What can you say about the group of dots on the right side of the plot?")
    print("The group of dots on the right side tells me that felons are more likely to be rearrested for felony crimes.\n")


# 2. Create a scatterplot where the x-axis is prediction for felony rearrest and the y-axis is whether someone was actually rearrested.
def scatterplot_felony_rearrest_vs_actual(merged_charges):
    """
    Creates a scatter plot where the x-axis is the prediction for felony rearrest and the y-axis is the actual rearrest outcome.

    Parameters:
    - merged_charges: DataFrame containing the merged prediction data and actual rearrest data.

    Returns:
    - A scatter plot showing the relationship between the predicted and actual felony rearrest.
    """
    plt.clf()  # clears the current figure

    scatter_plot = sns.scatterplot(
        data=merged_charges,
        x='prediction_felony',  
        y='y_felony',  
        hue='has_felony_charge',  
        palette='Set1',
        marker='o',
        s=100 
    )

    # sets up labels
    scatter_plot.set_xlabel('Prediction for Felony Rearrest')
    scatter_plot.set_ylabel('Actual Rearrested for Felony')
    scatter_plot.set_title('Scatter Plot: Predicted vs Actual Felony Rearrest')

    plt.savefig('data/part5_plots/felony_rearrest_vs_actual.png', bbox_inches='tight')
    print("Would you say based off of this plot if the model is calibrated or not?")
    print("Based off this scatterplot, the model is not well calibrated. It clusters probabilties in the extreme values of either 0 or 1.")

def main():
    pred_universe, arrest_events, charge_counts, charge_counts_by_offense, felony_charge, merged_charges = part1.extract_transform()
    
    
    scatterplot_felony_nonfelony_scatter(merged_charges)
    scatterplot_felony_rearrest_vs_actual(merged_charges)


if __name__ == "__main__":
    main()

