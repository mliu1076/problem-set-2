'''
PART 4: CATEGORICAL PLOTS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part4_plots`
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.part1_etl as part1

##  UPDATE `part1_etl.py`  ##
# 1. The charge_no column in arrest events tells us the charge degree and offense category for each arrest charge. 
# An arrest can have multiple charges. We want to know if an arrest had at least one felony charge.
# 
# Use groupby and apply with lambda to create a new dataframe called `felony_charge` that has columns: ['arrest_id', 'has_felony_charge']
# 
# Hint 1: One way to do this is that in the lambda function, check to see if a charge_degree is felony, sum these up, and then check if the sum is greater than zero. 
# Hint 2: Another way to do thisis that in the lambda function, use the `any` function when checking to see if any of the charges in the arrest are a felony

# 2. Merge `felony_charge` with `pre_universe` into a new dataframe

# 3. You will need to update ## PART 1: ETL ## in main() to call these two additional dataframes

##  PLOTS  ##
# 1. Create a catplot where the categories are charge type and the y-axis is the prediction for felony rearrest. Set kind='bar'.
# Function to create catplot for felony rearrest prediction

def barplot_felony_rearrest(merged_charges):
    '''

    Creates a bar plot for felony rearrest predictions based on charge type.

    Parameters:
    - merged_charges: Dataframe containing the merged prediction data and felony charge status.

    Returns:
    - A bar plot for the felony rearrest predictions grouped by charge type.
    '''
    plt.clf()  # clears the current figure

    catplot = sns.catplot(
        data=merged_charges,
        x='has_felony_charge',  
        y='prediction_felony', 
        kind='bar', 
        height=5, 
        aspect=1.5
    )
    
    # sets labels
    catplot.set_xticklabels(['Misdemeanor', 'Felony'])
    catplot.set_axis_labels('Charge Type', 'Prediction for Felony Rearrest')
    
    plt.savefig('data/part4_plots/catplot_felony_rearrest.png')


def barplot_nonfelony_rearrest(merged_charges):
    '''

    Creates a bar plot for nonfelony rearrest predictions based on charge type.

    Parameters:
    - merged_charges: Dataframe containing the merged prediction data and felony charge status.

    Returns:
    - A bar plot for non-felony rearrest predictions grouped by charge type.
    '''
    plt.clf()  # clears the current figure

    catplot = sns.catplot(
        data=merged_charges,
        x='has_felony_charge', 
        y='prediction_nonfelony',  
        kind='bar', 
        height=5, 
        aspect=1.5
    )
    
    # sets labels
    catplot.set_xticklabels(['Misdemeanor', 'Felony'])
    catplot.set_axis_labels('Charge Type', 'Prediction for Nonfelony Rearrest')
    
    plt.savefig('data/part4_plots/catplot_nonfelony_rearrest.png')
    print("Part 4 Answers:")
    print("What might explain the difference between the plots?")
    print("One possible cause for the difference is because felons are more likely to commit crimes, " \
    "though the crimes they commit will be misdemeanors rather than felonies.\n")

def barplot_felony_rearrest_hue(merged_charges):
    '''
    Creates a bar plot for felony rearrest predictions with hue by actual felony rearrest outcome.

    Parameters:
    - merged_charges: Dataframe containing the merged prediction data and felony charge status.

    Returns:
    - A bar plot with hue based on actual rearrest outcome.
    '''
    plt.clf()  # clears the current figure
    merged_charges['rearrested_felony_label'] = merged_charges['y_felony'].map({1: 'Rearrested for Felony', 0: 'Not Rearrested for Felony'})

    catplot = sns.catplot(
        data=merged_charges,
        x='has_felony_charge', 
        y='prediction_felony', 
        hue='rearrested_felony_label',  
        kind='bar', 
        height=5, 
        aspect=1.5
    )
    
    # sets labels
    catplot.set_xticklabels(['Misdemeanor', 'Felony'])
    catplot.set_axis_labels('Charge Type', 'Prediction for Felony Rearrest')
    
    plt.savefig('data/part4_plots/catplot_felony_rearrest_with_hue.png')
    print("What does it mean that prediction for arrestees with a current felony charge, but who did not"\
           "get rearrested for a felony crime have a higher predicted probability than arrestees with a current misdemeanor charge, "\
            "but who did get rearrested for a felony crime?")
    print("This means the model heavily weights felony charges leading to higher rearrest probabilities (even if they didn't reoffend).")
    print("This shows the model overemphasizes the charge type for felonies, which is a bias.\n")


def main():
    pred_universe, arrest_events, charge_counts, charge_counts_by_offense, felony_charge, merged_charges = part1.extract_transform()

    barplot_felony_rearrest(merged_charges)
    barplot_nonfelony_rearrest(merged_charges)
    barplot_felony_rearrest_hue(merged_charges)
    


if __name__ == "__main__":
    main()