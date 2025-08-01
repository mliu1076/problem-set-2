import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import src.part1_etl as part1

def barplot_fta(pred_universe):
    """
    Creates a bar plot for the 'fta' column in the `pred_universe` dataFrame, 

    Parameters:
    - pred_universe dataframe

    Returns:
    - A bar plot showing the counts of individuals who failed to appear (FTA) vs. those who did not.
    """
    plt.clf() 

    fta_counts = pred_universe['fta'].value_counts()


    sns.barplot(
        x=fta_counts.index.astype(str),
        y=fta_counts.values
    )

    # sets labels
    plt.xlabel('Failure to Appear (FTA)')
    plt.ylabel('Count')
    plt.title('Failure to Appear (FTA) vs. Non-FTA')

    plt.savefig('data/part3_plots/fta_barplot.png', bbox_inches='tight')

def barplot_fta_by_sex(pred_universe):
    '''
    Creates a bar plot for the 'fta' column, with the hue based on 'sex'.

    Parameters:
    - pred_universe dataframe

    Returns:
    - A barplot for the 'fta' column
    '''
    plt.clf()
    sns.barplot(data=pred_universe, 
                x='fta', 
                hue='sex', 
                estimator=lambda x: len(x), 
                errorbar=None)
    plt.title('FTA by Sex')
    plt.savefig('./data/part3_plots/barplot_fta_by_sex.png', bbox_inches='tight')
    plt.clf()

def histogram_age_at_arrest(pred_universe):
    '''
    Creates a histogram for the 'age_at_arrest' column.

    Parameters:
    - pred_universe: Dataframe containing individual prediction data

    Returns:
    -  A histogram for the 'age_at_arrest' column.
    '''
    sns.histplot(pred_universe['age_at_arrest'], kde=False, bins=30)

    # sets labels
    plt.title('Histogram of Age at Arrest')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('./data/part3_plots/histogram_age_at_arrest.png', bbox_inches='tight')
    plt.clf()

def histogram_age_groups(pred_universe):
    '''
    Creates a histogram for 'age_at_arrest' with bins representing specific age groups.

    Parameters:
    - pred_universe: Dataframe containing individual prediction data

    Returns:
    - A histogram with age groups.
    '''
    bins = [18, 21, 30, 40, 100]
    labels = ['18-21', '21-30', '30-40', '40-100']
    pred_universe['age_group'] = pd.cut(pred_universe['age_at_arrest'], bins=bins, labels=labels, right=False)
    
    sns.histplot(pred_universe['age_group'], kde=False)

    # sets labels
    plt.title('Histogram of Age Groups at Arrest')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.savefig('./data/part3_plots/histogram_age_groups.png', bbox_inches='tight')
    plt.clf()

def main():
    '''
    Main function to generate all plots for Part 3.

    Returns:
    - None
    '''

    pred_universe, arrest_events, charge_counts, charge_counts_by_offense = part1.extract_transform()

    barplot_fta(pred_universe)

    barplot_fta_by_sex(pred_universe)

    histogram_age_at_arrest(pred_universe)

    histogram_age_groups(pred_universe)

if __name__ == "__main__":
    main()