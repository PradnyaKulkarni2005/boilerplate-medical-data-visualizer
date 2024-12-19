import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load data
df = pd.read_csv('medical_examination.csv')

# Step 2: Calculate BMI and overweight
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)
df = df.drop(columns=['BMI'])

# Step 3: Normalize cholesterol and glucose
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Step 4: Draw categorical plot
def draw_cat_plot():
    # Melt the data for categorical plotting
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Create the catplot
    plot = sns.catplot(
        data=df_cat, kind='bar',
        x='variable', y='total', hue='value', col='cardio'
    )
    plot.set_axis_labels("variable", "total")  # Explicitly set axis labels
    plot.set_titles("{col_name} cardio")      # Set titles for each subplot

    # Save the figure
    fig = plot.fig
    fig.savefig('catplot.png')
    return fig

# Step 5: Draw heatmap
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the heatmap
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.1f',
        square=True, cbar_kws={'shrink': .5}, ax=ax
    )

    # Save the figure
    fig.savefig('heatmap.png')
    return fig
