#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[73]:



df = pd.read_csv("medical_examination.csv")
df.head()


# In[74]:


#Insertion of ''overweight'' column and BMI calculation, 0 = good <25 BMI, 1= bad >25 BMI
df.insert(7 , 'overweight', True)
df['overweight'] = 0
for index, row in df.iterrows():
    if (row['weight']/(row['height']/100)**2) > 25:
        df.at[index, "overweight"] = 1

# Data normalization 0 = good, 1 = bad
# Normalization of 'cholesterol' column
for index, row in df.iterrows():
    if row['cholesterol'] <= 1:
        df.at[index, 'cholesterol'] = 0
    else:
        df.at[index, 'cholesterol'] = 1
        
# Normalization of 'gluc' column       
for index, row in df.iterrows():
    if row['gluc'] <= 1:
        df.at[index, 'gluc'] = 0
    else:
        df.at[index, 'gluc'] = 1
   


def draw_cat_plot():
    
    #categorical values filtering
    categorical_cols = ['sex', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    df_long_data = pd.melt(df, id_vars=['id', 'cardio'], value_vars=categorical_cols, var_name='variable', value_name='value')

    df_cat = df_long_data.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    
    #plotting based on categorical values
    g = sns.catplot(x='variable', y='size', hue='value', col='cardio', data=df_cat, kind='bar')
    
    fig = g.fig

    fig.savefig('catplot.png')
    return fig




# In[76]:



def draw_heat_map(df):
    # data cleaning
    df_cleaned = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # correlation matrix
    corr_matrix = df_cleaned.corr()

    # plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(corr_matrix)  # Keep only the lower triangle of the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt='.2f', vmin=0.0, vmax=0.3, square=True)
    plt.title('Correlation Matrix')
    plt.savefig('heatmap.png')
    plt.show()
    return fig







