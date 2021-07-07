import numpy as np
import pandas as pd
import seaborn as sns
import math
from features import desired_indicators, field_names

def merge(file1, file2):
    data = pd.read_csv(file1)                      # file1: "athlete_events.csv"
    world = pd.read_csv(file2)                     # file2: "noc_regions.csv"
    merged = pd.merge(data, world, on='NOC', how='left')
    merged.to_csv('merged_data.csv', index=False)

def clean_data(data, wdi, world, indicators = desired_indicators, field_name = field_names):
    #data = pd.read_csv("merged_data.csv")                   
    #wdi = pd.read_csv("WDIData.csv")
    #world = pd.read_csv("noc_regions.csv")
    data = data[data.Season == 'Summer']
    data = data.drop(columns=['Height', 'Weight', 'Age', 'Sex', 'Games'])
    data.Medal.fillna(False)
    
    my_cols = ['Nation', 'Year'] + \
        field_name + \
            ['Athletes', 'Athletes_Normalized',\
             'Medals_Last_Games',\
                 'Total_Medals_year',\
                     'Medals']
                
    # Replace Medal types with True/False
    data.Medal.replace('Gold',True, inplace=True)
    data.Medal.replace('Silver',True, inplace=True)
    data.Medal.replace('Bronze',True, inplace=True)
    
    # Summing medals for each country and creating new Dataframe
    newdf = pd.DataFrame(columns=my_cols)
    for region in data.region.unique():
        naming_discrpancy = True
        if not type(region)==str:
            if math.isnan(region):
                continue
        for year in data.Year.unique():
            subset_world = data.loc[(data.Year == year)]
            total_medals_year = np.sum(subset_world.Medal)
            total_athletes_year = len(subset_world['Name'].unique())
            
            subset = data.loc[(data.Year == year) & (data.region == region)]
            total_medals = np.sum(subset.Medal)
            total_athletes = len(subset['Name'].unique())
            
            total_medals_norm = total_medals / total_medals_year
            total_athletes_norm = total_athletes / total_athletes_year
            
            indicators_list = []
            
            for i in range(len(indicators)):
                ind = indicators[i]
                sub = wdi[wdi['Country Name'] == region]
                sub = sub[sub['Indicator Code'] == ind]
                if len(sub) == 1:
                    my_val = sub[str(year)].values
                    if len(my_val)>1:
                        raise Exception('More than one data point')
                    if 'Normalized' in field_names[i]:
                        subworld = wdi[wdi['Indicator Code'] == ind]
                        world_val = subworld[str(year)].values
                        indicators_list.append(my_val[0] / world_val[0])
                    else:
                        indicators_list.append(my_val[0])
                elif len(sub) == 0:
                    indicators_list.append(np.nan)
                else:
                    raise Exception('Indicators list greater than 1')
                    
                
                # Figuring out naming discrepancies
                if not np.sum(np.isnan(indicators_list)) == len(indicators_list):
                    naming_discrepancy = False
                
            newdf = newdf.append(pd.DataFrame([[region, year] + indicators_list + 
                                               [total_athletes, total_athletes_norm, np.nan, total_medals_year, total_medals]], columns = my_cols))
        if naming_discrepancy:
            print(region)
        
    unique_years = np.sort(data.Year.unique())
    
    for i in range(1,len(unique_years)):
        this_year = unique_years[i]
        last_year = unique_years[i-1]
        for nation in newdf[newdf.Year==this_year].Nation.unique():
            df_last = newdf.loc[(newdf.Year==last_year) & (newdf.Nation==nation)]
            medals_last_year = df_last.Medals.values
            if not len(medals_last_year)==1:
                raise Exception('Problem with number last year medals')
            newdf.loc[(newdf.Year==this_year) & (newdf.Nation==nation), 'Medals_Last_Games'] = medals_last_year[0]
    newdf.to_csv("newdata.csv", index=False)        
            
    return newdf
                
def remove_duplicate_medals(mgfile):
    data = pd.read_csv(mgfile)
    tempdf = pd.DataFrame(colums=data.columns)
    for year in data.Year.unique():            
        for event in data[data.Year==year].Event.unique():
            for medal in ['Gold','Silver','Bronze']:             
                subset = data.loc[(data.Year==year)&(data.Event==event)&(data.Medal==medal)]
                if (not subset.empty) and len(subset)>1 and len(subset.Team.unique())==1:
                    tempdf = tempdf.append(subset.iloc[1:])
                    
    return_df = data.drop(tempdf.index)
    return_df.to_csv('data_no_duplicate.csv', index=False)
    #return return_df



def main(year_begin, year_end, indicators = desired_indicators, field_name = field_names):
    merged = pd.read_csv('data_no_duplicate.csv')
    regions = pd.read_csv('noc_regions.csv')
    wdi = pd.read_csv('WDIData.csv')
    
    # This variable is included to calculate medals in previous game
    year_begin = year_begin-4
    merged=merged[merged.Year >= year_begin]
    merged=merged[merged.Year <= year_end]
    
    
    world = wdi[wdi['Country Name'] == 'World']
    wdi = wdi[wdi['Country Name'].isin(regions.region.unique())]
    
    wdi = wdi[wdi['Indicator Code'].isin(desired_indicators)]
    world = world[world['Indicator Code'].isin(desired_indicators)]
    
    years_list = [str(i) for i in merged.Year.unique() if (i>=year_begin and i<=year_end)]
    temp = wdi[years_list].notna()
    wdi = wdi[temp.eq(1).all(axis=1)]
    
    cleandf = clean_data(merged, wdi, world, desired_indicators, field_names)
    
    # Getting rid of exampless with NaN values
    heatmap = sns.heatmap(cleandf.isnull())
    heatmap.set(yticks=[])
    heatmap.get_figure().savefig('heatmap.ps', bbox_inches='tight')
    cleandf = cleandf.dropna(axis=0)
    sns.heatmap(cleandf.isnull())
    
    cleandf.to_csv('clean_data.csv', index=False)
    return cleandf
    
def train_test_split(data, validation_year, normalized=False):
    training_data = data[data.Year<validation_year]
    valid_data = data[data.Year == validation_year]
    
    x_train = training_data.drop(columns=['Nation','Medals'])
    y_train = training_data['Medals']
    
    x_valid = valid_data.drop(columns=['Nation','Medals'])
    y_valid = valid_data['Medals']
    
    return x_train,y_train,x_valid,y_valid

def to_numpy(x_train, y_train, x_valid, y_valid):
    xt = x_train.to_numpy()
    yt = y_train.to_numpy()
    xv = x_valid.to_numpy()
    yv = y_valid.to_numpy()
    
    return xt,yt,xv,yv

def to_clf_data(yt, yv):
    yt_bool = np.zeros(len(yt))
    yt_bool[yt!=0] = 1
    yv_bool = np.zeros(len(yv))
    yv_bool[yv!=0] = 1
    
    return yt_bool, yv_bool