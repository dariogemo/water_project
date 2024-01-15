import pandas as pd
import polars as pl
import funzioni as fn 

def clean_df():
    df = pl.read_csv('csv\Water Quality Prediction.csv')
    df = pd.DataFrame(df)
    df.drop([22, 21, 20, 19, 18, 17, 5, 0], axis = 1, inplace = True)
    df.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 'Color',
        'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
        'Chlorine', 'Manganese', 'Total_Diss_Solids', 'Potability']
    df['Color'].replace({'Colorless' : 0,
                        'Near Colorless' : 1,
                        'Faint Yellow' : 2,
                        'Light Yellow' : 3,
                        'Yellow' : 4}, inplace = True)
    for target in ['pH', 'Chloride', 'Sulfate', 'Conductivity', 'Chlorine']:
        fn.na_to_norm_distr(df, target)
    for target in ['Iron', 'Nitrate', 'Color', 'Turbidity', 'Manganese']:
        fn.na_to_median_target(df, target)
    for target in ['Odor', 'Total_Diss_Solids']:
        fn.na_to_unif_distr(df, target)
    for target in ['Zinc', 'Fluoride', 'Copper']:
        fn.na_to_median(df, target)
    for target in ['Iron', 'Nitrate', 'Chloride', 'Zinc', 'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity', 'Chlorine', 'Manganese', 'Total_Diss_Solids']:
        df = fn.outl_del(df, target, 0.99)
    
    return df