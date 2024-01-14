import pandas as pd
import polars as pl
import streamlit as st 

st.title('Water quality detection project')

box_sections = st.selectbox('Choose what you want to see of the project', ['Description', 'Exploratory Data Analysis', 'Plots', 'Model'])

if box_sections == 'Description':
       st.write('prova')

if box_sections == 'Exploratory Data Analysis':
       import io
       st.header('Exploratory Data Analysis')
       if st.checkbox('Before Cleaning'):
              df = pl.read_csv('csv\Water_Quality_Prediction.csv')
              df = pd.DataFrame(df)
              df.columns = ['Index', 'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color',
                            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                            'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source',
                            'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day',
                            'Target']
              st.write(df.head(5))
              '''
              General informations for our water quality dataset:
              '''
              st.write('Rows and columns:', df.shape)
              st.write('Total null values:', df.isnull().sum().sum())
              buffer = io.StringIO()
              df.info(buf = buffer)
              s = buffer.getvalue()
              st.text(s)
       if st.checkbox('After Cleaning'):
              df = pl.read_csv('csv\Water_Quality_Prediction_Clean.csv')
              df = pd.DataFrame(df)
              df.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color',
                            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                            'Chlorine', 'Manganese', 'Total_Diss_Solids', 'Potability']
              st.write(df.head(5))
              '''
              General informations for our water quality dataset:
              '''
              st.write('Rows and columns:', df.shape)
              st.write('Total null values:', df.isnull().sum().sum())
              buffer = io.StringIO()
              df.info(buf = buffer)
              s = buffer.getvalue()
              st.text(s)
  
if box_sections == 'Plots':
       st.header('Main plots')
       '''
       In the dataset we encountered a lot of null values
       '''
       st.image('images\inull_val.png')
       '''
       After cleaning our dataset, we ended up with 0 null values accross all variables.
       '''
       '''
       It's important that our cleaning of the dataset didn't impact too much the distributions of our variables. 
       If that was the case, we might encounter some problems in the model part of the project.
       '''
       box_distr_features = st.selectbox('Distribution of the variable:', ['None', 'pH', 'Iron', 'Nitrate', 
                                                                           'Chloride', 'Lead', 'Zinc', 'Color', 
                                                                           'Turbidity', 'Fluoride', 'Copper', 'Odor', 
                                                                           'Sulfate', 'Conductivity', 'Chlorine', 
                                                                           'Manganese', 'Total_Diss_Solids'], 
                                                                            label_visibility = 'collapsed')
       if box_distr_features == 'None':
              st.image('images\distr\iNone.png')
       if box_distr_features == 'pH':
              st.image('images\distr\pH.png')
       if box_distr_features == 'Iron':
              st.image('images\distr\Iron.png')
       if box_distr_features == 'Nitrate':
              st.image('images\distr\iNitrate.png')
       if box_distr_features == 'Chloride':
              st.image('images\distr\Chloride.png')
       if box_distr_features == 'Zinc':
              st.image('images\distr\Zinc.png')
       if box_distr_features == 'Color':
              st.image('images\distr\Color.png')
       if box_distr_features == 'Turbidity':
              st.image('images\distr\Turbidity.png')
       if box_distr_features == 'Fluoride':
              st.image('images\distr\Fluoride.png')
       if box_distr_features == 'Copper':
              st.image('images\distr\Copper.png')
       if box_distr_features == 'Odor':
              st.image('images\distr\Odor.png')
       if box_distr_features == 'Sulfate':
              st.image('images\distr\Sulfate.png')
       if box_distr_features == 'Conductivity':
              st.image('images\distr\Conductivity.png')
       if box_distr_features == 'Chlorine':
              st.image('images\distr\Chlorine.png')
       if box_distr_features == 'Manganese':
              st.image('images\distr\Manganese.png')
       if box_distr_features == 'Total_Diss_Solids':
              st.image('images\distr\Total_Diss_Solids.png')
       if box_distr_features == 'Lead':
              st.image('images\distr\Lead.png')
       '''
       The correlation between variables can be seen from the following heatmap:
       '''
       box_heatmap_features = st.select_slider
       st.image('images\heatmap.png')
       box_boxplot_features = st.selectbox('Feature', ['None', 'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 
                                                       'Color','Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 
                                                       'Conductivity', 'Chlorine', 'Manganese', 'Total_Diss_Solids'], label_visibility = 'collapsed')
       if box_boxplot_features == 'None':
              st.image('images\iboxplot\iNone.png')
       if box_boxplot_features == 'pH':
              st.image('images\iboxplot\pH.png')
       if box_boxplot_features == 'Iron':
              st.image('images\iboxplot\Iron.png')
       if box_boxplot_features == 'Nitrate':
              st.image('images\iboxplot\iNitrate.png')
       if box_boxplot_features == 'Chloride':
              st.image('images\iboxplot\Chloride.png')
       if box_boxplot_features == 'Lead':
              st.image('images\iboxplot\Lead.png')
       if box_boxplot_features == 'Zinc':
              st.image('images\iboxplot\Zinc.png')
       if box_boxplot_features == 'Color':
              st.image('images\iboxplot\Color.png')
       if box_boxplot_features == 'Turbidity':
              st.image('images\iboxplot\Turbidity.png')
       if box_boxplot_features == 'Fluoride':
              st.image('images\iboxplot\Fluoride.png')
       if box_boxplot_features == 'Copper':
              st.image('images\iboxplot\Copper.png')
       if box_boxplot_features == 'Odor':
              st.image('images\iboxplot\Odor.png')
       if box_boxplot_features == 'Sulfate':
              st.image('images\iboxplot\Sulfate.png')
       if box_boxplot_features == 'Conductivity':
              st.image('images\iboxplot\Conductivity.png')
       if box_boxplot_features == 'Chlorine':
              st.image('images\iboxplot\Chlorine.png')
       if box_boxplot_features == 'Manganese':
              st.image('images\iboxplot\Manganese.png')
       if box_boxplot_features == 'Total_Diss_Solids':
              st.image('images\iboxplot\Total_Diss_Solids.png')
       