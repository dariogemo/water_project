import pandas as pd
import polars as pl
import streamlit as st 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

st.title('Water Quality Detection Project')

box_sections = st.selectbox('What part of the project would you like to see?', ['Description', 'Exploratory Data Analysis', 'Plots', 'Model'])

if box_sections == 'Description':
       st.write('The Water Quality Prediction dataset can be found at:')
       st.link_button('Kaggle', 'https://www.kaggle.com/datasets/vanthanadevi08/water-quality-prediction')
       st.write('or in the GitHub repository for this project.')
       st.write('The full code for this project can be found at ')
       st.link_button('Github', 'https://github.com/dariogemo/water_project')

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
              st.write(df.tail(5))
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
              df.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 'Color',
                            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                            'Chlorine', 'Manganese', 'Total_Diss_Solids', 'Potability']
              st.write(df.head(5))
              st.write(df.tail(5))
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
       The correlation between variables can be seen from the following heatmap:
       '''
       st.image('images\heatmap.png')
       '''
       It's important that our cleaning of the dataset didn't impact too much our variables. 
       If that was the case, we might encounter some lower performances in the model part of the project.
       '''
       box_type_graph = st.selectbox('Choose what type of graph you would like to see:', ['None', 'Histogram', 'Boxplot', 'Violinplot'], key = 'graphs')
       if box_type_graph == 'None':
              pass
       if box_type_graph == 'Histogram': 
              box_distr_features = st.selectbox('Distribution of the variable:', ['None', 'pH', 'Iron', 'Nitrate', 
                                                                             'Chloride', 'Zinc', 'Color', 
                                                                             'Turbidity', 'Fluoride', 'Copper', 'Odor', 
                                                                             'Sulfate', 'Conductivity', 'Chlorine', 
                                                                             'Manganese', 'Total_Diss_Solids'], 
                                                                             label_visibility = 'collapsed', key = 'distr')
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
       if box_type_graph == 'Boxplot':
              box_boxplot_features = st.selectbox('Feature', ['None', 'pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 
                                                        'Color','Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 
                                                        'Conductivity', 'Chlorine', 'Manganese', 'Total_Diss_Solids'], label_visibility = 'collapsed', key = 'boxplot')
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
       if box_type_graph == 'Violinplot':
              box_violinplot_features = st.selectbox('Feature violinplot', ['None', 'pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 
                                                        'Color','Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 
                                                        'Conductivity', 'Chlorine', 'Manganese', 'Total_Diss_Solids'], label_visibility = 'collapsed', key = 'violinplot')
              if box_violinplot_features == 'None':
                     st.image('images\iviolinplot\iNone.png')
              if box_violinplot_features == 'pH':
                     st.image('images\iviolinplot\pH.png')
              if box_violinplot_features == 'Iron':
                     st.image('images\iviolinplot\Iron.png')
              if box_violinplot_features == 'Nitrate':
                     st.image('images\iviolinplot\iNitrate.png')
              if box_violinplot_features == 'Chloride':
                     st.image('images\iviolinplot\Chloride.png')
              if box_violinplot_features == 'Zinc':
                     st.image('images\iviolinplot\Zinc.png')
              if box_violinplot_features == 'Color':
                     st.image('images\iviolinplot\Color.png')
              if box_violinplot_features == 'Turbidity':
                     st.image('images\iviolinplot\Turbidity.png')
              if box_violinplot_features == 'Fluoride':
                     st.image('images\iviolinplot\Fluoride.png')
              if box_violinplot_features == 'Copper':
                     st.image('images\iviolinplot\Copper.png')
              if box_violinplot_features == 'Odor':
                     st.image('images\iviolinplot\Odor.png')
              if box_violinplot_features == 'Sulfate':
                     st.image('images\iviolinplot\Sulfate.png')
              if box_violinplot_features == 'Conductivity':
                     st.image('images\iviolinplot\Conductivity.png')
              if box_violinplot_features == 'Chlorine':
                     st.image('images\iviolinplot\Chlorine.png')
              if box_violinplot_features == 'Manganese':
                     st.image('images\iviolinplot\Manganese.png')
              if box_violinplot_features == 'Total_Diss_Solids':
                     st.image('images\iviolinplot\Total_Diss_Solids.png')
if box_sections == 'Model':
       with open('models\potability_classifier_svm.pkl', 'rb') as file:
              svm_model = pickle.load(file)
       t_size = st.slider('Choose test size', 10, 90, step = 10)
       df = pl.read_csv('csv\Water_Quality_Prediction_Balanced.csv')
       df = pd.DataFrame(df)
       df.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 'Color',
                            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                            'Chlorine', 'Manganese', 'Total_Diss_Solids', 'Potability']
       X = df.drop('Potability', axis = 1)
       y = df['Potability']
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size, random_state = 1)
       y_pred_svm = svm_model.predict(X_test)
       f1_score_svm = f1_score(y_test, y_pred_svm, average = None, labels = [0, 1])
       st.write(f'F1 score for the SVM model applied to the dataset: {f1_score_svm[0] * 100:.2f}%, {f1_score_svm[1] * 100:.2f}%')
       with open('models\potability_classifier_log.pkl', 'rb') as file:
              log_model = pickle.load(file)
       y_pred_log = log_model.predict(X_test)
       f1_score_log = f1_score(y_test, y_pred_log, average = None, labels = [0, 1])
       st.write(f'F1 score for the Log. Regression model applied to the dataset: {f1_score_log[0] * 100:.2f}%, {f1_score_log[1] * 100:.2f}%')
