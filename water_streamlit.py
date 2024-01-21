# import packages
import pandas as pd
import polars as pl
import streamlit as st 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
# set the title for all the pages
st.title('Water Quality Detection Project')
# create a drop-down menu for the 4 pages of the project
box_sections = st.selectbox('What part of the project would you like to see?', ['Description', 'Exploratory Data Analysis', 'Plots', 'Prediction Model'])

if box_sections == 'Description':
# general informations about project and dataset
       '''
       My main goal for this project is to build a prediction model, as accurate as possible, that classifies if a given water sample is potable based on the 15 variables provided.\n
       From the drop-down menu on the top of this page, you can select what part of the project you're interested in.\n
       **Exploratory Data Analysis:** gives insights on the general structure of the dataset, and it's divided between "Before Cleaning" and "After Cleaning".\n
       **Plots:** in here are stored the main plots that can help us have a better understanding of the relations between variables and their relative distributions.\n
       **Prediction Model:** stored in this page is an interesting visualization of the variables after dimensionality reduction thanks to PCA, and then a slider where you can select the test size for the Random Forest Classifier Model and its relative f1 score. Additionally, a Ten-Fold cross validation gives insight on the most convenient number of decision trees to use.\n
       
       **Details of the dataset variables:**\n
       1) *pH -* Measures how acidic or basic our sample is. More specifically, it indicates the concentration of hydrogen ions in the water. Unintuitively, an high pH means a higher concentration of hydrogen, a low pH means a lower concentratio. Its scale is between 0 and 14, and a neutral pH of 7 means that is neither acidic or basic.\n
       2) *Iron -* One of the most plentiful resources on earth, can enter underground water sources through rainwater. Iron is not hazardous to healt, but it can be a problem for the house pipe system if it's concentration is too high.\n
       3) *Nitrate -* A compound that naturally occurs in water, consuming too much of it can be harmful (especially for babies). Mainly, consuming too much nitrate can affect how blood carries hoxygen, and also have negative effects on heart rate, nausea, headaches and abdominal cramps.\n
       4) *Chloride -* Considered to be an essential nutrient for human healt, the main source of it should be food and not water. It's most often found as a component of salt, and as a consequence of this we can found higher concentration of Chloride in water acquifers near costal areas.\n
       5) *Zinc -* A naturally occurring metal element, its recommended limit for drinking water is considered to be 5 mg/l. Essential element for human metabolism, if consumed at extremely high concentrations (675 mg/l) can cause several problems to the digestive system.\n
       6) *Color -* The color of the water sample. It can range from colorless to yellow.\n
       7) *Turbidity -* Measures the relative clarity of a liquid. Materials that cause the turbidity include clay, silt, algae, plankton and other microscopic organism.\n
       8) *Fluoride -* Has beneficial effects on tooth decay and overall dental development, its often added to tap water by municipalities.\n
       9) *Copper -* Metal that occurs naturally, its presence in water is mainly cause of pipe corrosion. In most people drinking copper does not cause illness, but it's possible that some may experience headaches, vomiting, liver damage, stomach cramps with high level of consumption.\n
       10) *Odor -* It measures how strong is the water sample odor.\n
       11) *Sulfate -* Can be found in almost all natural water, sulfate stands out as a prominent dissolved element in rain. Elevated levels of sulfate in our drinking water, when interacting with calcium and magnesium can induce a laxative effect.\n
       12) *Conductivity -* Pure water demonstrates poor conductivity of electric current; instead, it serves as an effective insulator. The augmentation of ion concentration amplifies the electrical conductivity of water. In general, the level of dissolved solids in water dictates its electrical conductivity.\n
       13) *Chlorine -* An important element that helps prevent diseases caused by bacteria, viruses and other microorganisms since it kill them. Municipalities often add Chlorine to tap water.\n
       14) *Manganese -* Mineral that naturally occurs in groundwater. At concentrations higher than 0.05 mg/L manganese may cause a noticeable color, odor, or taste in water. Potential health effects from manganese are not a concern until concentrations are approximately six times higher.\n
       15) *Total Dissolved Solids -* Represents the total concentration of dissolved substances in water samples. A high concentration of dissolved solids is usually not a health hazard, and its level in water is often correlated with how strong the taste is.\n
       16) *Potability -* Indicates if the water of a specific sample is potable or not. 0 means potable, 1 means non-potable.\n\n
       
       '''
       st.write('The Water Quality Prediction dataset can be found on:')
       st.link_button('Kaggle', 'https://www.kaggle.com/datasets/vanthanadevi08/water-quality-prediction')
       st.write('or in the GitHub repository for this project.')
       st.write('The full code for this project can be found on ')
       st.link_button('Github', 'https://github.com/dariogemo/water_project')
# EDA part of the project
if box_sections == 'Exploratory Data Analysis':
       import io
       st.header('Exploratory Data Analysis')
# create two buttons before and after cleaning to display informations of the dataset
       if st.checkbox('Before Cleaning'):
# load the before cleaning dataset using polars
              df = pl.read_csv('csv\Water_Quality_Prediction.csv')
              df = pd.DataFrame(df)
              df.columns = ['Index', 'pH', 'Iron', 'Nitrate', 'Chloride', 'Lead', 'Zinc', 'Color',
                            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                            'Chlorine', 'Manganese', 'Total Dissolved Solids', 'Source',
                            'Water Temperature', 'Air Temperature', 'Month', 'Day', 'Time of Day',
                            'Target']
# display head and tail of the dataset
              st.write(df.head(5))
              st.write(df.tail(5))
              '''
              **General informations for our water quality dataset:**
              '''
# display the shape and the null values 
              col1, col2 = st.columns(2)
              col1.write(f'Rows and columns: {df.shape}')
              col2.write(f'Total null values: {df.isnull().sum().sum()}')
# use a string buffer to correctly display pd.info()
              buffer = io.StringIO()
              df.info(buf = buffer)
              s = buffer.getvalue()
              st.text(s)
       if st.checkbox('After Cleaning'):
# load the after cleaning dataset
              df = pl.read_csv('csv\Water_Quality_Prediction_Clean.csv')
              df = pd.DataFrame(df)
              df.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 'Color',
                            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                            'Chlorine', 'Manganese', 'Total_Diss_Solids', 'Potability']
# display head and tail of the dataset
              st.write(df.head(5))
              st.write(df.tail(5))
              '''
              **General informations for our water quality dataset, cleaned:**
              '''
# display the shape and the null values 
              col1, col2 = st.columns(2)
              col1.write(f'Rows and columns: {df.shape}')
              col2.write(f'Total null values: {df.isnull().sum().sum()}')
# use a string buffer to correctly display pd.info()
              buffer = io.StringIO()
              df.info(buf = buffer)
              s = buffer.getvalue()
              st.text(s)
# Plots part of the project
if box_sections == 'Plots':
       st.header('Main plots')
       '''
       In the dataset we encountered a lot of null values
       '''
# load the barplot of the null values
       st.image('images\inull_val.png')
       '''
       After cleaning our dataset, we ended up with 0 null values accross all variables.
       '''
       '''
       The correlation between variables can be seen from the following heatmap:
       '''
# load the heatmap of the dataset
       st.image('images\heatmap.png')
       '''
       No big correlations between variables, so we'll keep them all.\n 
       It seems that Color and Turbidity have one of the highest correlation with Potability: this was somewhat expected, because even in our daily life we are suspicious of water that isn't transparent or seems turbid. We can better check the relation between Color and Potability.
       '''
# create a button to display the frequency table between color and potability
       if st.checkbox('Show frequency table of Color and Potability'):
              st.image('images\Color-Potability.png')
       '''
       Another "relevant" correlation might be between Manganese and Turbidity. Potable water usually has very low levels of Manganese and Turbidity levels between 0 and 1.
       '''
# create a button to display the scatterplot between manganese and turbidity
       if st.checkbox('Show scatterplot between Manganese and Turbidity'):
              st.image('images\Manganese-Turbidity.png')
       '''
       Also, it might be important to check the correlation between Color and Turbidity since intuitively they should have some type of connection.
       '''
# create a button to display the boxplot between color and turbidity
       if st.checkbox('Show boxplot of Color and Turbidity'):
              st.image('images\Color-Turbidity.png')
       '''
       It's crucial that our cleaning of the dataset didn't impact too much our variables.\n 
       If that was the case, we might encounter some lower performances in the model part of the project.
       '''
# create a drop-down menu to select the type of graph
       box_type_graph = st.selectbox('Choose what type of graph you would like to see:', ['None', 'Histogram', 'Boxplot', 'Violinplot'], key = 'graphs')
       if box_type_graph == 'None':
              pass
       if box_type_graph == 'Histogram': 
# create a drop-down menu to select the variable
              box_distr_features = st.selectbox('Distribution of the variable:', ['None', 'pH', 'Iron', 'Nitrate', 
                                                                             'Chloride', 'Zinc', 'Color', 
                                                                             'Turbidity', 'Fluoride', 'Copper', 'Odor', 
                                                                             'Sulfate', 'Conductivity', 'Chlorine', 
                                                                             'Manganese', 'Total_Diss_Solids'], 
                                                                             label_visibility = 'collapsed', key = 'distr')
# load the image of the type of graph and variable selected
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
# create a drop-down menu to select the variable
              box_boxplot_features = st.selectbox('Feature', ['None', 'pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 
                                                        'Color','Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 
                                                        'Conductivity', 'Chlorine', 'Manganese', 'Total_Diss_Solids'], label_visibility = 'collapsed', key = 'boxplot')
# load the image of the type of graph and variable selected
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
# create a drop-down menu to select the variable
              box_violinplot_features = st.selectbox('Feature violinplot', ['None', 'pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 
                                                        'Color','Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 
                                                        'Conductivity', 'Chlorine', 'Manganese', 'Total_Diss_Solids'], label_visibility = 'collapsed', key = 'violinplot')
# load the image of the type of graph and variable selected
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
# Prediction Model part of the project
if box_sections == 'Prediction Model':
       '''
       We can first look for clusters that are not evident with our high-dimensional data. We'll use Principal Component Analsisys to reduce the dimensionality of our dataset.\n
       As we can see from the Scree Plot, 2 Principal Components should be enough since most of the information is passed by in the dimensionality reduction.
       '''
       # load the scree plot png
       st.image('images\scree.png')
       '''
       We can now visualize our data thanks to the PCA. Two clusters are visible. 
       '''
       # load the PCA scatterplot png
       st.image('images\pca.png')
       '''
       ---
       The model used for predicting the potability of a sample water is RandomForestClassifier with a default number of decision trees equal to 100.
       '''
       # create a slider to select the size of the dataset and save the resulting integer in a variable
       t_size = st.slider('Choose test size', 10, 90, step = 10)
       # load the cleaned and already resampled dataset
       df = pl.read_csv('csv\Water_Quality_Prediction_res.csv')
       df = pd.DataFrame(df)
       df.columns = ['pH', 'Iron', 'Nitrate', 'Chloride', 'Zinc', 'Color',
                            'Turbidity', 'Fluoride', 'Copper', 'Odor', 'Sulfate', 'Conductivity',
                            'Chlorine', 'Manganese', 'Total_Diss_Solids', 'Potability']
       # create X dataset so that it contains all the features
       X = df.drop('Potability', axis = 1)
       # create y series so that it contains only the potability column
       y = df['Potability']
       # create X and y datasets used for training and testing the model
       # test_size is the number selected by the user from the above slider
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size/100, random_state = 1)
       # create columns to centered information
       col1, col2, col3, col4 = st.columns([1, 2, 1, 2.3])
       # display how many rows are in train and test dataset
       col2.write(f'Train size: {X_train.shape[0]}')
       col4.write(f'Test size: {X_test.shape[0]}')
       # load the pre-trained model
       with open('model\potability_classifier.pkl', 'rb') as file:
              model = pickle.load(file)
       # use the model to predict the potability of our test dataset
       y_pred_svm = model.predict(X_test)
       # evaluate the prediction by using f1 score and classification report       
       f1 = f1_score(y_test, y_pred_svm, average = None, labels = [0, 1])
       st.write(f'**F1 score** for the model applied to the dataset with a test size of {t_size}%:')
       # create columns to center the scores
       col1, col2, col3, col4 = st.columns([1, 2, 1, 2.3])
       col2.write(f'**{f1[0] * 100:.2f}**%')
       col4.write(f'**{f1[1] * 100:.2f}**%')
       '''
       **Classification report**:
       '''
       # create columns to center the classification report
       col1, col2, col3 = st.columns([1, 5, 1])
       col2.text(classification_report(y_test, y_pred_svm))
       '''
       **Ten-Fold Cross Validation using KFold**\n
       We can now cross-validate our model to check for overfitting and then see how this behaviour changes as we increase the number of decision trees.
       '''
       n_est = [5, 25, 50, 75, 100, 150, 200]
       scores = [90.12, 91.69, 91.82, 91.76, 91.78, 91.79, 91.78]
       scores = [str(x)+'%' for x in scores]
       n_est = pd.Series([round(int(x), 0) for x in n_est], name = 'N. Decision Trees')
       scores = pd.Series(scores, name = 'Average Accuracy Scores')
       cv_df = pd.concat([n_est, scores], axis = 1).T
       col1, col2, col3 = st.columns([0.5, 5, 0.5])
       col2.write(cv_df)
       st.image('images\k_fold.png', caption = 'How the CV mean accuracy score changes as the number of decision trees increases')
       '''
       Except the case with 5 decision trees, the differences between the scores aren't really relevant, so we might choose for computational reasons to fix the number of decision trees to 50.
       '''