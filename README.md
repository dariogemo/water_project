# Water Potability Prediction

My main goal for this project is to build a model that classifies if a given water sample is potable based on the 15 variables provided.

**Details of the files in the repository:**

- *csv:* all the csv files that are needed for the streamlit presentation, as well as the Water Quality Prediction starting csv file.

- *images:* all the images that are needed for the streamlit presentation. For computational reasons, I found it better to store the plots inside here instead of re-creating them in the streamlit file.

- *models:* the Logistic Regression and SVM models resulting from this project, stored in .pkl files.

- *clean_df.py:* 

- *funzioni.py:* all the functions defined for the data cleaning of the dataset.

- *water_project.ipynb:* the notebook used for this project, where is stored almost all the code.

. *water_streamlit.py:* the python file used for the streamlit presentation.

**Details of the dataset features:**

1) *pH -* Measures how acidic or basic our sample is. More specifically, it indicates the concentration of hydrogen ions in the water. Unintuitively, an high pH means a higher concentration of hydrogen, a low pH means a lower concentratio. Its scale is between 0 and 14, and a neutral pH of 7 means that is neither acidic or basic.

2) *Iron -* One of the most plentiful resources on earth, can enter underground water sources through rainwater. Iron is not hazardous to healt, but it can be a problem for the house pipe system if it's concentration is too high.

3) *Nitrate -* A compound that naturally occurs in water, consuming too much of it can be harmful (especially for babies). Mainly, consuming too much nitrate can affect how blood carries hoxygen, and also have negative effects on heart rate, nausea, headaches and abdominal cramps.

4) *Chloride -* Considered to be an essential nutrient for human healt, the main source of it should be food and not water. It's most often found as a component of salt, and as a consequence of this we can found higher concentration of Chloride in water acquifers near costal areas.

5) *Zinc -* A naturally occurring metal element, its recommended limit for drinking water is considered to be 5 mg/l. Essential element for human metabolism, if consumed at extremely high concentrations (675 mg/l) can cause several problems to the digestive system.

6) *Color -* The color of the water sample. It can range from colorless to yellow.

7) *Turbidity -* Measures the relative clarity of a liquid. Materials that cause the turbidity include clay, silt, algae, plankton and other microscopic organism.

8) *Fluoride -* Has beneficial effects on tooth decay and overall dental development, its often added to tap water by municipalities.

9) *Copper -* Metal that occurs naturally, its presence in water is mainly cause of pipe corrosion. In most people drinking copper does not cause illness, but it's possible that some may experience headaches, vomiting, liver damage, stomach cramps with high level of consumption.

10) *Odor -* It measures how strong is the water sample odor.

11) *Sulfate -* Can be found in almost all natural water, sulfate stands out as a prominent dissolved element in rain. Elevated levels of sulfate in our drinking water, when interacting with calcium and magnesium can induce a laxative effect.

12) *Conductivity -* Pure water demonstrates poor conductivity of electric current; instead, it serves as an effective insulator. The augmentation of ion concentration amplifies the electrical conductivity of water. In general, the level of dissolved solids in water dictates its electrical conductivity.

13) *Chlorine -* An important element that helps prevent diseases caused by bacteria, viruses and other microorganisms since it kill them. Municipalities often add Chlorine to tap water.

14) *Manganese -* Mineral that naturally occurs in groundwater. At concentrations higher than 0.05 mg/L manganese may cause a noticeable color, odor, or taste in water. Potential health effects from manganese are not a concern until concentrations are approximately six times higher.

15) *Total Dissolved Solids -* Represents the total concentration of dissolved substances in water samples. A high concentration of dissolved solids is usually not a health hazard, and its level in water is often correlated with how strong the taste is.

16) *Potability -* Indicates if the water of a specific sample is potable or not. 0 means potable, 1 means non-potable.

The Water Quality Prediction dataset can be found in [Kaggle](https://www.kaggle.com/datasets/vanthanadevi08/water-quality-prediction) or in the [Github](https://github.com/dariogemo/water_project) repository for this project, where you can also find the full code.