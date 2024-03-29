# Heart-Failure-Prediction
## Context 
Cardiovascular diseases (CVDs) stand as the primary cause of global mortality, claiming approximately 17.9 million lives annually, representing 31% of all global deaths. Among these fatalities, four out of five are attributed to heart attacks and strokes, with one-third transpiring prematurely in individuals below 70 years old. Heart failure, a prevalent consequence of CVDs, is a significant concern. This dataset comprises 11 features potentially useful for predicting the likelihood of heart disease. Timely detection and effective management are imperative for individuals with cardiovascular disease or those at high risk due to factors such as hypertension, diabetes, hyperlipidemia, or existing conditions. Machine learning models offer promising avenues for aiding in early identification and intervention in such cases.
## Attribute Information:
1. Age: Age of the patient [years]
2. Sex: Sex of the patient [M: Male, F: Female]
3. Chest Pain Type: Type of chest pain experienced [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. Resting Blood Pressure: Resting blood pressure [mm Hg]
5. Cholesterol: Serum cholesterol level [mm/dl]
6. Fasting Blood Sugar: Fasting blood sugar level [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. Resting Electrocardiogram Results: Results of resting electrocardiogram [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. Maximum Heart Rate Achieved: Maximum heart rate achieved during exercise [Numeric value between 60 and 202]
9. Exercise-Induced Angina: Presence of exercise-induced angina [Y: Yes, N: No]
10. Oldpeak: ST depression induced by exercise relative to rest [Numeric value measured in depression]
11. ST Slope: Slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12. Heart Disease: Output class indicating presence of heart disease [1: heart disease, 0: Normal]
## Dataset
This dataset was created by combining different datasets that were already available independently 
but had not been combined before. In this dataset, five heart datasets are combined over 11 
common features, which makes it the largest heart disease dataset available so far for research 
purposes. The five datasets used for its curation are:
Cleveland: 303 observations, Hungarian: 294 observations, Switzerland: 123 observations, Long 
Beach VA: 200 observations, Stalog (Heart) Data Set: 270 observations
Total: 1190 observations, Duplicated: 272 observations, Final dataset Size: 918 observations [1]
## Visualizations:
![image](https://github.com/shereenIbdah/Heart-Failure-Prediction/assets/108181177/709466c5-c5ca-463f-b776-54600d928fc0)
## Methodology:

Data Preprocessing: We conducted initial data exploration to identify missing values, outliers, and potential inconsistencies. Categorical variables were encoded appropriately, and numerical features were scaled to ensure uniformity.
Exploratory Data Analysis (EDA): EDA involved visualizing the distribution of features, exploring correlations, and identifying patterns that could offer insights into the relationships between variables and the target.
Feature Selection: We employed techniques such as correlation analysis, feature importance ranking, and domain knowledge to select the most relevant features for model training.
Model Development: Several machine learning algorithms, including logistic regression, decision trees, random forests, and gradient boosting, were trained on the dataset. Hyperparameter tuning and cross-validation were performed to optimize model performance.
Model Evaluation: The models were evaluated using metrics such as accuracy, precision, recall, F1-score, and area under the receiver operating characteristic curve (AUC-ROC) to assess their predictive capabilities.
## Conclusion:
Our study demonstrates the feasibility of using machine learning techniques to predict the risk of heart disease based on various patient attributes. By leveraging these predictive models, healthcare professionals can identify high-risk individuals early and implement targeted interventions to mitigate the impact of cardiovascular diseases.

## Future Directions:
Future research could focus on incorporating additional medical data, such as genetic markers and lifestyle factors, to enhance the predictive accuracy of the models. Furthermore, deploying the developed models in clinical settings and assessing their real-world performance would be valuable for validating their utility in practice.

