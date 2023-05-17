# Project goals:
We would be analyzing, exploring single family houses that had a transaction in 2017 from the Zillow dataset. 
This data would not be used on future properties or for real life prediction.
# Project description:
 For this project we would be exploring the different factors that affect the tax value
 We would be attempting to improve the tax value predictions for these properties.
 Some of these factors are: bathroom count, bedroom count, squared footage(area), county, etc. Focusing on these factors would help us come up with ideas on how to predict tax value.

# Project planning:
 1. Planning: During this process we asked ourselves important questions about the project. Data planning will be shown in this readme.
 2. Acquisition: We would be acquiring the data from Codeup Data Server in MYSQL. Raw data would be downloaded and "zillow.csv" has been      created which would be use to pull the data during this project.
 3. Preparation: The zillow data would be clean and prepared for exploration. Columns were listed along with data types. Columns with incorrect data types were transformed, columns were created to encode a int/float instead of a string, and columns that are not to be used were dropped. Nulls were handled accordingly and quality assurance was practiced to ensure the validity of each attribute.
 4. Exploration: During the data exploration we would visualize and answer our questions and hypotheses. We would be using statistical tests and plots, to help proving our hypotheses and understand the different tax value by county.
 5. Evaluation & Modeling:Zilloq data would be scaled. feature engineering would be utilize to assist on choosing the best features for our data to make the best prediction of the tax values.
 6. Delivery- Data Delivery- We would be using a jupyter notebook, were we would be showing our visualizations, questions and findings. We would also show the score metrics of our best model. This project would be a written and a verbal presentation.

# Initial hypotheses and/or questions you have of the data, ideas:
1. Having a pool affects the tax value of the property?
2. The amount of bedrooms affects the tax value of the property
3. Does county affects the tax value prices?
4. Does having air conditioning affects tax value of the property?

# Data dictionary:
<img width="755" alt="Data Dictionary" src="https://github.com/Chellyann-moreno/regression-project/assets/126615525/e7c8f62c-11a3-49cf-931e-d80f2ebc40f2">


# Instructions on how someone else can reproduce your project and findings:
For an user to succesfully reproduce this project, they must have a connection to the CodeUp Server in MySQL. User must have a "env.py" with the username, password, and database name, to establish a connection. Also, the wrangle.py, explore.py and evaluate.py must be download in the same repository/folder as the final_report to run it successfully. Once all files are download, user may run the final_report notebook.
# Key findings, recommendations, and takeaways from your project:
During exploration we proved that the following features have an effect on tax value of the properties: pool, bedrooms, air_conditioning and county.

During feature engineering we have found: SelecKBest and RFE selected bathrooms, bedrooms, and area as the top best features that affect tax value.

The best model with the best RMSE and R2 is: Polymnomial Regression with a degree of three, which performed ~44% better than baseline.

Based on our observations, we recomment using the machine model to better predict the tax value amount of properties. Keeping in mind utilizing the features used above.

Given more time I would like to explore other features to best predict tax value.

Look at different hypermaters and features to better the machine model.
