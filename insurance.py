import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.title("Predict Insurance Premiums")

file = st.sidebar.file_uploader("Select a file to upload", type = ["csv"])

if file is not None:
    df = pd.read_csv(file)

    if st.sidebar.checkbox("Show data"):
        st.markdown("## Raw Data")
        st.dataframe(df.head())
    
        st.write("This data has {} rows and {} columns.".format(df.shape[0],df.shape[1]))

    if st.sidebar.checkbox("Show statistical summary"):
        #st.markdown("## EDA")
        st.markdown("### Statistical summary")
        st.write(df.describe())
    
    if st.sidebar.checkbox("Show plots"):
        st.markdown("### Distribution of Premium Charges")
        fig, ax = plt.subplots()
        ax = sns.histplot(df["charges"])
        st.pyplot(fig)
    
        st.markdown("### Box Plots")
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)
    
        with row1_col1:
            fig, ax = plt.subplots()
            ax.set_title("Smokers vs. Charges")
            ax = sns.boxplot(df["smoker"], df["charges"])
            st.pyplot(fig)

        with row1_col2:
            fig, ax = plt.subplots()
            ax.set_title("Sex vs. Charges")
            ax = sns.boxplot(df["sex"], df["charges"], hue = df["smoker"])
            st.pyplot(fig)

        with row2_col1:
            fig, ax = plt.subplots()
            ax.set_title("Region vs. Charges")
            ax = sns.boxplot(df["region"], df["charges"], hue = df["smoker"])
            st.pyplot(fig)

        with row2_col2:
            fig, ax = plt.subplots()
            ax.set_title("Children vs. Charges")
            ax = sns.boxplot(df["children"], df["charges"], hue = df["smoker"])
            st.pyplot(fig)

    
        st.markdown("### Scatter Plots")
        col1, col2 = st.columns(2)
    
        with col1:
            fig, ax = plt.subplots()
            ax.set_title("BMI vs. Charges")
            ax = sns.scatterplot(df["bmi"], df["charges"], hue = df["smoker"])
            st.pyplot(fig)
      
        with col2:
            fig, ax = plt.subplots()
            ax.set_title("Age vs. Charges")
            ax = sns.scatterplot(df["age"], df["charges"], hue = df["smoker"])
            st.pyplot(fig)
        
    #st.markdown("## Data Pre-Processing")
    columns = ["sex", "smoker"]
    df_encoded = pd.get_dummies(df, columns = columns, drop_first = True, prefix = "enc")
    df_encoded.rename(columns = {"enc_male": "gender", "enc_yes": "smoker"}, inplace = True)
    df_encoded.drop(["children", "region"], axis = "columns", inplace = True)
    
    X = df_encoded.drop("charges", axis = "columns")
    y = df_encoded["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
    if st.sidebar.checkbox("Show pre-processed data"):
        st.markdown("### Categorical Variable Encoding")
        st.table(df_encoded.head())
        
        st.markdown("### Train and Test Data")
        tcol1, tcol2 = st.columns(2)
    
        with tcol1:
            st.dataframe(X_train.sample(5))
        with tcol2:
            st.dataframe(y_train.sample(5))
    
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    df_coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(model.coef_))], axis = 1)
    df_coefficients.columns = ["variable", "coefficient"]
    y_intercept = round(model.intercept_,2)
    
    if st.sidebar.checkbox("Show coefficients and model evaluation"):
        st.markdown("## Model Building and Evaluation")
        mcol1, mcol2 = st.columns(2)
    
        with mcol1:
            st.markdown("#### Coefficients")
            st.dataframe(df_coefficients)
            st.metric("The Y-Intercept is:", y_intercept)
        with mcol2:
            st.markdown("#### Predicted vs. Actual")
            fig, ax = plt.subplots()
            ax = sns.regplot(predictions, y_test)
            ax.set(xlabel='Predicted Charges',
                   ylabel='Actual Charges')
            st.pyplot(fig)
            st.metric("Mean Squared Error:", round(mean_squared_error(predictions, y_test),2))
        
    age_coeff, bmi_coeff, gender_coeff, smoker_coeff = df_coefficients["coefficient"]

    with st.form("Data to Calculate Premiums"):
        
        st.write("Use this form to find out how much you will be charged in insurance premium.")
        age = st.number_input("What is your age?", min_value = 18, max_value = 100)
        bmi = st.number_input("What is your BMI?", min_value = 0.0)
        gender = st.radio("What is your gender?", ["Male", "Female"])
        smoker = st.radio("Do you smoke?", ["Yes", "No"])        

        if gender == "Male":
            gender = 1
        else:
            gender = 0

        if smoker == "Yes":
            smoker = 1
        else:
            smoker = 0
        
        premium_charge = age_coeff * age +\
                     bmi_coeff * bmi +\
                     gender_coeff * gender +\
                     smoker_coeff * smoker +\
                     y_intercept
        if st.form_submit_button("Click to find out your premium"):
            st.metric("Your premium in dollars is:", round(premium_charge,2))
