import numpy as np
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, \
    classification_report
import joblib

# Import the dataset
df = pd.read_csv('data.csv')

# we will drop the id & convert categorical variables to numeric using Encoding
df.drop("id", inplace=True, axis=1)
#test.drop("id", inplace=True, axis=1)

df['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
df['Vehicle_Damage'].replace({'Yes': 1, 'No': 0}, inplace=True)
df['Vehicle_Age'].replace({'< 1 Year': 1, '1-2 Year': 2, '> 2 Years': 3}, inplace=True)



# Set page config with a sidebar
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# set the app's title
st.title("Insurance Prediction with Machine Learning")

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Description", "Data Visualisation", "Data Modelling"]
    )

if selected == "Home":
    st.write("Our client is an Insurance company that has provided Health Insurance "
             "to its customers now they need your help in building a model "
             "to predict whether the policyholders (customers) from the past year "
             "will also be interested in Vehicle Insurance provided by the company.")

if selected == "Data Description":
    # header
    st.header("Data Description")
    # add an image
    image = Image.open('data_description.png')
    st.image(image, caption='Data Description')
    st.write('Source: [Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)')
    st.write('We have a dataset contains information about health insurance clients such as '
             'demographics (gender, age, region code type), '
             'Vehicles (Vehicle Age, Damage), '
             'Policy (Premium, sourcing channel), etc. '
             'in order to predict, whether the customer would be interested in buying a vehicle insurance,')

    st.write("Below is a snippet of how the data looks like:")
    st.write(df.head())

    st.write(f'We have {df.shape[0]} rows in the data and no null values.')

if selected == "Data Visualisation":
    # Data Visualisation
    st.header("Data Visualisation")

    st.subheader('Interactive distribution between features and response')
    def interactive_histogram_plot(dataframe):
        #x_axis_val = st.selectbox('Select X-Axis', options=dataframe.columns)
        feature = st.selectbox('Select feature', options=dataframe.columns[:-1])
        #plot = px.scatter(dataframe, x=x_axis_val, y=y_axis_val)
        plot = px.histogram(dataframe,
                            x='Response',
                            marginal='box',
                            color=feature,
                            title=f'Response and {feature} correlation')
        plot.update_layout(bargap=0.1)
        st.plotly_chart(plot)

    interactive_histogram_plot(df)


    st.subheader('Correlation Heatmap')
    sns.set(font_scale=0.5)
    fig, ax = plt.subplots()
    plt.figure(figsize=(20, 17))
    matrix = np.triu(df.corr())
    sns.heatmap(df.corr(),
                ax=ax,
                annot=True,
                linewidth=.8,
                mask=matrix,
                cmap="rocket")
    st.write(fig)

if selected == "Data Modelling":
    # Data Modelling
    st.header("Data Modelling")
    features = list(df.columns)[0:-1]
    target = "Response"

    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    st.write(f'We split the data into train and test dataset. We have {X_train.shape[0]} rows in the traindata'
             f' and {X_test.shape[0]}')

    # Logistic Regression
    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train, y_train)

    # Decision Tree
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)

    # Random Forest
    forest = RandomForestClassifier(n_jobs=-1, random_state=42)
    forest.fit(X_train, y_train)

    # Compare and display chosen model
    chosen_model = st.radio('Please select to model to see its performance:',
                         ("LogisticRegression",
                          "DecisionTreeClassifier",
                          "RandomForestClassifier"))

    st.write('You selected:', chosen_model)
    if chosen_model == "LogisticRegression":
        model = log_reg
    elif chosen_model == "DecisionTreeClassifier":
        model = tree
    elif chosen_model == "RandomForestClassifier":
        model = forest
    else:
        st.write("You didn't pick any model.")

    def predict_and_show_report(model):

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[::,1]

        # display classification report
        st.write("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write(report_df)

        Q1, Q2 = st.columns(2)
        # display confusion matrix
        with Q1:
            fig1, ax1 = plt.subplots()
            cf = confusion_matrix(y_test, y_pred, normalize='true')
            sns.heatmap(cf, annot=True)
            plt.xlabel('Prediction')
            plt.ylabel('Target')
            plt.title('{} Confusion Matrix'.format(model));
            st.write(fig1)
        # display roc curve
        with Q2:
            fig2 = plt.figure(figsize=(8, 8))
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
            auc = metrics.roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label="auc="+str(auc))
            plt.legend(loc=4)
            plt.title('{} AUC ROC Curve'.format(model))
            st.pyplot(fig2)

    predict_and_show_report(model)

# Playground

