import os
import sys
# os.system(f'{sys.executable} -m pip install -r requirements.txt') #take care for path of file

import pandas as pd              # data manipulation
import numpy as np               # math operations on arrays / random number gen
import matplotlib.pyplot as plt  # visualization package
import seaborn as sns            # visualization package
from math import ceil, floor     # math rounding operationS
import streamlit as st           # web app
import os                        # handle png image

# ) Displaying page title & subtitle
st.title("Data Visualization Dashboard :bar_chart: ")
st.subheader("for EDA (Exploration Data Analysis)")
st.subheader(" ")

# st.markdown(sns.__version__)
# st.markdown(st.__version__)
# st.subheader(" ")

st.markdown(":arrow_forward: This dashboard visualizes: \n"
                "\n  👉 Count Plot on categorical variables \n"
                "\n  👉 Distribution plot on numerical variables")

st.markdown(":arrow_forward: The plots assist you to understand your initial data condition e.g. variables classes, shape of distribution, etc.")

st.markdown(":arrow_forward: Uploaded data conditions: \n"
                "\n  👉 **.csv** file format \n"
                "\n  👉 Structured data with Header & rows \n"
                "\n  👉 Preferably no Index/ID/Unique Keys column \n"
                "\n  👉 Max file size: **200MB** \n"
                "\n  👉 No encoding issue ")
st.markdown(":arrow_forward: For wide screen view, select ≡ > Settings > Wide mode ")
st.markdown("Kindly wait while file is running 🏃")
st.subheader(" ")

# ) Read Data

filepath1 = './dataset/WA_Fn-UseC_-Sales-Win-Loss.csv'
filepath2 = './dataset/earthquake_data_tsunami.csv'
filepath3 = './dataset/Global finance data.csv'
filepath4 = './dataset/Sales Transaction v.4a.csv'

df = pd.read_csv(filepath1)

button1, button2, button3, button4 = st.columns(4)
if button1.button("Sales win/loss dataset"):
    # button1.markdown("You clicked the plain button.")
    df = pd.read_csv(filepath1)
if button2.button("Tsunami dataset"):
    df = pd.read_csv(filepath2)
if button3.button("Finance dataset"):
    df = pd.read_csv(filepath3)
if button4.button("Transaction dataset"):
    df = pd.read_csv(filepath4)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
      # Can be used wherever a "file-like" object is accepted:
        df_upload = pd.read_csv(uploaded_file)
        st.write(df_upload)
        df = df_upload
st.markdown(" ")


# ------------------------
# INITIALIZE PLOT SETTING
#-------------------------
# Defining function to set figure size
def figure(a,b):
    sns.set(rc={'figure.figsize':(a,b)})

# figure(15,10)
plt_cols = 3                                                           # Customized no. of columns in subplot

# st.markdown('Cat variables:' + str(df.columns[df.dtypes == 'object'].tolist()))
# st.markdown('Num variables:' + str(df.columns[df.dtypes != 'object'].tolist()))
    
# CATEGORICAL VARIABLES
# ---------------------
# if condition: Plot categorical variable plots if any cat variable is available
if df.columns[df.dtypes == 'object'].tolist() != []:
    df_string = df.loc[:,df.dtypes == 'object']

    plt_rows = ceil(len(df_string.columns)/plt_cols)                       # Set the no. of rows in subplot by dividing: 
                                                                        #  roundup(no. of variables / no. of columns)
    figure(15, 10)
    fig1, axes = plt.subplots(plt_rows,plt_cols)
    fig1.suptitle("Countplots of Categorical Variables \n (x-axis: Variable) \n (y-axis: Count of samples)", 
                fontsize="x-large")
    axes = axes.ravel()


    for i in range(0, len(df_string.columns)):
        # sns.countplot(df_string.iloc[:,i], ax=axes[i])                     # Plot countplot for each categorical variable
        sns.histplot(df_string.iloc[:,i], ax=axes[i])
        # sns.histplot(df_numeric.iloc[:,i], kde=True)
        axes[i].set_title('Countplot of ' + df_string.columns[i], size=15) # Set title of every subplot
        axes[i].tick_params(axis='x', labelrotation=90, pad=0)             # Rotate x-axis of every subplot
        axes[i].set_xlabel('')                                             # Turn off subplots' x-axis titles for tidiness
    fig1.tight_layout(rect=[0, 0, 1, 0.88])                                # Adjust tight_layout to accommodate suptitle
    fig1.subplots_adjust(top=0.85)

    # save image, display it, and delete after usage.
    plt.savefig('fig1',dpi=1000)
    # st.image('fig1.png')


# ---------------------
# NUMERICAL VARIABLES
# ---------------------
# if condition: Plot numerical variable plots if any numeric variable is available
if df.columns[df.dtypes != 'object'].tolist() != []:
    df_numeric0 = df.loc[:, df.dtypes!='object']

    # Check presence of Index / index 
    #  and drop those columns
    id_cols = ['Index', 'index']
    df_numeric = df_numeric0.loc[:, ~df_numeric0.columns.isin(id_cols)]

    plt_rows = ceil(len(df_numeric.columns)/plt_cols)             # Set the no. of rows in subplot by dividing: 
                                                                #  roundup(no. of variables / no. of columns)
    figure(15, 10) 
    fig2, axes = plt.subplots(plt_rows,plt_cols)
    fig2.suptitle("Distribution Plots of Numerical Variables \n (x-axis: Variable) \n (y-axis: Distribution proportion)", 
                fontsize="x-large")
    axes = axes.ravel()


    for i in range(0, len(df_numeric.columns)):
        # sns.distplot(df_numeric.iloc[:,i], ax=axes[i])            # Plot countplot for each categorical variable
        sns.histplot(df_numeric.iloc[:,i], kde=True, ax=axes[i])
        axes[i].set_title(df_numeric.columns[i], size=15)         # Set title of every subplot
        axes[i].tick_params(axis='x', labelrotation=90, pad=0)    # Rotate x-axis of every subplot
        axes[i].set_xlabel('')                                    # Turn off subplots' x-axis titles for tidiness
    fig2.tight_layout(rect=[0, 0, 1, 0.88])                       # Adjust tight_layout to accommodate suptitle
    fig2.subplots_adjust(top=0.85)

    # save image, display it, and delete after usage.
    plt.savefig('fig2',dpi=1000)


# ---------------------
# DASHBOARD DISPLAY
#----------------------
# Displaying results for Categorical Variables
if df.columns[df.dtypes == 'object'].tolist() != []:
    st.subheader("Categorical Variables")
    result1 = st.container()

    with result1:
        result1.image('fig1.png')
        result1.markdown("Head of the dataframe that contains only categorical variables:")
        result1.dataframe(df_string.head())
        result1.markdown(" ")
    
# Displaying results for Numerical Variables
if df.columns[df.dtypes != 'object'].tolist() != []:
    st.subheader(" ")
    st.subheader("Numerical Variables")
    result2 = st.container()

    with result2:
        result2.image('fig2.png')    
        result2.markdown("Head of the dataframe that contains only numerical variables:")
        result2.dataframe(df_numeric0.head())
    
    
# ) Delete displayed images from system
if os.path.exists('fig1.png'):
    os.remove('fig1.png')
if os.path.exists('fig2.png'):
    os.remove('fig2.png')
 
    
# Running Locally in VSCode
# PS C:\Users\atirazmi\Desktop\project_\eda-visualization_> & C:/Users/atirazmi/AppData/Local/Programs/Python/Python312/python.exe -m streamlit run st-main.py