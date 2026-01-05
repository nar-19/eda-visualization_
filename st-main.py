import os
import sys
# os.system(f'{sys.executable} -m pip install -r requirements.txt') #take care for path of file

import pandas as pd              # data manipulation
import numpy as np               # math operations on arrays / random number gen
import matplotlib.pyplot as plt  # visualization package
import seaborn as sns            # visualization package
from math import ceil, floor     # math rounding operationS
import streamlit as st           # web app

# 1) Displaying page title & subtitle
# ------------------------------------
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

# --- Initialize session state for DataFrame and current dataset name ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_dataset_name' not in st.session_state:
    st.session_state.current_dataset_name = "No dataset loaded"

# --- Define Data File Paths ---
filepath1 = 'dataset/WA_Fn-UseC_-Sales-Win-Loss.csv'
filepath2 = 'dataset/earthquake_data_tsunami.csv'
filepath3 = 'dataset/Global finance data.csv'

# --- Handle Initial Default Data Loading ---
# This block runs only once on the first app load if no df is in session state
if st.session_state.df is None:
    with st.status("Loading default dataset...", expanded=True) as status_initial_load:
        try:
            st.session_state.df = pd.read_csv(filepath1)
            st.session_state.current_dataset_name = "Sales win/loss dataset (default)"
            status_initial_load.write(f"Default dataset '{st.session_state.current_dataset_name}' loaded.")
            # We mark it complete here, and the analysis status will pick up next.
            status_initial_load.update(label="Default dataset loaded!", state="complete", expanded=False)
        except FileNotFoundError:
            status_initial_load.error(f"Default dataset not found at '{filepath1}'. Please ensure the 'dataset' folder exists and contains the file.")
            st.stop()
        except Exception as e:
            status_initial_load.error(f"Error loading default dataset: {e}")
            st.stop()

# --- Data Selection UI ---
st.subheader("Select a predefined dataset or upload your own:")
button1, button2, button3 = st.columns(3)

# Flag to indicate if a new dataset was selected by user action
new_dataset_selected = False 

if button1.button("Sales win/loss dataset"):
    st.session_state.df = pd.read_csv(filepath1)
    st.session_state.current_dataset_name = "Sales win/loss dataset"
    new_dataset_selected = True
if button2.button("Tsunami dataset"):
    st.session_state.df = pd.read_csv(filepath2)
    st.session_state.current_dataset_name = "Tsunami dataset"
    new_dataset_selected = True
if button3.button("Finance dataset"):
    st.session_state.df = pd.read_csv(filepath3)
    st.session_state.current_dataset_name = "Finance dataset"
    new_dataset_selected = True

uploaded_file = st.file_uploader("Or, upload your own .csv file")
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.current_dataset_name = uploaded_file.name
    new_dataset_selected = True
st.markdown(" ")

# --- Main Analysis and Plotting Logic (within a status block) ---
# This block runs on every rerun if df is available (initial load or user selection)
if st.session_state.df is not None:
    # Use a local variable for convenience, referring to the session state df
    df = st.session_state.df

    # Use a single status for all processing after data load
    with st.status(f"Processing and visualizing '{st.session_state.current_dataset_name}'...", expanded=True) as status_analysis:
        status_analysis.write("Starting data analysis...")

        # ------------------------
        # INITIALIZE PLOT SETTING
        #-------------------------
        # Defining function to set figure size
        def figure(a,b):
            sns.set(rc={'figure.figsize':(a,b)})

        plt_cols = 3                                                           # Customized no. of columns in subplot
        
        # Flags to track if plots were generated
        cat_vars_present = False
        num_vars_present = False
        
        # CATEGORICAL VARIABLES
        if df.columns[df.dtypes == 'object'].tolist():
            cat_vars_present = True
            df_string = df.loc[:,df.dtypes == 'object']

            plt_rows = ceil(len(df_string.columns)/plt_cols)                    
            figure(12, 10)
            fig1, axes = plt.subplots(plt_rows,plt_cols)
            fig1.suptitle("Countplots of Categorical Variables \n (x-axis: Variable) \n (y-axis: Count of samples)", 
                        fontsize="x-large")
            axes = axes.ravel()

            for i in range(0, len(df_string.columns)):
                sns.histplot(df_string.iloc[:,i], ax=axes[i])
                axes[i].set_title('Countplot of ' + df_string.columns[i], size=15)
                axes[i].tick_params(axis='x', labelrotation=90, pad=0)
                axes[i].set_xlabel('')
            fig1.tight_layout(rect=[0, 0, 1, 0.88])
            fig1.subplots_adjust(top=0.85)
            plt.savefig('fig1',dpi=1000)
            status_analysis.write("Categorical variable plots generated.")
        else:
            # if os.path.exists('fig1.png'): os.remove('fig1.png')
            status_analysis.write("No categorical variables found for plotting.")


        # NUMERICAL VARIABLES
        if df.columns[df.dtypes != 'object'].tolist():
            num_vars_present = True
            df_numeric0 = df.loc[:, df.dtypes!='object']

            id_cols = ['Index', 'index']
            df_numeric = df_numeric0.loc[:, ~df_numeric0.columns.isin(id_cols)]

            plt_rows = ceil(len(df_numeric.columns)/plt_cols)             
            figure(12, 10) 
            fig2, axes = plt.subplots(plt_rows,plt_cols)
            fig2.suptitle("Distribution Plots of Numerical Variables \n (x-axis: Variable) \n (y-axis: Distribution proportion)", 
                        fontsize="x-large")
            axes = axes.ravel()

            for i in range(0, len(df_numeric.columns)):
                sns.histplot(df_numeric.iloc[:,i], kde=True, ax=axes[i])
                axes[i].set_title(df_numeric.columns[i], size=15)
                axes[i].tick_params(axis='x', labelrotation=90, pad=0)
                axes[i].set_xlabel('')
            fig2.tight_layout(rect=[0, 0, 1, 0.88])
            fig2.subplots_adjust(top=0.85)
            plt.savefig('fig2',dpi=1000)
            status_analysis.write("Numerical variable plots generated.")
        else:
            # if os.path.exists('fig2.png'): os.remove('fig2.png')
            status_analysis.write("No numerical variables found for plotting.")

        # Mark analysis as complete and collapse the status widget
        status_analysis.update(label=f"Analysis of '{st.session_state.current_dataset_name}' complete!", state="complete", expanded=False)

    # --- DASHBOARD DISPLAY (Outside the status container, so it's always visible) ---
    st.markdown("---")
    st.subheader(f"Analysis Results for: {st.session_state.current_dataset_name}")
    
    if cat_vars_present:
        st.subheader("Categorical Variables")
        st.image('fig1.png')
        st.markdown("Head of the dataframe that contains only categorical variables:")
        st.dataframe(df_string.head())
        st.markdown(" ")
    
    if num_vars_present:
        st.subheader(" ")
        st.subheader("Numerical Variables")
        st.image('fig2.png')    
        st.markdown("Head of the dataframe that contains only numerical variables:")
        st.dataframe(df_numeric0.head())
    
    # ) Delete displayed images from system
    if os.path.exists('fig1.png'):
        os.remove('fig1.png')
    if os.path.exists('fig2.png'):
        os.remove('fig2.png')
else:
    # This message should ideally not be reached if default load is successful
    st.info("Please select a dataset using the buttons above or upload a CSV file to begin.")

# Running Locally in VSCode
# PS C:\Users\a**r**\Desktop\project_\eda-visualization_> & C:/Users/a**r**/AppData/Local/Programs/Python/Python312/python.exe -m streamlit run st-main.py
