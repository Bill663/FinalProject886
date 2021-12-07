from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt

# set the style for seaborn
sns.set_style('darkgrid')

# title of the app
st.title("Data visualization App")
st.write('A car features data exploration app design by Zhiyang Jiang')

st.markdown ("[CarData Set](https://www.kaggle.com/goyalshalini93/car-data/version/1)")

file = st.file_uploader("Please download the dataset to you personal computer and upload it to the following window.", type = ["csv"])  #question 3

if file is not None:
    df = pd.read_csv(file)   
    df = df.applymap(lambda x: np.nan if x=="" else x)
    df

    def can_be_numeric(c):    
        try:
            pd.to_numeric(df[c])
            return True
        except:
            return False

    good_cols = [c for c in df.columns if can_be_numeric(c)]   
    
    df[good_cols] = df[good_cols].apply(pd.to_numeric, axis = 0)
    
    st.write("The following data explorier allow you to compare the realtion between every elements from the dataset")
    
    x_axis = st.selectbox("Choose an x_value",good_cols)
    y_axis = st.selectbox("Choose an y_value",good_cols)

    values = st.slider("Select a range of values",0, len(df.index)-1,(60,len(df.index)-1))

    st.write(f"plotting ({x_axis},{y_axis}) for rows {values}")
    val = list(values)
    
    row_val = df.loc[val[0]:val[1]]

    my_chart = alt.Chart(row_val).mark_point().encode(x = x_axis, y = y_axis)
    st.altair_chart(my_chart)
    
    st.write("if you selected a group of data you desire, download the data is an option for further analysis.")
    
    st.download_button(
        label = "Download selected values as CSV",
        data = row_val.to_csv().encode('utf-8'),
        file_name = 'Selected_values.csv',
        mime='text/csv'
        )
    st.write("For example: the following chart show the realtion between a vehicle's horsepower and citympg with information about the cylindernumber of its engine.")
    col_a = "horsepower"
    col_b = "citympg"
    df = df[[col_a,col_b,"cylindernumber"]].copy()
    df_old = df[[col_a,col_b,"cylindernumber"]].copy()
    mychart2 = alt.Chart(df).mark_circle().encode(
                x = alt.X(col_a,scale=alt.Scale(zero=False)),
                y = alt.Y(col_b,scale=alt.Scale(zero=False)),
                color='cylindernumber'
    )
    
    st.altair_chart(mychart2)
    df = df[df.notna().all(axis = 1)].copy()
    df.shape
    df_old = df_old[df_old.notna().all(axis = 1)].copy()
    scaler = StandardScaler()
    df2 = df[[col_a,col_b]]
    scaler.fit(df2)
    df2 = scaler.transform(df2)
    df[[col_a,col_b]] = df2
    
    st.write("the following graph is fitted with the use of python tool Seaborn")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.lineplot(x='horsepower', y='citympg', data=df)
    st.pyplot()
    
    st.write("Through the graphs display above, we can safely conclude that more performence oriented vehicle tend to have bad citympg. ")
    
    st.write("the following graph is generated using K-Nearest Neighbors classifier, with 10 neighbors. ")
    
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(df[[col_a,col_b]],df["cylindernumber"])
    df["pred1"] = clf.predict(df[[col_a,col_b]])
    mychart3 = alt.Chart(df).mark_circle().encode(
                x = alt.X("horsepower",scale=alt.Scale(zero=False)),
                y = alt.Y("citympg",scale=alt.Scale(zero=False)),
                color='pred1'
)
    st.altair_chart(mychart3)
    st.write("the system make prediction utilizing data of cylindernumber.")
    
    
    
st.write("reference list:")
st.write("Cardata set from https://www.kaggle.com/goyalshalini93/car-data/version/1")
st.write("I reference my own homework assignment for Homework 4 and Homework 8 in the making of this project.")
st.markdown("[Original code](https://github.com/Bill663/FinalProject663)")
st.write("Pandas was used to clean and modified dataset.")
st.write("Altair was used to visualized data.")
st.write("Scikit-learn was used for data prediction.")
st.write("Mutiple Streamlit widgets like slider, file uploader, and file downloader were featured.")
st.write("Seaborn was used to display graph.")
st.markdown("[Reference to official Seaborn documentation] (https://seaborn.pydata.org/) ")
    
    
   