import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from joblib import dump, load
from sklearn.model_selection import train_test_split 
# from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    # set up the page title and widemode
    st.set_page_config(page_title="Dashboard", layout="wide")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Streamlit Dashboard for Heart Failure Prediction")

    # get the data
    data = st.sidebar.file_uploader("Upload your data file", type=["csv", "txt"])
    st.sidebar.title("Select Model Parameters to train the model and check performance")

    # get the model parameters
    kernel = st.sidebar.multiselect("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    c = st.sidebar.slider("C", 0.01, 10.0, step=0.01)
    gamma = st.sidebar.slider("Gamma", 0.01, 10.0, step=0.01)
    degree = st.sidebar.slider("Degree (if poly kernel used)", 1, 5, 1)

    # preview data yes/no
    preview = st.sidebar.checkbox("Preview", value=False)
    if data is not None:
        data = pd.read_csv(data)

        #Preprocessing
        to_convert=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina","ST_Slope"]
        for feature in to_convert:
            replace_with = {value: num for num, value in enumerate(data[feature].unique())}
            data[feature]=data[feature].replace(replace_with)
        labels=data["HeartDisease"]
        X_train, X_test, y_train, y_test=train_test_split(data.drop(columns="HeartDisease"),labels,test_size=0.2,random_state=1234)
        kernels=kernel
    else:
        st.error("Please upload a data file (CSV/TXT)")

    if preview:
        st.subheader("Preview of the data")
        st.dataframe(data.head())


    if st.sidebar.button("Train Model(s)"):
        col_num = st.columns(len(kernel))
        best_p = 0
        best_r = 0
        best_f1 = 0
        for i, kernel in enumerate(kernels):
            with col_num[i]:
                st.write("Confusion matrix for :", kernel)
                svm = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree)
                svm.fit(X_train, y_train)
                preds=svm.predict(X_test)
                dump(svm,f"models/svm{kernel}.joblib")
                plot_confusion_matrix(svm,X_test, y_test).plot(cmap='Blues')
                st.pyplot()
                precision = precision_score(y_test,preds)
                recall = recall_score(y_test,preds)
                f1 = f1_score(y_test,preds)
                st.metric("Precision Score:",f'{precision:.2f}',(precision-best_p))
                st.metric("Recall Score:",f'{recall:.2f}',(recall-best_r))
                st.metric("F1 Score:",f'{f1:.2f}',(f1-best_f1))
                if precision > best_p:
                    best_p = precision
                if recall > best_r:
                    best_r = recall
                if f1 > best_f1:
                    best_f1 = f1
                

if __name__ == '__main__':
    main()

