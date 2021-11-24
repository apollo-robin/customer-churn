# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 00:38:51 2021

@author: robin
"""
import pandas as pd
from joblib import load
import warnings 
warnings.filterwarnings("ignore")
import streamlit as st


class CreditChurnClassification():
    
    def __init__(self, model, features, acc):
        self.model = model
        self.features = features
        self.acc = acc
        


prediction_model = load('../Models/Churn_xgb_top.pkl')

#Setting up page configuration
st.set_page_config(page_title= 'Group 3', initial_sidebar_state='auto') 

st.markdown('<p style= "font-weight: bold; color: #1f4886; text-align:center; font-family:Segoe Script; font-size: 44px"> Credit Card Attrition <p>', unsafe_allow_html=True  )
st.markdown('<p style= "text-align:center; color: black;  font-size: 18px">We intend to solve the problem of cusotmer attrition by prediciting customers who are likely churn.  <p>', unsafe_allow_html=True) 


with st.form("info"):
    st.markdown("**Just fill in the details and we'll let you know** :yum: ")
    col1 , col2 = st.beta_columns(2)
    age = col1.slider('Age', min_value=18,max_value=75)
    gender = col2.selectbox('Gender', options=("M","F"))
    total_rel_cnt = col2.number_input("Total Relationship Count",value = 5)
    mth_inactive = col1.number_input("Months Inactive", value = 1)
    tot_rev_bal = col2.number_input("Total Revolving Balance",value = 1000)
    amt_chng_Q4_1 = col1.number_input("Total Amnout Change Q4-Q1",value = 0.4 )
    trans_amt = col2.number_input("Total Transaction Amount", value = 1000 )
    trans_cnt = col1.number_input("Total Transaction Count", value = 50)
    cnt_chng_Q4_1 = col2.number_input("Total Transaction Count Change Q4-Q1", value =0.4 )
    avg_util = col1.number_input("Average Utilisation Ratio", value = 0.5)
    
    submit = st.form_submit_button("Submit")
    

if submit:
    data = {'Customer_Age':age,
            'Gender': gender,
           'Total_Relationship_Count':total_rel_cnt,
           'Months_Inactive_12_mon':mth_inactive,
           'Total_Revolving_Bal':tot_rev_bal,
           'Total_Amt_Chng_Q4_Q1':amt_chng_Q4_1,
           'Total_Trans_Amt':trans_amt,
           'Total_Trans_Ct':trans_cnt,
           'Total_Ct_Chng_Q4_Q1':cnt_chng_Q4_1,
           'Avg_Utilization_Ratio':avg_util}
    
    X = pd.DataFrame(data, index = [0])
    attrition = prediction_model.model.predict(X)
    
    if attrition[0] == 0:
        st.balloons()
        st.success("The customer is not likely to attrite !")
    else:
        st.warning("This customer may attrite. Do something !")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    