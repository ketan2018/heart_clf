import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import requests
from streamlit_lottie import st_lottie 
# import streamlit as st
# st.title("""Heart Disease Prediction App""")
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.markdown(""" <style> .mine {
font-size:50px ; font-family: 'Sans Serif';text-align:center;} 
</style> """, unsafe_allow_html=True)
st.markdown('<h2 class="mine"> Get to Know Your Heart</p>', unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:
    cp_value = st.selectbox('chest pain type:',('Typical Angina','Atypical Angina','Non Anginal Pain','Asymptomatic'))  
                                                                

with col2:
    sex_value = st.selectbox('sex:',('Male','Female'))

with col3:
    restecg_value = st.selectbox('resting electrocardiography',('Normal','Having ST-T Wave Abnormality',"showing probable or definite left ventricular hypertrophy by Estes' criteria"))




col4, col5, col6 = st.columns(3)

with col4:
    exang_value = st.selectbox('exercise induced angina',('Yes','No'))

with col5:
    slope_value=st.selectbox('slope of the peak exercise',('Unsloping','Downsloping','Flat'))

with col6:
    ca_value=st.selectbox('number of major vessels',('0','2','1','3','4'))


col7,col8,col9=st.columns(3)

with col7:
    thal_value=st.selectbox('thalassemia',('Null','Fixed Defect','Normal bloodflow','Reversible defect'))

with col8:
     age_value = st.number_input('age', 28,78,35)

with col9:
    old_peak_value=st.number_input('ST depression induced by exercise relative to rest', 0.0,6.2,3.2)




col10,col11,col12=st.columns(3)


with col10:
     trestbps_value = st.slider('resting blood pressure', 94,200,110)

with col11:
     thalach_value=st.slider('maximum heart rate achieved', 70,202,160)

with col12:
    chol_value= st.slider('serum cholestoral in mg/dl', 126,564,250)









# cp_value = st.sidebar.selectbox('cp',('3','2','1','0'))
# sex_value = st.sidebar.selectbox('sex',('1','0'))
# restecg_value = st.sidebar.selectbox('restecg',('0','1','2'))
# trestbps_value = st.sidebar.slider('trestbps', 94,200,110)
# exang_value = st.sidebar.selectbox('exang',('0','1'))
# chol_value= st.sidebar.slider('chol', 126,564,250)
# slope_value=st.sidebar.selectbox('slope',('0','2','1'))
# ca_value=st.sidebar.selectbox('ca',('0','2','1','3','4'))
# thal_value=st.sidebar.selectbox('thal',('0','2','1','3'))
# thalach_value=st.sidebar.slider('thalach', 70,202,160)
# old_peak_value=st.sidebar.slider('oldpeak', 0.0,6.2,3.2)

chest_pain={'Typical Angina': 0,'Atypical Angina':1,'Non Anginal Pain': 2,'Asymptomatic':3}

sex_type={'Male':1,'Female':0}

restecg_types={'Normal':0,'Having ST-T Wave Abnormality': 1,"showing probable or definite left ventricular hypertrophy by Estes' criteria": 2}

exang_type={'Yes': 1,'No': 0}

slope_type={'Unsloping': 0,'Downsloping':2,'Flat':1}

thal_types={'Null':0,'Fixed Defect':2,'Normal bloodflow':1,'Reversible defect':3}

o,p,q=st.columns(3)

with p:
    r_button=st.button("Let's Checkout The Results...")


data = {'age': age_value,
        'chol': chol_value,
        'trestbps': trestbps_value,
        'thalach':thalach_value,
        'oldpeak':old_peak_value,
        'restecg': restecg_types[restecg_value],
        'cp': chest_pain[cp_value],
        'exang':exang_type[exang_value],
        'sex': sex_type[sex_value],
        'slope':slope_type[slope_value],
        'ca':ca_value,
        'thal':thal_types[thal_value]}

features = pd.DataFrame(data, index=[0])

input_df = features[:1]

num_feat=['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler=StandardScaler()

scaler.fit(input_df[num_feat])

input_df[num_feat]=scaler.transform(input_df[num_feat])




load_clf = pickle.load(open('knn.pkl', 'rb'))


prediction = load_clf.predict(input_df)

good_msg='No Need To Worry, Your Heart Is In Good Condition'
bad_msg='Sorry to Say,There Is Some Problem With Your Heart'
if r_button==True:
    if prediction[:1]==1:
        st.markdown(""" <style> .result{
        font-size:50px ; font-family: "Gill Sans Extrabold", sans-serif;text-align:center;} 
        </style> """, unsafe_allow_html=True)
        st.markdown(f'<h2 class="result"> {bad_msg} </p>', unsafe_allow_html=True)

    else:
        st.markdown(""" <style> .result{
        font-size:50px ; font-family: "Gill Sans Extrabold", sans-serif;text-align:center;} 
        </style> """, unsafe_allow_html=True)
        st.markdown(f'<h2 class="result"> {good_msg} </p>', unsafe_allow_html=True)
        st.balloons()