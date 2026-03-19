import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Disease Prediction", page_icon="💊", layout="centered")

st.title("🩺 Disease Prediction From Symptoms")
st.write("Select up to 5 symptoms to predict the most likely disease using our Machine Learning model.")

l1=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
    'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue',
    'weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat',
    'irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion',
    'headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps',
    'bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'fluid_overload','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
        'Peptic ulcer diseae','AIDS','Diabetes ','Gastroenteritis','Bronchial Asthma','Hypertension ',
        'Migraine','Cervical spondylosis',
        'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heart attack','Varicose veins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("Training.csv")
    
    # Matching exact replacement logic:
    df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)
        
    # Remove any unmapped values
    df = df[df['prognosis'].apply(lambda x: isinstance(x, (int, float)))]
    
    X = df[l1]
    y = np.ravel(df[["prognosis"]]).astype('int')
    
    gnb = MultinomialNB()
    gnb.fit(X, y)
    return gnb

try:
    with st.spinner("Training model..."):
        model = load_and_train_model()
except Exception as e:
    st.error(f"Error loading training data: {e}")
    st.stop()

options = ["None"] + sorted(l1)

st.markdown("### First Symptom")
s1 = st.selectbox("First symptom", options, key="s1", label_visibility="collapsed")
st.markdown("### Second Symptom")
s2 = st.selectbox("Second symptom", options, key="s2", label_visibility="collapsed")
st.markdown("### Third Symptom")
s3 = st.selectbox("Third symptom", options, key="s3", label_visibility="collapsed")
st.markdown("### Fourth Symptom")
s4 = st.selectbox("Fourth symptom", options, key="s4", label_visibility="collapsed")
st.markdown("### Fifth Symptom")
s5 = st.selectbox("Fifth symptom", options, key="s5", label_visibility="collapsed")

if st.button("Predict 🔮", type="primary"):
    psymptoms = [s1, s2, s3, s4, s5]
    if all(s == "None" for s in psymptoms):
        st.warning("Please enter at least one symptom to predict.")
    else:
        l2 = [0] * len(l1)
        for k in range(len(l1)):
            for z in psymptoms:
                if z == l1[k]:
                    l2[k] = 1
                    
        inputtest = [l2]
        predict = model.predict(inputtest)
        predicted = int(predict[0])
        
        # Check if predictable disease
        predicted_disease = "Unknown"
        if 0 <= predicted < len(disease):
            predicted_disease = disease[predicted]
            
        st.success(f"### Predicted Disease: **{predicted_disease}**")
