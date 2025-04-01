import streamlit as st
import os
import keras
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing import image

st.title("CLASSIFICATION OF LUNG CANCER USING VISION TRANSFORMER")
st.text("Upload an Image")

def chestScanPrediction(path, _model):
    model = keras.models.load_model(_model)
    # Loading Image
    img = image.load_img(path, target_size=(350, 350))
    # Normalizing Image
    norm_img = image.img_to_array(img) / 255
    # Converting Image to Numpy Array
    input_arr_img = np.array([norm_img])
    # Getting Predictions
    pred = np.argmax(model.predict(input_arr_img))
    
    # Printing Model Prediction with Detailed Prescription
    if pred == 0:
        st.write("### Diagnosis: Adenocarcinoma")
        st.write("""
        **Prescription:**
        - **Diet:**
          - Consume foods rich in antioxidants (berries, green tea, dark chocolate).
          - Include cruciferous vegetables like broccoli, cauliflower, and cabbage.
          - Opt for lean proteins such as fish, chicken, tofu, and legumes.
          - Avoid processed foods, red meat, and sugary drinks.
        - **Lifestyle:**
          - Stop smoking immediately (if applicable).
          - Engage in light physical activities such as walking, yoga, or swimming.
          - Practice breathing exercises like diaphragmatic breathing to improve lung capacity.
          - Avoid exposure to pollutants, dust, and secondhand smoke.
        - **Medical Recommendations:**
          - Consult with an oncologist for staging and treatment options, which may include targeted therapy, chemotherapy, or surgery.
          - Take prescribed medications on time and avoid self-medication.
          - Consider enrolling in a support group for emotional and psychological well-being.
        - **Follow-up:**
          - Schedule regular imaging tests (CT scans, PET scans) and blood work to monitor the progression or remission of the disease.
          - Maintain close communication with your healthcare provider.
        """)

    elif pred == 1:
        st.write("### Diagnosis: Large Cell Carcinoma")
        st.write("""
        **Prescription:**
        - **Diet:**
          - Eat a plant-based diet rich in fruits, vegetables, and whole grains.
          - Incorporate foods high in omega-3 fatty acids, such as walnuts, flaxseeds, and fatty fish.
          - Limit alcohol consumption and avoid high-fat, fried, and overly salty foods.
        - **Lifestyle:**
          - Maintain a smoke-free environment and avoid air pollutants.
          - Engage in moderate aerobic exercises to improve lung health, but avoid overexertion.
          - Practice mindfulness and meditation to manage stress.
        - **Medical Recommendations:**
          - Seek immediate consultation with a pulmonologist or oncologist.
          - Treatment may include a combination of surgery, radiation therapy, and immunotherapy depending on the stage.
          - Follow a personalized treatment plan tailored by your medical team.
        - **Follow-up:**
          - Regular check-ups are crucial to assess treatment effectiveness and adjust plans as needed.
          - Inform your doctor of any new or worsening symptoms such as persistent cough or shortness of breath.
        """)

    elif pred == 2:
        st.write("### Diagnosis: Normal")
        st.write("""
        **Prescription:**
        - **Diet:**
          - Maintain a balanced diet with plenty of fruits, vegetables, whole grains, and lean proteins.
          - Stay hydrated by drinking at least 2-3 liters of water daily.
          - Avoid smoking, alcohol, and high-sodium foods to preserve lung health.
        - **Lifestyle:**
          - Engage in regular physical activities, including brisk walking, swimming, or cycling.
          - Practice breathing exercises like pursed-lip breathing or pranayama to maintain optimal lung function.
          - Avoid exposure to airborne irritants, including dust, fumes, and pollen.
        - **Medical Recommendations:**
          - No medical intervention is required at this stage, but regular health check-ups are recommended.
          - Stay updated with vaccinations, including flu and pneumococcal vaccines.
        - **Follow-up:**
          - Consider an annual health check-up and lung function tests to monitor any changes.
          - Maintain a healthy lifestyle to prevent future complications.
        """)

    else:
        st.write("### Diagnosis: Squamous Cell Carcinoma")
        st.write("""
        **Prescription:**
        - **Diet:**
          - Include foods rich in vitamins A, C, and E (carrots, oranges, spinach) to support immune health.
          - Consume high-fiber foods like oats, whole grains, and legumes to aid digestion during treatment.
          - Drink herbal teas (e.g., chamomile, green tea) to help reduce inflammation.
          - Avoid smoked, grilled, or charred foods as they may increase the risk of complications.
        - **Lifestyle:**
          - Quit smoking and avoid alcohol consumption entirely.
          - Rest adequately but also include light exercises like stretching to maintain strength.
          - Manage stress through counseling, support groups, or relaxation techniques.
        - **Medical Recommendations:**
          - Treatment options may include surgery to remove affected tissue, radiation therapy, or chemotherapy.
          - Work closely with your oncologist to tailor a treatment plan based on your condition.
          - Keep track of medications and attend all scheduled therapy sessions.
        - **Follow-up:**
          - Regularly monitor progress through imaging tests and blood markers.
          - Report any side effects from treatment (e.g., nausea, fatigue) to your doctor promptly.
        """)

uploaded_file = st.file_uploader("Choose a scan ...", type="png")

if uploaded_file is not None:
    data = Image.open(uploaded_file)
    st.image(data, caption='Uploaded Scan.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    chestScanPrediction(uploaded_file, 'ct_vgg_best_model.keras')

