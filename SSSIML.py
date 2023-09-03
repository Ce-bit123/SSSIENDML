import streamlit as st
import shap
from streamlit_shap import st_shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Title
st.header("Risk prediction of early neurological deterioration within 72 hours in single small subcortical infarct")

# Input bar 1
# Input bar 2
IL8 = st.number_input("Enter IL8 level")
uPA = st.number_input("Enter uPA level")
AXIN1 = st.number_input("Enter AXIN1 level")
IL10 = st.number_input("Enter IL10 level")
STAMBP= st.number_input("Enter STAMBP level")
# Dropdown input
#posteriortype = st.selectbox("Whether it conforms to the posterior type", ("Yes", "No"))
#antiplatelet = st.selectbox("Whether dual antiplatelet therapy was administered after admission",("Yes","No"))
#PAD = st.selectbox("Whether it manifests as parent artery disease",("Yes","No"))
# If button is pressed

if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("clfgbm.pkl")
    # Store inputs into dataframe
    X = pd.DataFrame([[IL8,uPA,AXIN1,IL10,STAMBP]],
                     columns=["IL8", "uPA","AXIN1",
                       "IL10","STAMBP"])
    #X = X.replace(["Yes", "No"], [1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]

    #explainer = shap.TreeExplainer(clf)
    #shap_values = explainer.shap_values(X)
    # f = plt.figure()
    # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
    # f.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
    # Output prediction
    # P = mpimg.imread("shap_force_plot.png")
    # st.image(P, caption="shap_force_plot", channels="RGB")
    # st_shap(shap.plots.waterfall(shap_values[0]), height=300)
    # st_shap(shap.plots.beeswarm(shap_values), height=300)
    #st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :],X.iloc[0, :]), height=200, width=700)
    if prediction == 0:
        st.text(f"This patient has a higher probability of Non-END within 72 hours")
    else:
        st.text(f"This patient has a higher probability of END within 72 hours")

