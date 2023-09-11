import streamlit as st
import numpy as np
import pandas as pd 
from joblib import dump, load
from PIL import Image 

img = Image.open('logo_cti.png')
 

st.image(img, width=200)

# Chargement du modèle entrainé
model =load('regression_model_saved.pkl')

# Fonction de prédiction et de probabilité
def prediction_faux_billets(features):
    X = pd.DataFrame([features])
    predictions = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return predictions, proba



# Créons notre interface Streamlit
def main ():
    st.title("DETECTION DES FAUX BILLETS!!!")
    st.header("!!!Entrez les caracteristiques du billet!!!")

    # entrer votre nom
    nom = st.text_input("Entrer votre nom", "Type Here ...")
 
    # display the name when the submit button is clicked
    # .title() is used to get the input text string
    if(st.button('Submit')):
      result = nom.title()
      st.success(result)
       
    #choix de l'option
      
    st.subheader("Cliquez sur le boutton pour choisir votre option ")
      
    # Selection box
    options = st.selectbox("options: ",
                   ['Sélectionez une option','Prédire', 'Testez'])
    if (options == 'Sélectionez une option'):
       st.write(nom ,'sélectionez une option avant de continuer' )

    else:
         # print the selected options
         st.write( nom, "votre option est: ", options)

    # Entrons des caractéristiques du billet
    diagonal = st.number_input('Diagonal', min_value=0.0, max_value=200.0)
    height_left = st.number_input('Hauteur gauche', min_value=0.0, max_value=200.0)
    height_right = st.number_input('Hauteur droite', min_value=0.0, max_value=200.0)
    margin_low = st.number_input('Marge inférieure', min_value=0.0, max_value=200.0)
    margin_up = st.number_input('Marge supérieure', min_value=0.0, max_value=200.0)
    length = st.number_input('Longueur', min_value=0.0, max_value=200.0)

   # Bouton prédire
    if st.button('Prédiction'):
       features = [diagonal, height_left, height_right, margin_low, margin_up, length]
       predictions, proba = prediction_faux_billets(features)
       if predictions == True:
           st.write(nom ,'votre billet est un bon billet.')
          
           st.write(f'Probabilité d\'obtention d\'un vrai billet : {proba[1]}')
       else:
           st.write(nom ,'votre billet est un faux billet.')
           
           st.write(f'Probabilité d\'obtention d\'un faux billet : {proba[0]}')
        
       # Ajout de la jauge pour afficher la probabilité
       st.progress(proba[1])

       #remerciement
       st.write( nom, "merci d'avoir utilisé notre application à bientôt ")
       


if __name__ == '__main__':
    main()
      