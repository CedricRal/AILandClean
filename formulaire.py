import streamlit as st
import base64
import pandas as pd


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("forest.jpg")

page_bg_img = f"""
<style>

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.header("AI-Clean Land")

#Collecter les régions
myData = pd.read_excel("D:\ODC\Projet\Dataset\Dataodc.xlsx")
regi = myData["Region"].drop_duplicates()


st.title(":green[Formulaire] de collecte de données")
    
option=st.selectbox("Région", regi)
    
#Collecter les districts
df=myData.loc[myData["Region"]==str(option)]
districts=df["District"].drop_duplicates()
option_dist=st.selectbox("District", districts)
 
#Collecter les communes
df2=myData.loc[myData["District"]==str(option_dist)]
communes=df2["Commune"].drop_duplicates()
option_comm=st.selectbox("Commune", communes)

#Collecter les Fokontany
df3=myData.loc[myData["Commune"]==str(option_comm)]
fokontany=df3["Fokontany"].drop_duplicates()
option_fokontany=st.selectbox("Fokontany", fokontany)

params=[]
p1=st.text_input(
    "Anciennes ZDAL*",
)
params.append(p1)

p2=st.text_input(
    "ZDAL* simplement nettoyées",
)
params.append(p2)

p3=st.text_input(
    "ZDAL* transformées",
)
params.append(p3)

p4=st.text_input(
    "Latrines non flyproof",
)
params.append(p4)

p5=st.text_input(
    "Latrines flyproof partagées ",
)
params.append(p5)

p6=st.text_input(
    "Latrines flyproof non partagées (par menage)",
)
params.append(p6)

p7=st.text_input(
    "Latrines avec dalle lavable, NON flyproof ",
)
params.append(p7)

p8=st.text_input(
    "Latrines avec dalle lavable, flyproof et partagée ",
)
params.append(p8)

p9=st.text_input(
    "Latrines avec dalle lavable flyproof par ménage (non partagée)",
)
params.append(p9)

submit='<a href=\"traitement\"><button type="submit">Valider</button></a>'
st.markdown(submit, unsafe_allow_html=True)

if 'key' not in st.session_state:
    st.session_state['key'] = params

# Session State also supports the attribute based syntax
if 'key' not in st.session_state:
    st.session_state.key = params

st.write(st.session_state.key)

abbreviation='<p style=" font-weight: italic; font-size: 15px; color : gray"> *ZDAL: Zone de Défecation  à l Air Libre <p>'
st.markdown(abbreviation, unsafe_allow_html=True)

