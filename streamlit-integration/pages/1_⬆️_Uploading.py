import streamlit as st
from data_classes.data import Data
from data_classes.model import Model
from widgets.upload import upload as up

if "data" not in st.session_state or "model" not in st.session_state:
    st.session_state.data = Data()
    st.session_state.model = Model()

data = st.session_state.data
model = st.session_state.model

with st.container():
    up.upload_file_ui(data, model)
    up.upload_model_ui(model)
