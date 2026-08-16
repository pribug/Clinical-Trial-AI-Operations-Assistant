import streamlit as st
from study_setup import render_study_setup

st.set_page_config(
    page_title="Clinical Trial AI Operations Assistant",
    layout="wide"
)

render_study_setup()