# app.py — CLEAN LIVE DEMO
import streamlit as st
from transformers import pipeline
import joblib
import numpy as np

st.title("Support Ticket Triage")
st.caption("Paste ticket → Instant routing + SLA risk")

txt = st.text_area("Ticket", height=120)

if txt:
    p = pipeline("text-classification", model="models/priority")
    d = pipeline("text-classification", model="models/department")
    reg = joblib.load("models/sla_regressor.pkl")
    feat_pipe = pipeline("feature-extraction", model="models/sla_features")

    priority = p(txt)[0]['label'].split('_')[-1]
    dept = d(txt)[0]['label'].split('_')[-1]
    risk = reg.predict_proba(feat_pipe(txt)[0][0][0].reshape(1, -1))[0][1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Priority", f"P{priority}")
    col2.metric("Department", dept)
    col3.metric("SLA Risk", f"{int(risk*100)}%")

    st.success("Done!")
    st.balloons()