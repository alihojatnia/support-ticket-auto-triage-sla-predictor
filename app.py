# app.py — INSTANT DEMO
import streamlit as st
from transformers import pipeline

p = pipeline("text-classification", model="models/priority")
d = pipeline("text-classification", model="models/department")

st.title("Support Ticket Triage")
txt = st.text_area("Paste ticket", height=120)
if txt:
    prio = p(txt)[0]['label'].split('_')[-1]
    dept = d(txt)[0]['label'].split('_')[-1]
    risk = int(100 * (txt.lower().count('urgent') + txt.lower().count('down')) / max(1, len(txt.split())))
    st.success(f"Priority: P{prio}")
    st.info(f"Department: {dept}")
    st.metric("SLA Breach Risk", f"{risk}%")
    st.balloons()