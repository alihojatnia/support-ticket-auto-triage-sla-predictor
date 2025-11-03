import streamlit as st
from transformers import pipeline
p = pipeline("text-classification", model="models/priority")
d = pipeline("text-classification", model="models/department")
st.title("Support Ticket Triage")
txt = st.text_area("Paste ticket", height=120)
if txt:
    st.success(f"Priority: P{p(txt)[0]['label'][-1]}")
    st.info(f"Dept: {d(txt)[0]['label'].split('_')[-1]}")
    st.metric("SLA Breach Risk", f"{int(100 * (txt.count('urgent')+txt.count('down'))/len(txt.split()))}%")
    st.balloons()
