
<img width="696" height="439" alt="git_git" src="https://github.com/user-attachments/assets/a2862b6b-5ff5-4422-9b50-16388ee18637" />




### support Ticket Auto-Triage  


---

#### What it does
Paste any support ticket → get **instant**:
- **Priority** (P0–P3)  
- **Department** (AR · Billing · Tech)  
- **SLA Breach Risk %**  
- **Balloons** when routed

Built with **DistilBERT** + **Logistic Regression** on CPU.  

---

#### 3 Commands to Glory

```bash
# 1. Clone & enter
git clone https://github.com/alihojatnia/support-ticket-auto-triage-sla-predictor.git
cd support-ticket-auto-triage-sla-predictor

# 2. Install
pip install -r requirements.txt

# 3. Train (6 sec) + Launch
python train_models.py
streamlit run app.py
```

Open **http://localhost:8501** → paste a ticket → watch magic.

---

#### Results (tiny data, big flex)

```
Priority F1     : 1.000
Department F1   : 1.000
SLA MAE         : 0.00
```

| Metric       | Score |
|--------------|-------|
| Train time   | 6 sec |
| Model size   | < 300 MB |


---

#### 🛠 Tech Stack
- **Backbone**: `distilbert-base-uncased`  
- **Framework**: HuggingFace Transformers + Datasets  
- **UI**: Streamlit (1-click live demo)  
- **SLA**: Logistic on BERT [CLS] embeddings  
- **Data**: 188 perfect real tickets (included)

---

#### Project Structure
```
├── train_models.py      
├── app.py               
├── requirements.txt
├── metrics.txt          
├── models/              
└── data/
    └── processed/
        └── train_tiny.csv   
```

---

#### Live Demo Example
**Paste this:**
```
URGENT: server down, losing $10k/min, fix NOW or we sue
```

**You get:**
- **Priority:** P0  
- **Department:** Tech  
- **SLA Risk:** 94%  

---




