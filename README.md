# AI-Driven-Citizen-Grievance-Sentiment-Analysis-System
AI-Driven Citizen Grievance &amp; Sentiment Analysis System
Build an AI-powered NLP system that automatically ingests citizen
feedback, categorizes complaints into relevant government departments, and
performs sentiment analysis to prioritize issues based on urgency and
emotional tone.

### Technology Stack

| Component               | Technology                     |
| ----------------------- | ------------------------------ |
| Data Processing & NLP   | Python, NLTK, spaCy            |
| Machine Learning        | Scikit-Learn                   |
| Transformers (Advanced) | HuggingFace BERT               |
| API Deployment          | FastAPI, Uvicorn               |
| Visualization           | Matplotlib, Seaborn, WordCloud |

---
###  4-Week Engineering Roadmap

| Week   | Notebook                                | Key Deliverables                                             |
| ------ | --------------------------------------- | ------------------------------------------------------------ |
| Week 1 | `Week1_Text_Cleaning_EDA.ipynb`         | Data generation, text cleaning, word clouds, n-gram analysis |
| Week 2 | `Week2_Department_Categorization.ipynb` | TF-IDF, Naive Bayes, SVM, cross-validation                   |
| Week 3 | `Week3_Sentiment_Analysis.ipynb`        | Sentiment classifier, class balancing, priority scoring      |
| Week 4 | `Week4_API_Evaluation.ipynb`            | Confusion matrices, FastAPI deployment, API testing          |

---

###  Models Used

| Model               | Task                                   | Why                                                      |
| ------------------- | -------------------------------------- | -------------------------------------------------------- |
| Naive Bayes         | Department Classification (Baseline)   | Fast and effective for text classification               |
| Logistic Regression | Department + Sentiment Classification  | Performs well with TF-IDF features                       |
| Linear SVM          | Department Classification (Best Model) | State-of-the-art performance for NLP text classification |

---

###  Evaluation Metrics

| Metric         | Used For           | Why Important                              |
| -------------- | ------------------ | ------------------------------------------ |
| Macro F1 Score | Both classifiers   | Handles class imbalance fairly             |
| Accuracy       | Both classifiers   | Measures overall correctness               |
| Precision      | Sentiment Analysis | Reduces false urgency alerts               |
| Recall         | Sentiment Analysis | Ensures Critical/Urgent cases are detected |

---
###  Users & Use Cases

| Persona              | Goal                                          | Key Benefit                                           |
| -------------------- | --------------------------------------------- | ----------------------------------------------------- |
|  Civic Official | Rapid complaint triage without manual reading | AI-tagged dashboard with priority queue               |
|  Citizen / User    | Quick complaint acknowledgment and routing    | Automatically routes complaints to correct department |
|  Policy Maker      | Understand public sentiment across wards      | Data-driven insights and trend analysis               |

---

####  Civic Official

**Use Case:**

* Reviews AI-tagged complaint dashboard daily
* Instantly identifies Critical/Urgent complaints
* Avoids manually reading hundreds of complaints

**Key Benefit:**
 Reduces manual triage time by nearly **80%**

---

####  Citizen / User

**Use Case:**

* Submits complaints in plain natural language
* AI automatically detects department and urgency
* No need to manually select departments or categories

**Key Benefit:**
 Faster complaint acknowledgment and routing

---

####  Policy Maker

**Use Case:**

* Analyzes complaint sentiment across Bengaluru wards
* Tracks departments receiving the most critical complaints
* Monitors trends and service improvements over time

**Key Benefit:**
 Enables data-driven governance and policy decisions

---
