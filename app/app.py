import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Intelligent IT Operations",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_ticket_priority_model.pkl"

ISSUES_PATH = ROOT / "data" / "raw" / "issues.csv"

UTTERANCES_PATH = ROOT / "data" / "raw" / "sample_utterances.csv"


# ==========================================
# CUSTOM CSS
# ==========================================

st.html("""
<style>

/* Main background image */

.stApp {
    background:
    linear-gradient(
        rgba(5, 12, 28, 0.91),
        rgba(5, 12, 28, 0.94)
    ),
    url("https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=2000&q=85");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main content */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: rgba(8, 18, 38, 0.96);
    border-right: 1px solid rgba(0, 180, 255, 0.2);
}

/* Headings */

h1, h2, h3 {
    color: #ffffff !important;
}

/* Paragraphs */

p, label, span {
    color: #d5e5f5;
}

/* Header */

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(
        120deg,
        rgba(0, 130, 255, 0.20),
        rgba(105, 60, 255, 0.16)
    );
    border: 1px solid rgba(0, 180, 255, 0.3);
    margin-bottom: 25px;
    text-align: center;
    backdrop-filter: blur(12px);
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 16px;
    color: #c5d9f2;
}

/* Metric cards */

.metric-card {
    background: rgba(15, 32, 58, 0.83);
    border: 1px solid rgba(0, 180, 255, 0.25);
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.metric-card h2 {
    color: #00c8ff !important;
    font-size: 30px;
    margin: 5px;
}

.metric-card p {
    color: #a9c5df;
    font-size: 14px;
}

/* Input box */

textarea {
    background: rgba(10, 25, 48, 0.9) !important;
    color: white !important;
    border: 1px solid #1679bd !important;
    border-radius: 12px !important;
}

/* Buttons */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    background: linear-gradient(90deg, #007cf0, #6655ff);
    color: white;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(0, 150, 255, 0.35);
    color: white;
}

/* Result cards */

.result-card {
    padding: 25px;
    border-radius: 15px;
    background: rgba(12, 31, 57, 0.92);
    border: 1px solid rgba(0, 180, 255, 0.35);
    text-align: center;
    margin-top: 15px;
}

.priority {
    font-size: 32px;
    font-weight: 800;
    margin-top: 10px;
}

/* Info cards */

.info-card {
    background: rgba(15, 32, 58, 0.85);
    border: 1px solid rgba(0, 180, 255, 0.2);
    padding: 20px;
    border-radius: 14px;
    height: 150px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    background: rgba(10, 25, 48, 0.9);
}

/* Divider */

hr {
    border-color: rgba(0, 180, 255, 0.2);
}

</style>
""")


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# ==========================================
# LOAD HISTORICAL DATA
# ==========================================

@st.cache_data
def load_ticket_data():

    issues = pd.read_csv(ISSUES_PATH)

    utterances = pd.read_csv(UTTERANCES_PATH)

    text_df = utterances[
        ['issueid', 'actionbody']
    ].dropna()

    text_df = text_df.groupby('issueid')['actionbody'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()

    df = issues[
        ['id', 'issue_priority']
    ].merge(
        text_df,
        left_on='id',
        right_on='issueid',
        how='inner'
    )

    df = df.rename(
        columns={'actionbody': 'ticket_text'}
    )

    df['cleaned_text'] = df['ticket_text'].apply(clean_text)

    return df


# ==========================================
# LOAD RESOURCES
# ==========================================

try:

    model = load_model()

    df = load_ticket_data()

    tfidf = model.named_steps['tfidf']

    ticket_vectors = tfidf.transform(df['cleaned_text'])

except Exception as e:

    st.error(f"Unable to load project resources: {e}")

    st.info(
        "Check that your model and CSV files are in the correct folders."
    )

    st.stop()


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_priority(ticket):

    cleaned = clean_text(ticket)

    prediction = model.predict([cleaned])[0]

    return prediction


# ==========================================
# SIMILAR TICKET FUNCTION
# ==========================================

def find_similar_tickets(ticket, top_n=5):

    cleaned = clean_text(ticket)

    new_vector = tfidf.transform([cleaned])

    similarities = cosine_similarity(
        new_vector,
        ticket_vectors
    ).flatten()

    if not np.any(new_vector.toarray()):
        return pd.DataFrame()

    top_indices = np.argsort(similarities)[::-1]

    results = []

    for idx in top_indices:

        if not cleaned:
            continue

        results.append({
            "Ticket ID": df.iloc[idx]['id'],
            "Historical Priority": df.iloc[idx]['issue_priority'],
            "Ticket Description": df.iloc[idx]['ticket_text'],
            "Similarity Score": round(
                float(similarities[idx]) * 100, 2
            )
        })

        if len(results) == top_n:
            break

    return pd.DataFrame(results)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🤖 INTELLIGENT AI")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 Ticket Analysis",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.markdown("### TECHNOLOGIES")

    st.html("""
    - Python
    - Machine Learning
    - Natural Language Processing
    - TF-IDF
    - Linear SVM
    - Streamlit
    """)

    st.markdown("---")

    st.caption("Developed by Gaurav Katre")


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "🏠 Dashboard":

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        st.title("🤖 INTELLIGENT IT OPERATIONS")

        st.markdown(
            "**AI-POWERED IT SUPPORT TICKET CLASSIFICATION "
            "& INCIDENT MANAGEMENT SYSTEM**"
        )

        st.caption(
            "Machine Learning | Natural Language Processing"
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.html(f"""
        <div class="metric-card">
            <p>HISTORICAL TEXT TICKETS</p>
            <h2>{len(df):,}</h2>
        </div>
        """)

    with col2:

        st.html("""
        <div class="metric-card">
            <p>ML ALGORITHM</p>
            <h2>Linear SVM</h2>
        </div>
        """)

    with col3:

        st.html("""
        <div class="metric-card">
            <p>CORE TECHNOLOGY</p>
            <h2>NLP + ML</h2>
        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## 🚀 What Can This System Do?")

    col1, col2 = st.columns(2)

    with col1:

        st.html("""
        <div class="info-card">

            <h3>🎯 Priority Prediction</h3>

            <p>
            Automatically analyzes a support ticket
            and predicts its priority.
            </p>

        </div>
        """)

    with col2:

        st.html("""
        <div class="info-card">

            <h3>🔎 Similar Ticket Detection</h3>

            <p>
            Finds historically reported tickets
            with similar descriptions.
            </p>

        </div>
        """)

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Go to Ticket Analysis to test the Machine Learning model."
    )


# ==========================================
# TICKET ANALYSIS PAGE
# ==========================================

elif page == "🔍 Ticket Analysis":

    st.html("""
    <div class="hero">

        <h1>🔍 AI TICKET ANALYZER</h1>

        <p>
        Enter a technical problem and let the AI
        analyze its priority and find similar incidents.
        </p>

    </div>
    """)

    st.markdown("## 📝 Enter Ticket Description")

    ticket_text = st.text_area(
        "Describe the technical problem:",
        height=180,
        placeholder=(
            "Example: Customer is unable to complete "
            "an online payment. The transaction failed "
            "but the amount was deducted..."
        )
    )

    analyze_button = st.button(
        "🚀 ANALYZE TICKET"
    )

    if analyze_button:

        if len(ticket_text.strip()) < 10:

            st.warning(
                "Please enter a meaningful ticket description."
            )

        else:

            with st.spinner(
                "AI is analyzing the ticket..."
            ):

                prediction = predict_priority(ticket_text)

                similar_tickets = find_similar_tickets(
                    ticket_text,
                    top_n=5
                )

            st.markdown("---")

            st.markdown("## 🎯 Prediction Result")

            priority_colors = {
                "Blocker": "#ff3030",
                "Highest": "#ff5c5c",
                "High": "#ff9f43",
                "Medium": "#00c8ff",
                "Low": "#00d68f"
            }

            priority_color = priority_colors.get(
                str(prediction),
                "#00c8ff"
            )

            st.html(f"""
            <div class="result-card">

                <p>PREDICTED TICKET PRIORITY</p>

                <div class="priority"
                style="color:{priority_color};">

                    {prediction.upper()}

                </div>

            </div>
            """)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("## 🔎 Similar Historical Tickets")

            st.caption(
                "These tickets are retrieved using cosine similarity "
                "between TF-IDF text vectors. Similarity is not prediction confidence."
            )

            if len(similar_tickets) > 0:

                st.dataframe(
                    similar_tickets,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "No similar historical tickets were found."
                )

            st.markdown("---")

            st.success(
                "Ticket analysis completed successfully!"
            )


# ==========================================
# ABOUT PROJECT PAGE
# ==========================================

elif page == "ℹ️ About Project":

    st.html("""
    <div class="hero">

        <h1>ABOUT THE PROJECT</h1>

        <p>
        Intelligent IT Operations
        </p>

    </div>
    """)

    st.markdown("## 🎯 Main Objective")

    st.write("""
    The objective of this project is to build an
    AI-powered system that helps IT support teams
    analyze customer complaints.

    It predicts the priority of a technical issue
    and identifies similar historical tickets.
    """)

    st.markdown("## ⚙️ How It Works")

    st.markdown("""
    1. User enters a technical problem.
    2. NLP cleans and processes the text.
    3. TF-IDF converts text into numerical features.
    4. Linear SVM predicts ticket priority.
    5. Cosine Similarity finds similar historical tickets.
    6. Results are displayed on the dashboard.
    """)

    st.markdown("## 🧠 Machine Learning Model")

    st.write("""
    Algorithm: Linear Support Vector Machine (Linear SVM)

    Text Representation: TF-IDF

    Similarity Technique: Cosine Similarity

    Application: Streamlit
    """)

    st.markdown("---")

    st.caption(
        "Developed by Gaurav Katre | Intelligent IT Operations"
    )