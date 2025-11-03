# streamlit_app_final_corrected_6_with_auth.py
# PdM Prototype Dashboard with modern futuristic UI
# Enhanced with digital futuristic design, AR-inspired visuals and interactive elements

import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import joblib, io, os, json, hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Import TensorFlow/Keras for Autoencoder (ensure it's installed)
try:
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# Import for PDF generation (Using ReportLab)
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from PIL import Image

# -----------------------------
# Streamlit app starts here - MUST BE FIRST
# -----------------------------
st.set_page_config(layout='wide', page_title='PdM Futuristic Dashboard', page_icon="🔮")

# Apply custom CSS for futuristic design
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main background with futuristic gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    /* ========== SIDEBAR STYLING - ENHANCED ========== */
    /* Sidebar background with futuristic gradient - FIXED */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460) !important;
    }
    
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 198, 255, 0.5);
        box-shadow: 0 0 20px rgba(0, 198, 255, 0.3);
    }
    
    /* Sidebar text color - LIGHT BLUE/GREEN */
    section[data-testid="stSidebar"] * {
        color: #00ffcc !important;
        font-family: 'Arial', sans-serif;
    }
    
    /* Specific sidebar elements */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stText,
    section[data-testid="stSidebar"] .stTitle,
    section[data-testid="stSidebar"] .stHeader,
    section[data-testid="stSidebar"] .stSubheader,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #00ffcc !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    
    /* ========== ALL BUTTONS STYLING - FIXED ========== */
    /* Apply to ALL buttons globally */
    .stButton > button {
        background: linear-gradient(45deg, #00c6ff, #0072ff) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px 0 rgba(0, 114, 255, 0.4) !important;
        font-family: 'Arial', sans-serif !important;
        width: 100% !important;
        margin: 5px 0 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(0, 114, 255, 0.6) !important;
        background: linear-gradient(45deg, #0072ff, #00c6ff) !important;
    }
    
    /* Specific button overrides to ensure consistency */
    section[data-testid="stSidebar"] .stButton > button,
    .stForm .stButton > button,
    .main-sheet .stButton > button,
    div[data-testid="stVerticalBlock"] .stButton > button {
        background: linear-gradient(45deg, #00c6ff, #0072ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px 0 rgba(0, 114, 255, 0.4) !important;
        font-family: 'Arial', sans-serif !important;
    }
    
    /* Sidebar expander headers */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #00ffcc !important;
        font-weight: bold !important;
        font-size: 16px !important;
        background: rgba(0, 255, 204, 0.1) !important;
        border-radius: 10px !important;
        padding: 10px 15px !important;
        margin: 5px 0 !important;
        border: 1px solid rgba(0, 255, 204, 0.3) !important;
    }
    
    /* Sidebar select boxes and inputs */
    section[data-testid="stSidebar"] .stSelectbox>div>div>div,
    section[data-testid="stSidebar"] .stTextInput>div>div>input,
    section[data-testid="stSidebar"] .stDateInput>div>div>input {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #00ffcc !important;
        border: 1px solid rgba(0, 255, 204, 0.5) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar date input */
    section[data-testid="stSidebar"] .stDateInput>div>div>input {
        color: #00ffcc !important;
    }
    
    /* Cards and containers */
    .main-sheet {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(100, 100, 255, 0.3);
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Headers with neon effect */
    .futuristic-header {
        background: linear-gradient(90deg, #00c6ff, #0072ff, #00c6ff);
        background-size: 200% auto;
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 0 20px rgba(0, 198, 255, 0.7);
        animation: shimmer 3s linear infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% 0%; }
        100% { background-position: 200% 0%; }
    }
    
    .futuristic-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Metrics with glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.4);
    }
    
    /* Sliders with modern look */
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
    }
    
    /* Dataframes with glass effect */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Alarm level indicators */
    .alarm-critical { background: linear-gradient(45deg, #ff416c, #ff4b2b) !important; }
    .alert-alarm { background: linear-gradient(45deg, #ff9966, #ff5e62) !important; }
    .alert-warning { background: linear-gradient(45deg, #f9d423, #ff4e50) !important; }
    .alert-normal { background: linear-gradient(45deg, #56ab2f, #a8e063) !important; }
    
    /* Form styling */
    .stForm {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Text input styling - FIXED FOR READABILITY */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(100, 100, 255, 0.5);
        color: #1f1f1f !important;
        border-radius: 10px;
        font-weight: 500;
    }
    
    /* Select box styling - FIXED FOR READABILITY */
    .stSelectbox>div>div>div {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(100, 100, 255, 0.5);
        color: #1f1f1f !important;
        border-radius: 10px;
        font-weight: 500;
    }
    
    /* Text color for all Streamlit text elements in main area */
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption {
        color: white !important;
    }
    
    /* Fix for all text elements in main area to be white */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] h1, 
    div[data-testid="stMarkdownContainer"] h2, 
    div[data-testid="stMarkdownContainer"] h3, 
    div[data-testid="stMarkdownContainer"] h4, 
    div[data-testid="stMarkdownContainer"] h5, 
    div[data-testid="stMarkdownContainer"] h6 {
        color: white !important;
    }
    
    /* Fix for widget labels in main area */
    .stTextInput label, .stSelectbox label, .stSlider label, .stDateInput label {
        color: white !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #00c6ff, #0072ff) !important;
    }
    
    /* Admin section styling */
    .admin-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Date input styling */
    .stDateInput>div>div>input {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #1f1f1f !important;
        border-radius: 10px;
    }
    
    /* Green text for algorithm summary */
    .green-summary {
        color: #00ffcc !important;
        font-weight: bold;
        font-size: 16px;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
        background: rgba(0, 255, 204, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 204, 0.3);
        margin: 10px 0;
    }
    
    /* Login page buttons */
    .login-button {
        background: linear-gradient(45deg, #00c6ff, #0072ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px 0 rgba(0, 114, 255, 0.4) !important;
        font-family: 'Arial', sans-serif !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom styled headers
def styled_header(title_text):
    st.markdown(f"""
    <div class="futuristic-header">
        {title_text}
    </div>
    """, unsafe_allow_html=True)

# Custom metric display with futuristic design
def futuristic_metric(label, value, delta=None, level=0):
    level_class = ""
    if level == 3:
        level_class = "alarm-critical"
    elif level == 2:
        level_class = "alert-alarm" 
    elif level == 1:
        level_class = "alert-warning"
    else:
        level_class = "alert-normal"
    
    st.markdown(f"""
    <div class="metric-card {level_class}">
        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; margin-bottom: 5px;">{value}</div>
        {f'<div style="font-size: 12px; opacity: 0.8;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Authentication / user DB code
# -----------------------------
USERS_FILE = "users_db.json"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def init_users_db():
    """Create the users DB file with a default admin if it doesn't exist."""
    if not os.path.exists(USERS_FILE):
        admin_pw = hash_password("1234")  # default admin password (change after first login)
        users = {
            "admin": {"password": admin_pw, "role": "admin"}
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
        return users
    else:
        return load_users_db()

def load_users_db():
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users_db(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def verify_credentials(username: str, password: str) -> (bool, str):
    """
    Return (success, role_or_error)
    """
    users = load_users_db()
    if username not in users:
        return False, "User not found"
    stored = users[username]["password"]
    if stored == hash_password(password):
        return True, users[username].get("role", "viewer")
    return False, "Invalid password"

def require_admin_confirmation(admin_password_input: str) -> bool:
    users = load_users_db()
    admin_hash = users.get("admin", {}).get("password")
    return admin_hash == hash_password(admin_password_input)

def add_user(admin_password: str, new_username: str, new_password: str, role: str) -> (bool, str):
    if not require_admin_confirmation(admin_password):
        return False, "Admin confirmation failed"
    users = load_users_db()
    if new_username in users:
        return False, "Username already exists"
    users[new_username] = {"password": hash_password(new_password), "role": role}
    save_users_db(users)
    return True, "User added"

def delete_user(admin_password: str, username_to_delete: str) -> (bool, str):
    if not require_admin_confirmation(admin_password):
        return False, "Admin confirmation failed"
    users = load_users_db()
    if username_to_delete not in users:
        return False, "User not found"
    if username_to_delete == "admin":
        return False, "Cannot delete primary admin"
    users.pop(username_to_delete)
    save_users_db(users)
    return True, "User deleted"

def change_user_password(admin_password: str, username: str, new_password: str) -> (bool, str):
    if not require_admin_confirmation(admin_password):
        return False, "Admin confirmation failed"
    users = load_users_db()
    if username not in users:
        return False, "User not found"
    users[username]["password"] = hash_password(new_password)
    save_users_db(users)
    return True, "Password changed"

def change_own_password(username: str, old_password: str, new_password: str) -> (bool, str):
    users = load_users_db()
    if username not in users:
        return False, "User not found"
    if users[username]["password"] != hash_password(old_password):
        return False, "Current password incorrect"
    users[username]["password"] = hash_password(new_password)
    save_users_db(users)
    return True, "Password updated"

# Initialize user DB if needed
init_users_db()

# Apply custom CSS
apply_custom_css()

# -----------------------------
# Streamlit app starts here
# -----------------------------

# Session state keys for auth
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'current_role' not in st.session_state:
    st.session_state.current_role = None

def login_widget():
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        color: white;
        font-size: 42px;
        font-weight: bold;
        margin: 50px auto;
        max-width: 600px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        animation: pulse 2s infinite;">
        🔮 PdM Futuristic Dashboard
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='
        background: rgba(255,255,255,0.05); 
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255,255,255,0.2);
        max-width: 500px;
        margin: 0 auto;'>
    """, unsafe_allow_html=True)
    
    st.markdown("### Secure Login Access")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("👤 Username", value="admin", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("🚀 Access System", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if submitted:
        ok, role_or_msg = verify_credentials(username.strip(), password)
        if ok:
            st.session_state.logged_in = True
            st.session_state.current_user = username.strip()
            st.session_state.current_role = role_or_msg
            st.success(f"🛸 Welcome {username}! Access granted to futuristic dashboard.")
            st.balloons()
            st.rerun()
        else:
            st.error(f"🚫 {role_or_msg}")

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.current_role = None
    st.rerun()

# If not logged in, show login page and stop
if not st.session_state.logged_in:
    login_widget()
    st.stop()

# After here, user is authenticated
with st.sidebar:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.9), rgba(22, 33, 62, 0.9));
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 198, 255, 0.3);
        margin-bottom: 20px;
        border: 1px solid rgba(0, 255, 204, 0.3);">
        <div style="font-size: 18px; font-weight: bold; color: #00ffcc; text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);">👤 {st.session_state.current_user}</div>
        <div style="font-size: 14px; color: #00ffcc; opacity: 0.8;">{st.session_state.current_role.upper()}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Log Out", use_container_width=True):
        logout()

# Main header after login
st.markdown("""
<div style="
    background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
    background-size: 200% auto;
    padding: 30px;
    border-radius: 25px;
    text-align: center;
    color: white;
    font-size: 46px;
    font-weight: bold;
    letter-spacing: 1px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    margin-bottom: 40px;
    border: 1px solid rgba(255,255,255,0.3);
    animation: shimmer 3s linear infinite;">
    🚀 Predictive Maintenance Futuristic Dashboard
</div>
""", unsafe_allow_html=True)

# Admin-only user management UI - REORGANIZED
def user_management_ui():
    with st.container():
        st.markdown('<div class="main-sheet">', unsafe_allow_html=True)
        st.markdown("### 👥 User Management Portal")
        
        users = load_users_db()
        st.markdown("##### Current System Users")
        user_df = pd.DataFrame([{"username": k, "role": v.get("role", "viewer")} for k,v in users.items()])
        st.dataframe(user_df, use_container_width=True)
        
        st.markdown("---")
        
        # Use tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["➕ Add User", "🗑️ Delete User", "🔄 Change Password", "🔐 My Password"])
        
        with tab1:
            st.markdown("#### Add New User")
            with st.form("add_user_form"):
                admin_pw = st.text_input("🔑 Admin Password", type="password", key="add_admin_pw")
                new_username = st.text_input("👤 New Username", key="add_username")
                new_password = st.text_input("🔒 New Password", type="password", key="add_password")
                role = st.selectbox("🎭 Role", ["viewer", "engineer", "admin"], index=0, key="add_role")
                add_sub = st.form_submit_button("✨ Create User", use_container_width=True)
            if add_sub:
                ok, msg = add_user(admin_pw, new_username.strip(), new_password, role)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

        with tab2:
            st.markdown("#### Delete User")
            with st.form("del_user_form"):
                admin_pw2 = st.text_input("🔑 Admin Password", type="password", key="del_admin_pw")
                users = load_users_db()
                del_username = st.selectbox("👤 Select User to Delete", list(users.keys()), key="del_user_select")
                del_sub = st.form_submit_button("❌ Delete User", use_container_width=True)
            if del_sub:
                ok, msg = delete_user(admin_pw2, del_username.strip())
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

        with tab3:
            st.markdown("#### Change User Password")
            with st.form("change_pw_form"):
                admin_pw3 = st.text_input("🔑 Admin Password", type="password", key="chg_admin_pw")
                users = load_users_db()
                target_user = st.selectbox("👤 Select User", list(users.keys()), key="chg_user_select")
                new_pw = st.text_input("🔒 New Password", type="password", key="chg_new_pw")
                chg_sub = st.form_submit_button("🔄 Change Password", use_container_width=True)
            if chg_sub:
                ok, msg = change_user_password(admin_pw3, target_user.strip(), new_pw)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

        with tab4:
            st.markdown("#### Change My Password")
            with st.form("own_pw_form"):
                cur_pw = st.text_input("🔑 Current Password", type="password", key="own_cur_pw")
                new_pw2 = st.text_input("🔒 New Password", type="password", key="own_new_pw")
                own_sub = st.form_submit_button("🔄 Update Password", use_container_width=True)
            if own_sub:
                username = st.session_state.current_user
                ok, msg = change_own_password(username, cur_pw, new_pw2)
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# If current user is admin, show user management in sidebar expander
if st.session_state.current_role == "admin":
    with st.sidebar.expander("👑 Admin Portal", expanded=False):
        user_management_ui()

# -----------------------------
# Data loading and main application
# -----------------------------

FEATURES_CSV = "stranding_features_advanced.csv"

@st.cache_data
def load_features():
    try:
        df = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'], encoding='utf-8')
        return df
    except FileNotFoundError:
        st.error(f"🚫 Error: Feature file '{FEATURES_CSV}' not found. Please run feature_extraction.py first.")
        return pd.DataFrame()
    except UnicodeDecodeError:
        # Try fallback encoding
        df = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'], encoding='latin1')
        return df

df = load_features()
if df.empty:
    st.stop()

# --- Sidebar Controls (Time Window Selection and Machine Tag) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Control Panel")

tags = df['tag'].unique().tolist()
tag = st.sidebar.selectbox('🏷️ Machine Tag', tags)
window_df = df[df['tag']==tag].reset_index(drop=True)

# Time Window Selection
min_date = window_df['timestamp'].min().date()
max_date = window_df['timestamp'].max().date()

date_range = st.sidebar.date_input(
    "📅 Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) # Include the end day
    window_df = window_df[(window_df['timestamp'] >= start_date) & (window_df['timestamp'] < end_date)].reset_index(drop=True)

st.sidebar.markdown(f'📊 **Windows in Range: {len(window_df)}**')

# --- Alarm Policy Implementation ---
def apply_alarm_policy(anomaly_scores_series, contamination):
    if anomaly_scores_series.empty:
        return pd.Series(0, index=anomaly_scores_series.index)

    anomaly_threshold = anomaly_scores_series.quantile(1 - contamination)
    anomalies = anomaly_scores_series[anomaly_scores_series > anomaly_threshold]
    if anomalies.empty:
        return pd.Series(0, index=anomaly_scores_series.index)

    warning_threshold = anomalies.quantile(0.33)
    alarm_threshold = anomalies.quantile(0.66)

    alarm_levels = pd.Series(0, index=anomaly_scores_series.index)
    critical_indices = anomalies[anomalies >= alarm_threshold].index
    alarm_levels.loc[critical_indices] = 3
    alarm_indices = anomalies[(anomalies >= warning_threshold) & (anomalies < alarm_threshold)].index
    alarm_levels.loc[alarm_indices] = 2
    warning_indices = anomalies[anomalies < warning_threshold].index
    alarm_levels.loc[warning_indices] = 1
    return alarm_levels

# --- Autoencoder Training/Prediction Helper ---
@st.cache_resource
def train_autoencoder_model(X_train, encoding_dim=16, epochs=50, batch_size=32):
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="linear")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0, callbacks=[early_stopping])
    return autoencoder

def predict_autoencoder_anomaly(model, Xs):
    Xs_pred = model.predict(Xs, verbose=0)
    mse = np.mean(np.power(Xs - Xs_pred, 2), axis=1)
    return mse

# --- Anomaly Detection Function ---
def run_anomaly_detection(df, selected_algo, contam):
    feat_cols = [c for c in df.columns if c not in ['timestamp', 'tag', 'rpm']]
    X = df[feat_cols].fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    anomaly_scores = np.zeros(len(df))
    if selected_algo == 'IsolationForest':
        clf = IsolationForest(contamination=contam, random_state=42)
        clf.fit(Xs)
        anomaly_scores = -clf.decision_function(Xs)
    elif selected_algo == 'OneClassSVM':
        clf = OneClassSVM(nu=contam, kernel='rbf', gamma='scale')
        clf.fit(Xs)
        anomaly_scores = -clf.decision_function(Xs)
    elif selected_algo == 'Autoencoder' and TENSORFLOW_AVAILABLE:
        st.info("🧠 Training Autoencoder model... This may take a moment.")
        try:
            autoencoder = train_autoencoder_model(Xs)
            anomaly_scores = predict_autoencoder_anomaly(autoencoder, Xs)
        except Exception as e:
            st.error(f"❌ An error occurred during Autoencoder training: {e}")
            return np.zeros(len(df)), pd.Series(0, index=df.index)
    anomaly_scores_series = pd.Series(anomaly_scores, index=df.index)
    alarm_levels = apply_alarm_policy(anomaly_scores_series, contam)
    return anomaly_scores_series, alarm_levels

# --- PDF Report Generation Function (ReportLab) ---
def create_pdf_report(df, alarm_levels, selected_algo, contam, features_to_plot, fig_anom):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width/2, height - 50, "Predictive Maintenance Status Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 80, f"Machine Tag: {df['tag'].iloc[0]}")
    pdf.drawString(50, height - 95, f"Algorithm: {selected_algo} | Contamination: {contam}")
    pdf.drawString(50, height - 110, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 140, "Alarm Summary")
    pdf.setFont("Helvetica", 10)
    critical_count = (alarm_levels == 3).sum()
    alarm_count = (alarm_levels == 2).sum()
    warning_count = (alarm_levels == 1).sum()
    y_start = height - 160
    pdf.drawString(50, y_start, "Critical (3):")
    pdf.drawString(150, y_start, str(critical_count))
    pdf.drawString(50, y_start - 15, "Major Alarm (2):")
    pdf.drawString(150, y_start - 15, str(alarm_count))
    pdf.drawString(50, y_start - 30, "Warning (1):")
    pdf.drawString(150, y_start - 30, str(warning_count))
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 250, "Anomaly Detection Plot")
    buf = io.BytesIO()
    fig_anom.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    image_width = 500
    img_width, img_height = img.size
    aspect_ratio = img_height / img_width
    image_height = image_width * aspect_ratio
    max_height = 300
    if image_height > max_height:
        image_height = max_height
        image_width = image_height / aspect_ratio
    x_pos = (width - image_width) / 2
    y_pos = height - 300 - image_height
    pdf.drawInlineImage(img, x_pos, y_pos, width=image_width, height=image_height)
    pdf.showPage()
    pdf.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# --- Alarm Dashboard Display ---
def display_alarm_dashboard():
    styled_header('📊 Real-Time Alarm Dashboard')
    
    col1, col2, col3, col4 = st.columns(4)
    
    # FIX: Always get the latest alarm levels from session state
    if 'current_alarm_levels' in st.session_state and st.session_state.current_alarm_levels is not None:
        alarm_levels = st.session_state.current_alarm_levels
        critical_count = (alarm_levels == 3).sum()
        alarm_count = (alarm_levels == 2).sum()
        warning_count = (alarm_levels == 1).sum()
    else:
        critical_count = 0
        alarm_count = 0
        warning_count = 0
    
    with col1:
        futuristic_metric('🚨 Critical Alarms', critical_count, level=3)
    with col2:
        futuristic_metric('⚠️ Major Alarms', alarm_count, level=2)
    with col3:
        futuristic_metric('🔶 Warnings', warning_count, level=1)
    with col4:
        futuristic_metric('📈 Total Windows', len(window_df), level=0)
    
    return critical_count, alarm_count, warning_count

# Initialize additional session state variables
if 'current_alarm_levels' not in st.session_state:
    st.session_state.current_alarm_levels = None
if 'anomaly_detection_run' not in st.session_state:
    st.session_state.anomaly_detection_run = False
if 'anomaly_results_df' not in st.session_state:
    st.session_state.anomaly_results_df = None
if 'anomaly_plot_fig' not in st.session_state:
    st.session_state.anomaly_plot_fig = None
if 'last_anomaly_summary' not in st.session_state:
    st.session_state.last_anomaly_summary = None
if 'alarm_counts' not in st.session_state:
    st.session_state.alarm_counts = {'critical': 0, 'alarm': 0, 'warning': 0}

# --- Main Layout ---
critical_count, alarm_count, warning_count = display_alarm_dashboard()

# EDA Section
with st.container():
    st.markdown('<div class="main-sheet">', unsafe_allow_html=True)
    st.markdown('### 🔍 Exploratory Data Analysis (EDA) - Feature Trends')
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        rpm_values = sorted(window_df['rpm'].unique())
        selected_rpm = st.selectbox('🎯 Select RPM for detailed view', rpm_values, key='rpm_selector')
    
    with col_eda2:
        all_cols = window_df.columns.tolist()
        non_feature_cols = ['timestamp', 'tag', 'rpm']
        feature_cols = [col for col in all_cols if col not in non_feature_cols]
        feature_groups = {
            'RMS': [c for c in feature_cols if c.endswith('_rms')],
            'Kurtosis': [c for c in feature_cols if c.endswith('_kurtosis')],
            'Skewness': [c for c in feature_cols if c.endswith('_skew')],
            'Band Energies': [c for c in feature_cols if 'band_energy' in c],
            'Spectral Kurtosis': [c for c in feature_cols if 'spectral_kurtosis' in c],
            'Cepstrum Peak': [c for c in feature_cols if 'cepstrum_peak' in c],
            'Other': [c for c in feature_cols if not any(s in c for s in ['_rms', '_kurtosis', '_skew', 'band_energy', 'spectral_kurtosis', 'cepstrum_peak'])]
        }
        selected_group = st.selectbox('📊 Select Feature Group to Plot', list(feature_groups.keys()), key='feature_group_selector')
    
    features_to_plot = feature_groups[selected_group]
    rpm_df = window_df[window_df['rpm']==selected_rpm]
    
    if features_to_plot and not rpm_df.empty:
        st.markdown(f'**📈 {selected_group} Trends for RPM={selected_rpm}**')
        fig, ax = plt.subplots(figsize=(12, 6))
        for feature in features_to_plot:
            ax.plot(rpm_df['timestamp'], rpm_df[feature], label=feature, linewidth=2)
        ax.set_title(f'{selected_group} Trends', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestamp', fontweight='bold')
        ax.set_ylabel('Feature Value', fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=20)
        st.pyplot(fig)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Anomaly Detection Section
with st.container():
    st.markdown('<div class="main-sheet">', unsafe_allow_html=True)
    st.markdown('### 🤖 AI Anomaly Detection & Alarm Generation')
    
    col_alg1, col_alg2 = st.columns(2)
    
    with col_alg1:
        algorithms = ['IsolationForest', 'OneClassSVM']
        if TENSORFLOW_AVAILABLE:
            algorithms.append('Autoencoder')
        selected_algo = st.selectbox('🧠 Select AI Algorithm', algorithms, key='algo_selector')
    
    with col_alg2:
        contam = st.slider('🎚️ Contamination Level', 0.001, 0.2, 0.02, key='contam_slider',
                          help="Estimated fraction of anomalies in the data")

    # FIX: Use a separate flag to track if we need to run detection
    if 'run_detection' not in st.session_state:
        st.session_state.run_detection = False

    if st.button('🚀 Run AI Anomaly Detection', use_container_width=True, key='run_detection_btn'):
        st.session_state.run_detection = True

    # Run detection only when the flag is set
    if st.session_state.run_detection:
        st.session_state.anomaly_detection_run = True
        st.session_state.run_detection = False  # Reset flag
        
        with st.spinner('🔄 AI is analyzing your data... This may take a moment.'):
            anomaly_scores, alarm_levels = run_anomaly_detection(window_df, selected_algo, contam)
        
        if alarm_levels is not None and not window_df.empty:
            results_df = window_df.copy()
            results_df['anomaly_score'] = anomaly_scores
            results_df['alarm_level'] = alarm_levels
            
            # Update session state with new alarm levels
            st.session_state.current_alarm_levels = alarm_levels
            st.session_state.anomaly_results_df = results_df
            
            critical_count = (alarm_levels == 3).sum()
            alarm_count = (alarm_levels == 2).sum()
            warning_count = (alarm_levels == 1).sum()
            
            st.session_state.alarm_counts = {
                'critical': critical_count,
                'alarm': alarm_count,
                'warning': warning_count
            }
            
            st.session_state.last_anomaly_summary = {
                'algorithm': selected_algo,
                'contamination': contam,
                'critical_count': critical_count,
                'alarm_count': alarm_count,
                'warning_count': warning_count
            }
            
            # Create visualization plot
            fig, ax = plt.subplots(figsize=(14, 8))
            colors = {0: 'green', 1: 'orange', 2: 'red', 3: 'purple'}
            labels = {0: 'Normal', 1: 'Warning', 2: 'Alarm', 3: 'Critical'}
            
            for level in [0, 1, 2, 3]:
                subset = results_df[results_df['alarm_level'] == level]
                if not subset.empty:
                    ax.scatter(subset['timestamp'], subset['anomaly_score'], 
                              c=colors[level], label=labels[level], s=40, alpha=0.7)
            
            ax.set_title(f'Anomaly Detection Results ({selected_algo})', fontsize=16, fontweight='bold')
            ax.set_xlabel('Timestamp', fontweight='bold')
            ax.set_ylabel('Anomaly Score', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=20)
            st.session_state.anomaly_plot_fig = fig
            
            st.success(f'✅ Anomaly detection completed! Found: {critical_count} Critical, {alarm_count} Major Alarms, {warning_count} Warnings')
            
            # FIX: Force refresh to update the dashboard
            st.rerun()
        else:
            st.error("❌ Anomaly detection failed or produced no results.")
    
    # Display results section - Show this section independently
    if st.session_state.anomaly_detection_run and st.session_state.anomaly_results_df is not None:
        st.markdown('---')
        st.markdown('### 📋 Anomaly Detection Results')
        results_df = st.session_state.anomaly_results_df
        
        # Display summary info with GREEN COLOR
        if st.session_state.last_anomaly_summary:
            summary = st.session_state.last_anomaly_summary
            st.markdown(f"""
            <div class="green-summary">
            <strong>Detection Summary:</strong><br/>
            • <strong>Algorithm:</strong> {summary['algorithm']}<br/>
            • <strong>Contamination:</strong> {summary['contamination']}<br/>
            • <strong>Critical Alarms:</strong> {summary['critical_count']}<br/>
            • <strong>Major Alarms:</strong> {summary['alarm_count']}<br/>
            • <strong>Warnings:</strong> {summary['warning_count']}<br/>
            • <strong>Total Anomalies:</strong> {summary['critical_count'] + summary['alarm_count'] + summary['warning_count']}
            </div>
            """, unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown('**🔍 Detailed Results Table**')
            display_cols = ['timestamp', 'rpm', 'anomaly_score', 'alarm_level']
            st.dataframe(results_df[display_cols].sort_values('anomaly_score', ascending=False), use_container_width=True)
        
        with col_res2:
            st.markdown('**📊 Alarm Level Distribution**')
            alarm_counts = results_df['alarm_level'].value_counts().sort_index()
            alarm_labels = {0: 'Normal', 1: 'Warning', 2: 'Alarm', 3: 'Critical'}
            alarm_counts.index = [alarm_labels.get(i, i) for i in alarm_counts.index]
            st.bar_chart(alarm_counts)
        
        # Display the visualization plot
        if st.session_state.anomaly_plot_fig is not None:
            st.markdown('**📈 Anomaly Detection Visualization**')
            st.pyplot(st.session_state.anomaly_plot_fig)
        
        # Generate Report Section
        st.markdown('---')
        st.markdown('### 📄 Generate PDF Report')
        
        if st.button('📥 Download Anomaly Detection Report', use_container_width=True, key='download_report_btn'):
            if st.session_state.anomaly_plot_fig is not None:
                pdf_bytes = create_pdf_report(
                    results_df, 
                    results_df['alarm_level'], 
                    selected_algo, 
                    contam, 
                    features_to_plot, 
                    st.session_state.anomaly_plot_fig
                )
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"PdM_Report_{tag}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("📄 PDF report generated successfully!")
            else:
                st.warning("⚠️ Please run anomaly detection first to generate a report.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 14px;">
    🚀 <strong>Predictive Maintenance Futuristic Dashboard</strong> | 
    Advanced AI-Powered Anomaly Detection System
</div>
""", unsafe_allow_html=True)