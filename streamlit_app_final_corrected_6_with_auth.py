# streamlit_app_final_corrected_6_with_auth.py
# PdM Prototype Dashboard with simple user management (JSON + SHA256 passwords)
# - Adds login page and admin-only user management (add / delete / change passwords)
# - Users stored in users_db.json (SHA256 hashed passwords)
# - Default admin: admin / 1234 (created automatically if missing)
# Usage: streamlit run streamlit_app_final_corrected_6_with_auth.py

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

# -----------------------------
# Streamlit app starts here
# -----------------------------
st.set_page_config(layout='wide', page_title='PdM Prototype Dashboard (Auth)')

# Session state keys for auth
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'current_role' not in st.session_state:
    st.session_state.current_role = None

def login_widget():
    st.title("PdM Prototype - Login")
    st.write("Please log in to continue.")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        ok, role_or_msg = verify_credentials(username.strip(), password)
        if ok:
            st.session_state.logged_in = True
            st.session_state.current_user = username.strip()
            st.session_state.current_role = role_or_msg
            st.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.error(role_or_msg)

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
st.sidebar.write(f"Logged in as: **{st.session_state.current_user}** ({st.session_state.current_role})")
if st.sidebar.button("Log out"):
    logout()

# Admin-only user management UI
def user_management_ui():
    st.header("User Management (Admin only)")
    users = load_users_db()
    st.write("Existing users:")
    user_df = pd.DataFrame([{"username": k, "role": v.get("role", "viewer")} for k,v in users.items()])
    st.dataframe(user_df, use_container_width=True)
    st.markdown("----")
    st.subheader("Add New User (requires Admin confirmation)")
    with st.form("add_user_form"):
        admin_pw = st.text_input("Admin password (confirm)", type="password")
        new_username = st.text_input("New username")
        new_password = st.text_input("New password", type="password")
        role = st.selectbox("Role", ["viewer", "engineer", "admin"], index=0)
        add_sub = st.form_submit_button("Add user")
    if add_sub:
        ok, msg = add_user(admin_pw, new_username.strip(), new_password, role)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.subheader("Delete User (requires Admin confirmation)")
    with st.form("del_user_form"):
        admin_pw2 = st.text_input("Admin password (confirm) for delete", type="password", key="del_confirm")
        users = load_users_db()
        del_username = st.selectbox("Select user to delete", list(users.keys()), key="del_username")

        del_sub = st.form_submit_button("Delete user")
    if del_sub:
        ok, msg = delete_user(admin_pw2, del_username.strip())
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.subheader("Change any user's password (requires Admin confirmation)")
    with st.form("change_pw_form"):
        admin_pw3 = st.text_input("Admin password (confirm) for change", type="password", key="chg_confirm")
        users = load_users_db()
        target_user = st.selectbox("Select user to change password", list(users.keys()), key="chg_user")

        new_pw = st.text_input("New password", type="password", key="chg_newpw")
        chg_sub = st.form_submit_button("Change password")
    if chg_sub:
        ok, msg = change_user_password(admin_pw3, target_user.strip(), new_pw)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("----")
    st.subheader("Change your own password")
    with st.form("own_pw_form"):
        cur_pw = st.text_input("Current password", type="password", key="own_cur")
        new_pw2 = st.text_input("New password", type="password", key="own_new")
        own_sub = st.form_submit_button("Update my password")
    if own_sub:
        username = st.session_state.current_user
        ok, msg = change_own_password(username, cur_pw, new_pw2)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

# If current user is admin, show user management in sidebar expander
if st.session_state.current_role == "admin":
    with st.sidebar.expander("Admin: User Management", expanded=False):
        user_management_ui()

# -----------------------------
# The rest of the original app code follows below (data loading, EDA, anomaly detection...)
# For brevity, this file re-uses the same functions but ensures they are executed only after login.
# -----------------------------

FEATURES_CSV = "stranding_features_advanced.csv"

@st.cache_data
def load_features():
    try:
        df = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'], encoding='utf-8')
        return df
    except FileNotFoundError:
        st.error(f"Error: Feature file '{FEATURES_CSV}' not found. Please run feature_extraction.py first.")
        return pd.DataFrame()
    except UnicodeDecodeError:
        # Try fallback encoding
        df = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'], encoding='latin1')
        return df

df = load_features()
if df.empty:
    st.stop()

# --- Sidebar Controls (Time Window Selection and Machine Tag) ---
tags = df['tag'].unique().tolist()
tag = st.sidebar.selectbox('Machine Tag', tags)
window_df = df[df['tag']==tag].reset_index(drop=True)

# Time Window Selection
min_date = window_df['timestamp'].min().date()
max_date = window_df['timestamp'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) # Include the end day
    window_df = window_df[(window_df['timestamp'] >= start_date) & (window_df['timestamp'] < end_date)].reset_index(drop=True)

st.sidebar.markdown(f'Windows in Range: {len(window_df)}')

# --- Alarm Policy Implementation (Simplified for Demonstration) ---
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
        st.info("Training Autoencoder model... This may take a moment.")
        try:
            autoencoder = train_autoencoder_model(Xs)
            anomaly_scores = predict_autoencoder_anomaly(autoencoder, Xs)
        except Exception as e:
            st.error(f"An error occurred during Autoencoder training: {e}")
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
    st.subheader('Alarm Dashboard')
    col_alarm1, col_alarm2, col_alarm3, col_alarm4 = st.columns(4)
    if 'current_alarm_levels' in st.session_state and st.session_state.current_alarm_levels is not None:
        alarm_levels = st.session_state.current_alarm_levels
        critical_count = (alarm_levels == 3).sum()
        alarm_count = (alarm_levels == 2).sum()
        warning_count = (alarm_levels == 1).sum()
    else:
        critical_count = 0
        alarm_count = 0
        warning_count = 0
    def display_alarm_metric(col, label, count, level):
        if level == 3: color = 'red'
        elif level == 2: color = 'orange'
        elif level == 1: color = 'yellow'
        else: color = 'green'
        col.metric(label=label, value=count, delta_color="off")
        col.markdown(f'<div style="background-color:{color}; padding: 5px; border-radius: 5px; color: black; text-align: center;">{label}</div>', unsafe_allow_html=True)
    display_alarm_metric(col_alarm1, 'Critical Alarms', critical_count, 3)
    display_alarm_metric(col_alarm2, 'Major Alarms', alarm_count, 2)
    display_alarm_metric(col_alarm3, 'Warnings', warning_count, 1)
    col_alarm4.metric(label='Total Windows', value=len(window_df))
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
st.header(f'Overview for {tag}')
critical_count, alarm_count, warning_count = display_alarm_dashboard()
st.markdown('---')

# EDA
st.subheader('Exploratory Data Analysis (EDA) - Feature Trends per RPM')
rpm_values = sorted(window_df['rpm'].unique())
selected_rpm = st.selectbox('Select RPM for detailed view', rpm_values, key='rpm_selector')
rpm_df = window_df[window_df['rpm']==selected_rpm]
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
selected_group = st.selectbox('Select Feature Group to Plot', list(feature_groups.keys()), key='feature_group_selector')
features_to_plot = feature_groups[selected_group]
if features_to_plot and not rpm_df.empty:
    st.markdown(f'**{selected_group} Trends for RPM={selected_rpm}**')
    fig, ax = plt.subplots(figsize=(12, 6))
    for feature in features_to_plot:
        ax.plot(rpm_df['timestamp'], rpm_df[feature], label=feature)
    ax.set_title(f'{selected_group} Trends')
    ax.set_xlabel('Timestamp')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    plt.xticks(rotation=20)
    st.pyplot(fig)

st.markdown('---')
# Anomaly Detection
st.subheader('Anomaly Detection & Alarm Generation')
algorithms = ['IsolationForest', 'OneClassSVM']
if TENSORFLOW_AVAILABLE:
    algorithms.append('Autoencoder')
selected_algo = st.selectbox('Select Anomaly Detection Algorithm', algorithms, key='algo_selector')
contam = st.slider('Contamination (est. fraction of anomalies)', 0.001, 0.2, 0.02, key='contam_slider')

if st.button(f'Run Anomaly Detection & Generate Alarms', key='run_detection_btn'):
    st.session_state.anomaly_detection_run = True
    st.info(f'Running {selected_algo} on selected data...')
    anomaly_scores, alarm_levels = run_anomaly_detection(window_df, selected_algo, contam)
    if alarm_levels is not None and not window_df.empty:
        results_df = window_df.copy()
        results_df['anomaly_score'] = anomaly_scores
        results_df['alarm_level'] = alarm_levels
        st.session_state.anomaly_results_df = results_df
        st.session_state.current_alarm_levels = alarm_levels
        critical_count = (alarm_levels == 3).sum()
        alarm_count = (alarm_levels == 2).sum()
        warning_count = (alarm_levels == 1).sum()
        st.session_state.alarm_counts = {'critical': critical_count, 'alarm': alarm_count, 'warning': warning_count}
        st.session_state.last_anomaly_summary = {'selected_algo': selected_algo, 'critical_count': critical_count, 'alarm_count': alarm_count, 'warning_count': warning_count, 'total_alarms': alarm_levels[alarm_levels > 0].count()}
        if features_to_plot:
            fig_anom, ax_anom = plt.subplots(figsize=(12, 4))
            ax_anom.plot(results_df['timestamp'], results_df[features_to_plot[0]], label=features_to_plot[0], color='blue', alpha=0.7)
            critical_data = results_df[results_df['alarm_level'] == 3]
            alarm_data = results_df[results_df['alarm_level'] == 2]
            warning_data = results_df[results_df['alarm_level'] == 1]
            if not critical_data.empty:
                ax_anom.scatter(critical_data['timestamp'], critical_data[features_to_plot[0]], color='red', label=f'Critical ({critical_count})', marker='o', s=100, zorder=5)
            if not alarm_data.empty:
                ax_anom.scatter(alarm_data['timestamp'], alarm_data[features_to_plot[0]], color='orange', label=f'Alarm ({alarm_count})', marker='o', s=70, zorder=5)
            if not warning_data.empty:
                ax_anom.scatter(warning_data['timestamp'], warning_data[features_to_plot[0]], color='yellow', label=f'Warning ({warning_count})', marker='o', s=40, zorder=5)
            ax_anom.set_title(f'Alarm Levels on {features_to_plot[0]} using {selected_algo}')
            ax_anom.set_xlabel('Timestamp')
            ax_anom.set_ylabel(features_to_plot[0])
            ax_anom.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax_anom.grid(True)
            plt.xticks(rotation=20)
            st.session_state.anomaly_plot_fig = fig_anom
            st.session_state['last_anomaly_fig'] = fig_anom
            st.session_state['last_alarm_levels'] = alarm_levels
            st.session_state['last_selected_algo'] = selected_algo
            st.session_state['last_contam'] = contam
        st.success(f'Anomaly detection completed successfully!')
        st.rerun()
    else:
        st.error("Anomaly detection failed. Check the logs.")

if st.session_state.anomaly_detection_run and st.session_state.anomaly_results_df is not None:
    st.markdown("---")
    st.subheader("Anomaly Detection Results")
    results_df = st.session_state.anomaly_results_df
    summary = st.session_state.last_anomaly_summary
    alarm_counts = st.session_state.alarm_counts
    st.info(f"Algorithm: {summary['selected_algo']} | Total Alarms: {summary['total_alarms']} | Critical: {alarm_counts['critical']} | Major: {alarm_counts['alarm']} | Warning: {alarm_counts['warning']}")
    st.write("### Anomaly Results Table")
    display_columns = ['timestamp', 'rpm', 'alarm_level', 'anomaly_score'] + [c for c in results_df.columns if c.endswith('_rms')]
    st.dataframe(results_df[display_columns].head(200))
    if st.session_state.anomaly_plot_fig is not None:
        st.write("### Anomaly Visualization")
        st.pyplot(st.session_state.anomaly_plot_fig)

st.markdown('---')
# Reporting and logs
st.subheader('Reporting and Maintenance Logs')
col_rep, col_log = st.columns(2)
with col_rep:
    if st.button('Generate PDF Status Report', key='pdf_btn'):
        if 'last_anomaly_fig' in st.session_state and 'last_alarm_levels' in st.session_state:
            pdf_bytes = create_pdf_report(window_df, st.session_state['last_alarm_levels'], st.session_state['last_selected_algo'], st.session_state['last_contam'], features_to_plot, st.session_state['last_anomaly_fig'])
            st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="pdm_status_report.pdf", mime="application/pdf")
            st.success("PDF Report generated and ready for download.")
        else:
            st.warning("Please run Anomaly Detection first to generate the necessary data for the report.")
with col_log:
    st.markdown('**Maintenance Logs (Placeholder)**')
    st.text_area("Recent Maintenance Events", "2025-10-28 10:00:00: Bearing Inboard lubricated (No issues).\n2025-10-27 15:30:00: Vibration sensor recalibrated.", height=150, key='maintenance_logs')
    st.caption('This section will be linked to a real maintenance log database in the future.')
