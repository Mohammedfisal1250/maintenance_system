# streamlit_app.py - PdM Prototype Dashboard (Advanced)
# Run: streamlit run streamlit_app.py
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
# Import TensorFlow/Keras for Autoencoder (ensure it's installed)
try:
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import for PDF generation (Using ReportLab)
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
import io
from PIL import Image # Added for ReportLab image handling

# --- Configuration ---
FEATURES_CSV = "stranding_features_advanced.csv" # Use the advanced features file
PDF_REPORT_PLACEHOLDER = "report.pdf" # Placeholder for the generated report

st.set_page_config(layout='wide', page_title='PdM Prototype Dashboard')
st.title('PdM Prototype - Advanced Dashboard')

# --- Data Loading ---
@st.cache_data
def load_features():
    try:
        df = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'])
        return df
    except FileNotFoundError:
        st.error(f"Error: Feature file '{FEATURES_CSV}' not found. Please run feature_extraction.py first.")
        return pd.DataFrame()

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
    """
    Applies a simplified 3-level alarm policy based on anomaly scores.
    - Critical: Top 1/3 of the anomaly scores above the contamination threshold.
    - Alarm: Middle 1/3 of the anomaly scores above the contamination threshold.
    - Warning: Bottom 1/3 of the anomaly scores above the contamination threshold.
    """
    if anomaly_scores_series.empty:
        return pd.Series(0, index=anomaly_scores_series.index)

    # 1. Identify anomalies based on the contamination threshold
    # The anomaly threshold is calculated using the quantile of the anomaly scores Series
    # The contamination is the *fraction* of outliers, so we use (1 - contamination) for the threshold
    anomaly_threshold = anomaly_scores_series.quantile(1 - contamination)
    anomalies = anomaly_scores_series[anomaly_scores_series > anomaly_threshold]
    
    # 2. Assign levels based on score distribution among anomalies
    if anomalies.empty:
        return pd.Series(0, index=anomaly_scores_series.index)

    # Calculate percentiles for the anomalies only
    warning_threshold = anomalies.quantile(0.33)
    alarm_threshold = anomalies.quantile(0.66)

    alarm_levels = pd.Series(0, index=anomaly_scores_series.index)
    
    # Critical (Level 3)
    critical_indices = anomalies[anomalies >= alarm_threshold].index
    alarm_levels.loc[critical_indices] = 3 
    
    # Alarm (Level 2)
    alarm_indices = anomalies[(anomalies >= warning_threshold) & (anomalies < alarm_threshold)].index
    alarm_levels.loc[alarm_indices] = 2
    
    # Warning (Level 1)
    warning_indices = anomalies[anomalies < warning_threshold].index
    alarm_levels.loc[warning_indices] = 1
    
    return alarm_levels

# --- Autoencoder Training/Prediction Helper ---
@st.cache_resource
def train_autoencoder_model(X_train, encoding_dim=16, epochs=50, batch_size=32):
    input_dim = X_train.shape[1]

    # Define the Autoencoder model
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="linear")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile and train
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Early stopping to prevent overfitting and save time
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    # Train the model (using all data, as Autoencoders are trained on 'normal' data)
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        callbacks=[early_stopping]
    )
    return autoencoder

def predict_autoencoder_anomaly(model, Xs):
    # Predict the reconstruction
    Xs_pred = model.predict(Xs, verbose=0)
    # Calculate the reconstruction error (MSE)
    mse = np.mean(np.power(Xs - Xs_pred, 2), axis=1)
    return mse

# --- Anomaly Detection Function ---
def run_anomaly_detection(df, selected_algo, contam):
    # Select all feature columns (excluding timestamp, tag, rpm)
    feat_cols = [c for c in df.columns if c not in ['timestamp', 'tag', 'rpm']]
    X = df[feat_cols].fillna(0.0)
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    # Initialize anomaly_scores as a numpy array, which is the output of decision_function
    anomaly_scores = np.zeros(len(df))
    
    if selected_algo == 'IsolationForest':
        clf = IsolationForest(contamination=contam, random_state=42)
        clf.fit(Xs)
        # Isolation Forest score is the opposite: lower is more anomalous. We invert it.
        anomaly_scores = -clf.decision_function(Xs)
        
    elif selected_algo == 'OneClassSVM':
        clf = OneClassSVM(nu=contam, kernel='rbf', gamma='scale')
        clf.fit(Xs)
        # OCSVM score is the opposite: lower is more anomalous. We invert it.
        anomaly_scores = -clf.decision_function(Xs)
        
    elif selected_algo == 'Autoencoder' and TENSORFLOW_AVAILABLE:
        st.info("Training Autoencoder model... This may take a moment.")
        try:
            # Train the model on the scaled data
            autoencoder = train_autoencoder_model(Xs)
            # Get the anomaly scores (reconstruction error)
            anomaly_scores = predict_autoencoder_anomaly(autoencoder, Xs)
            
        except Exception as e:
            st.error(f"An error occurred during Autoencoder training: {e}")
            return np.zeros(len(df)), pd.Series(0, index=df.index)

    # Convert to Series for easier indexing and use in apply_alarm_policy
    anomaly_scores_series = pd.Series(anomaly_scores, index=df.index)

    # Apply the alarm policy to the anomaly scores
    alarm_levels = apply_alarm_policy(anomaly_scores_series, contam)
    
    return anomaly_scores_series, alarm_levels

# --- PDF Report Generation Function (ReportLab) ---

def create_pdf_report(df, alarm_levels, selected_algo, contam, features_to_plot, fig_anom):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter # 612 x 792 points
    
    # --- Title ---
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(width/2, height - 50, "Predictive Maintenance Status Report")
    
    # --- Metadata ---
    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 80, f"Machine Tag: {df['tag'].iloc[0]}")
    pdf.drawString(50, height - 95, f"Algorithm: {selected_algo} | Contamination: {contam}")
    pdf.drawString(50, height - 110, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- Summary Table ---
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 140, "Alarm Summary")
    
    pdf.setFont("Helvetica", 10)
    critical_count = (alarm_levels == 3).sum()
    alarm_count = (alarm_levels == 2).sum()
    warning_count = (alarm_levels == 1).sum()
    
    y_start = height - 160
    
    # Draw table-like structure
    pdf.drawString(50, y_start, "Critical (3):")
    pdf.drawString(150, y_start, str(critical_count))
    
    pdf.drawString(50, y_start - 15, "Major Alarm (2):")
    pdf.drawString(150, y_start - 15, str(alarm_count))
    
    pdf.drawString(50, y_start - 30, "Warning (1):")
    pdf.drawString(150, y_start - 30, str(warning_count))
    
    # --- Anomaly Plot ---
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 250, "Anomaly Detection Plot")
    
    # Save the matplotlib figure to a buffer
    buf = io.BytesIO()
    # استخدام bbox_inches='tight' لمنع تقطيع الصورة
    fig_anom.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    
    # Load the image from the buffer using PIL (Pillow)
    img = Image.open(buf)
    
    # حساب أبعاد الصورة المناسبة لمنع التقاطع
    image_width = 500
    # الحفاظ على نسبة الطول إلى العرض
    img_width, img_height = img.size
    aspect_ratio = img_height / img_width
    image_height = image_width * aspect_ratio
    
    # التأكد من أن الصورة لا تتجاوز المساحة المتاحة
    max_height = 300
    if image_height > max_height:
        image_height = max_height
        image_width = image_height / aspect_ratio
    
    # وضع الصورة في منتصف الصفحة
    x_pos = (width - image_width) / 2
    y_pos = height - 300 - image_height
    
    # Pass the PIL Image object to drawInlineImage
    pdf.drawInlineImage(img, x_pos, y_pos, width=image_width, height=image_height)
    
    # Finish the PDF writing
    pdf.showPage()
    pdf.save()
    
    # Get the value of the BytesIO buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# --- Function to display Alarm Dashboard ---
def display_alarm_dashboard():
    st.subheader('Alarm Dashboard')
    col_alarm1, col_alarm2, col_alarm3, col_alarm4 = st.columns(4)
    
    # استخدام الإنذارات المخزنة في session state
    if 'current_alarm_levels' in st.session_state and st.session_state.current_alarm_levels is not None:
        alarm_levels = st.session_state.current_alarm_levels
        critical_count = (alarm_levels == 3).sum()
        alarm_count = (alarm_levels == 2).sum()
        warning_count = (alarm_levels == 1).sum()
    else:
        # التشغيل الأولي إذا لم تكن هناك بيانات
        critical_count = 0
        alarm_count = 0
        warning_count = 0
    
    def display_alarm_metric(col, label, count, level):
        if level == 3: color = 'red'
        elif level == 2: color = 'orange'
        elif level == 1: color = 'yellow'
        else: color = 'green'
        
        col.metric(
            label=label, 
            value=count, 
            delta_color="off"
        )
        # Simple color coding using markdown for demonstration
        col.markdown(f'<div style="background-color:{color}; padding: 5px; border-radius: 5px; color: black; text-align: center;">{label}</div>', unsafe_allow_html=True)

    display_alarm_metric(col_alarm1, 'Critical Alarms', critical_count, 3)
    display_alarm_metric(col_alarm2, 'Major Alarms', alarm_count, 2)
    display_alarm_metric(col_alarm3, 'Warnings', warning_count, 1)
    col_alarm4.metric(label='Total Windows', value=len(window_df))
    
    return critical_count, alarm_count, warning_count

# --- Initialize Session State ---
if 'current_alarm_levels' not in st.session_state:
    st.session_state.current_alarm_levels = None
if 'anomaly_detection_run' not in st.session_state:
    st.session_state.anomaly_detection_run = False
# إضافة هذه المتغيرات الجديدة
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

# --- Alarm Dashboard Panel (يتم تحديثه تلقائياً) ---
critical_count, alarm_count, warning_count = display_alarm_dashboard()

st.markdown('---')

# --- EDA Section (Plotting for each RPM) ---
st.subheader('Exploratory Data Analysis (EDA) - Feature Trends per RPM')

rpm_values = sorted(window_df['rpm'].unique())
selected_rpm = st.selectbox('Select RPM for detailed view', rpm_values, key='rpm_selector')

rpm_df = window_df[window_df['rpm']==selected_rpm]

# Feature Selection for Plotting
all_cols = window_df.columns.tolist()
non_feature_cols = ['timestamp', 'tag', 'rpm']
feature_cols = [col for col in all_cols if col not in non_feature_cols]

# Group features by type for better visualization
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

# --- Anomaly Detection Section ---
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
        # حفظ النتائج في session state
        results_df = window_df.copy()
        results_df['anomaly_score'] = anomaly_scores
        results_df['alarm_level'] = alarm_levels
        
        st.session_state.anomaly_results_df = results_df
        st.session_state.current_alarm_levels = alarm_levels
        
        # حساب الإحصائيات مرة واحدة فقط باستخدام نفس البيانات
        critical_count = (alarm_levels == 3).sum()
        alarm_count = (alarm_levels == 2).sum()
        warning_count = (alarm_levels == 1).sum()
        
        # حفظ الإحصائيات في session state
        st.session_state.alarm_counts = {
            'critical': critical_count,
            'alarm': alarm_count, 
            'warning': warning_count
        }
        
        st.session_state.last_anomaly_summary = {
            'selected_algo': selected_algo,
            'critical_count': critical_count,
            'alarm_count': alarm_count,
            'warning_count': warning_count,
            'total_alarms': alarm_levels[alarm_levels > 0].count()
        }
        
        # إنشاء الرسمة وحفظها - استخدام نفس الإحصائيات المخزنة
        if features_to_plot:
            fig_anom, ax_anom = plt.subplots(figsize=(12, 4))
            
            # Base plot
            ax_anom.plot(results_df['timestamp'], results_df[features_to_plot[0]], 
                        label=features_to_plot[0], color='blue', alpha=0.7)
            
            # استخدام نفس البيانات المخزنة للإحصائيات
            critical_data = results_df[results_df['alarm_level'] == 3]
            alarm_data = results_df[results_df['alarm_level'] == 2]
            warning_data = results_df[results_df['alarm_level'] == 1]
            
            # Scatter plot for alarms - استخدام نفس الإحصائيات
            if not critical_data.empty:
                ax_anom.scatter(critical_data['timestamp'], critical_data[features_to_plot[0]], 
                              color='red', label=f'Critical ({critical_count})', marker='o', s=100, zorder=5)
            
            if not alarm_data.empty:
                ax_anom.scatter(alarm_data['timestamp'], alarm_data[features_to_plot[0]], 
                              color='orange', label=f'Alarm ({alarm_count})', marker='o', s=70, zorder=5)
            
            if not warning_data.empty:
                ax_anom.scatter(warning_data['timestamp'], warning_data[features_to_plot[0]], 
                              color='yellow', label=f'Warning ({warning_count})', marker='o', s=40, zorder=5)
            
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
        st.rerun()  # إعادة تحميل الصفحة لتحديث Alarm Dashboard
        
    else:
        st.error("Anomaly detection failed. Check the logs.")

# عرض النتائج المخزنة إذا كانت موجودة
if st.session_state.anomaly_detection_run and st.session_state.anomaly_results_df is not None:
    st.markdown("---")
    st.subheader("Anomaly Detection Results")
    
    results_df = st.session_state.anomaly_results_df
    summary = st.session_state.last_anomaly_summary
    
    # استخدام نفس الإحصائيات المخزنة في session state
    alarm_counts = st.session_state.alarm_counts
    
    # عرض الإحصائيات - استخدام نفس البيانات
    st.info(
        f"Algorithm: {summary['selected_algo']} | "
        f"Total Alarms: {summary['total_alarms']} | "
        f"Critical: {alarm_counts['critical']} | "
        f"Major: {alarm_counts['alarm']} | "
        f"Warning: {alarm_counts['warning']}"
    )
    
    # عرض الجدول
    st.write("### Anomaly Results Table")
    display_columns = ['timestamp', 'rpm', 'alarm_level', 'anomaly_score'] + \
                     [c for c in results_df.columns if c.endswith('_rms')]
    st.dataframe(results_df[display_columns].head(200))
    
    # عرض الرسمة
    if st.session_state.anomaly_plot_fig is not None:
        st.write("### Anomaly Visualization")
        st.pyplot(st.session_state.anomaly_plot_fig)

st.markdown('---')

# --- Report and Logs Section ---
st.subheader('Reporting and Maintenance Logs')

col_rep, col_log = st.columns(2)

with col_rep:
    # PDF Report Generation
    if st.button('Generate PDF Status Report', key='pdf_btn'):
        if 'last_anomaly_fig' in st.session_state and 'last_alarm_levels' in st.session_state:
            pdf_bytes = create_pdf_report(
                window_df, 
                st.session_state['last_alarm_levels'], 
                st.session_state['last_selected_algo'], 
                st.session_state['last_contam'],
                features_to_plot,
                st.session_state['last_anomaly_fig']
            )
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="pdm_status_report.pdf",
                mime="application/pdf"
            )
            st.success("PDF Report generated and ready for download.")
        else:
            st.warning("Please run Anomaly Detection first to generate the necessary data for the report.")

with col_log:
    # Placeholder for Maintenance Logs
    st.markdown('**Maintenance Logs (Placeholder)**')
    st.text_area(
        "Recent Maintenance Events",
        "2025-10-28 10:00:00: Bearing Inboard lubricated (No issues).\n"
        "2025-10-27 15:30:00: Vibration sensor recalibrated.",
        height=150,
        key='maintenance_logs'
    )
    st.caption('This section will be linked to a real maintenance log database in the future.')