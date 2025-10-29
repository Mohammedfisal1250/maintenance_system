"""train_models.py - modular trainer for PdM prototype
Usage:
  python train_models.py --algo iforest --out model.joblib
Dependencies: scikit-learn, pandas, joblib, tensorflow (for autoencoder)

# NOTE: For future work, consider adding a data ingestion module here to load
# real-time data from an API or database instead of a static CSV file.
# This would replace the 'load_features' function for production use.
"""
import argparse, joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
# Import TensorFlow/Keras for Autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Use the advanced features file now
FEATURES_CSV = "stranding_features_advanced.csv" 

def load_features():
    df = pd.read_csv(FEATURES_CSV, parse_dates=['timestamp'])
    # Select all feature columns (excluding timestamp, tag, rpm)
    cols = [c for c in df.columns if c not in ['timestamp', 'tag', 'rpm']]
    X = df[cols].fillna(0.0)
    return df, X, cols

def train_iforest(X, outpath, contamination=0.02):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(Xs)
    joblib.dump({'model': clf, 'scaler': scaler, 'features': list(X.columns)}, outpath)
    print('Saved IsolationForest to', outpath)

def train_ocsvm(X, outpath, nu=0.02):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    clf = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
    clf.fit(Xs)
    joblib.dump({'model': clf, 'scaler': scaler, 'features': list(X.columns)}, outpath)
    print('Saved OneClassSVM to', outpath)

def train_autoencoder(X, outpath, encoding_dim=16, epochs=50, batch_size=32):
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    input_dim = Xs.shape[1]

    # Define the Autoencoder model
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="linear")(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile and train
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Early stopping to prevent overfitting and save time
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model (using a simple train/validation split for demonstration)
    # In a real scenario, we would use only 'normal' data for training
    history = autoencoder.fit(
        Xs, Xs,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        verbose=0, # Set to 1 or 2 for progress updates
        callbacks=[early_stopping]
    )
    
    # Save the model and scaler
    # Keras models are saved differently than joblib models
    autoencoder.save(outpath + '_model.h5')
    joblib.dump({'scaler': scaler, 'features': list(X.columns)}, outpath + '_meta.joblib')
    print('Saved Autoencoder model and metadata to', outpath)

def determine_anomaly_threshold(model, Xs, contamination):
    """
    Helper function to determine the reconstruction error threshold for Autoencoder.
    This is a simplified approach: use the (1-contamination) percentile of the
    reconstruction errors on the training data as the threshold.
    """
    # Predict the reconstruction
    Xs_pred = model.predict(Xs, verbose=0)
    # Calculate the reconstruction error (MSE)
    mse = np.mean(np.power(Xs - Xs_pred, 2), axis=1)
    # Determine the threshold
    threshold = np.percentile(mse, (1 - contamination) * 100)
    return threshold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['iforest','ocsvm','autoencoder'], default='iforest')
    parser.add_argument('--out', default='model.joblib')
    # For Autoencoder, 'contam' will be used to set the anomaly threshold
    parser.add_argument('--contam', type=float, default=0.02) 
    args = parser.parse_args()
    
    # Check if the advanced features file exists and create a dummy if not (for testing)
    try:
        df, X, cols = load_features()
    except FileNotFoundError:
        print(f"Error: Feature file '{FEATURES_CSV}' not found. Please run feature_extraction.py first.")
        exit()

    if args.algo == 'iforest':
        train_iforest(X, args.out, contamination=args.contam)
    elif args.algo == 'ocsvm':
        train_ocsvm(X, args.out, nu=args.contam)
    elif args.algo == 'autoencoder':
        # Autoencoder training will save two files: model.h5 and model_meta.joblib
        train_autoencoder(X, args.out)
