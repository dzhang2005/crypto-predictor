# final_improved_crypto_training.py
"""
FINAL IMPROVED VERSION - Crypto Predictor with Key Enhancements
This is the complete, ready-to-use version with the most impactful improvements
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import warnings
import logging
import time
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ============================================================================
# IMPROVEMENT 1: BETTER LOSS FUNCTION (Focal Loss for imbalanced data)
# ============================================================================

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss - better than BCE for imbalanced crypto data
    Focuses learning on hard examples
    """
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = (1 - pt) ** gamma
    
    if alpha is not None:
        alpha_t = torch.where(target == 1, alpha, 1 - alpha)
        focal_weight = alpha_t * focal_weight
    
    return (focal_weight * bce).mean()

# ============================================================================
# IMPROVEMENT 2: ENHANCED MODEL WITH BETTER ARCHITECTURE
# ============================================================================

class ImprovedLSTMWithUncertainty(nn.Module):
    """
    Enhanced LSTM with:
    - GELU activation (better than ReLU for financial data)
    - Layer normalization (more stable than batch norm)
    - Dropout for uncertainty estimation
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ImprovedLSTMWithUncertainty, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Improved classifier with GELU and LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.LayerNorm(32),      # More stable than BatchNorm
            nn.GELU(),             # Better than ReLU for small datasets
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        return output
    
    def predict_with_uncertainty(self, x, n_samples=10):
        """
        Monte Carlo Dropout for uncertainty estimation
        Returns mean prediction and uncertainty
        """
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

# ============================================================================
# IMPROVEMENT 3: BETTER FEATURE ENGINEERING
# ============================================================================

class EnhancedFeatureEngineer:
    """
    Advanced feature engineering with market regime detection
    """
    
    @staticmethod
    def calculate_rsi(prices, window=14):
        """RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def add_advanced_features(df):
        """
        Add the most impactful technical indicators and market features
        """
        logger.info("Adding advanced features...")
        
        # Price-based features
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        df['RSI'] = EnhancedFeatureEngineer.calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-6)
        
        # Volatility and momentum
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_3d'] = df['Close'].pct_change(3)
        df['Price_Change_7d'] = df['Close'].pct_change(7)
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA'] + 1e-6)
        
        # Market regime features (NEW)
        df['Trend_Strength'] = (df['SMA_10'] - df['SMA_20']) / (df['Close'] + 1e-6)
        df['High_Low_Spread'] = (df['High'] - df['Low']) / (df['Close'] + 1e-6)
        df['Close_to_High'] = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-6)
        
        # Cyclical features using Fourier (NEW)
        n = len(df)
        for period in [7, 14, 30]:  # Weekly, bi-weekly, monthly
            df[f'sin_{period}'] = np.sin(2 * np.pi * np.arange(n) / period)
            df[f'cos_{period}'] = np.cos(2 * np.pi * np.arange(n) / period)
        
        # Rate of change
        df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        
        return df

# ============================================================================
# IMPROVEMENT 4: DATA AUGMENTATION
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation: blend samples to improve generalization
    Helps prevent overfitting on small datasets
    """
    if alpha > 0:
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha)
        
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    return x, y

# ============================================================================
# MAIN IMPROVED PREDICTOR CLASS
# ============================================================================

class FinalImprovedCryptoPredictor:
    """
    Final improved cryptocurrency predictor with all enhancements
    """
    
    def __init__(self, coin_symbol='BTC-USD', lookback_days=30, hidden_size=64):
        self.coin_symbol = coin_symbol
        self.lookback_days = lookback_days
        self.hidden_size = hidden_size
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.model = None
        self.device = device
        self.feature_columns = []
        
        logger.info(f"Initialized predictor for {coin_symbol}")
        
    def fetch_and_prepare_data(self, period='730d', interval='1h'):
        """
        Fetch data and prepare features
        """
        logger.info(f"Fetching {period} of data for {self.coin_symbol}...")
        
        # Fetch price data
        ticker = yf.Ticker(self.coin_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError("No data fetched")
        
        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        
        logger.info(f"Fetched {len(df)} days of data")
        
        # Add advanced features
        df = EnhancedFeatureEngineer.add_advanced_features(df)
        
        # Try to add sentiment (optional)
        try:
            response = requests.get("https://api.alternative.me/fng/?limit=365", timeout=10)
            if response.status_code == 200:
                fear_greed = pd.DataFrame(response.json()['data'])
                fear_greed['timestamp'] = pd.to_datetime(fear_greed['timestamp'], unit='s')
                fear_greed = fear_greed.set_index('timestamp')
                fear_greed['fear_greed'] = fear_greed['value'].astype(float)
                df = df.join(fear_greed[['fear_greed']], how='left')
                df['fear_greed'] = df['fear_greed'].fillna(method='ffill').fillna(50)
                logger.info("Added sentiment data")
        except:
            df['fear_greed'] = 50  # Neutral sentiment as fallback
            
        # Create target
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Prepared {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def select_best_features(self, df, n_features=20):
        """
        Select the most predictive features
        """
        logger.info(f"Selecting top {n_features} features...")
        
        # Define potential features (exclude target and non-numeric)
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df['Target']
        
        # Select best features
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        self.feature_columns = selected_features
        self.feature_selector = selector
        
        logger.info(f"Selected features: {selected_features[:5]}... ({len(selected_features)} total)")
        
        return df[selected_features + ['Target']]
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM
        """
        features = data[self.feature_columns].values
        targets = data['Target'].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.lookback_days, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_days:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def train(self, epochs=150, batch_size=32, lr=0.001, patience=20):
        """
        Train the improved model
        """
        logger.info("Starting training...")
        
        # Get data
        df = self.fetch_and_prepare_data(period='2y')
        df = self.select_best_features(df, n_features=20)
        
        # Create sequences
        X, y = self.create_sequences(df)
        logger.info(f"Created {len(X)} sequences")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = ImprovedLSTMWithUncertainty(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=0.3
        ).to(self.device)
        
        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                # Apply mixup augmentation
                batch_X, batch_y = mixup_data(batch_X, batch_y, alpha=0.2)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Use focal loss instead of BCE
                loss = focal_loss(outputs, batch_y, alpha=0.25, gamma=2.0)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = focal_loss(outputs, batch_y)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_targets, all_preds)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}] '
                          f'Train Loss: {avg_train_loss:.4f} '
                          f'Val Loss: {avg_val_loss:.4f} '
                          f'Val Acc: {val_accuracy:.4f}')
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Final evaluation with uncertainty
        self.model.eval()
        mean_pred, uncertainty = self.model.predict_with_uncertainty(X_val, n_samples=20)
        final_preds = (mean_pred > 0.5).float()
        final_accuracy = accuracy_score(y_val.cpu(), final_preds.cpu())
        
        logger.info(f"\nFinal Results:")
        logger.info(f"Validation Accuracy: {final_accuracy:.4f}")
        logger.info(f"Mean Uncertainty: {uncertainty.mean():.4f}")
        
        return history
    
    def predict_next(self, confidence_threshold=0.6):
        """
        Predict next day with uncertainty
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get recent data
        df = self.fetch_and_prepare_data(period='3mo')
        df = self.select_best_features(df, n_features=20)
        
        # Get last sequence
        features = df[self.feature_columns].values
        features_scaled = self.scaler.transform(features)
        
        if len(features_scaled) >= self.lookback_days:
            last_sequence = features_scaled[-self.lookback_days:]
            X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            
            # Predict with uncertainty
            mean_pred, uncertainty = self.model.predict_with_uncertainty(X, n_samples=30)
            
            prediction = mean_pred.item()
            confidence = 1 - uncertainty.item()  # Convert uncertainty to confidence
            
            # Decision based on confidence
            if confidence < confidence_threshold:
                decision = "HOLD (Low Confidence)"
            elif prediction > 0.5:
                decision = "BUY"
            else:
                decision = "SELL"
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'decision': decision,
                'next_day_up_probability': prediction
            }
        else:
            raise ValueError("Not enough data for prediction")
    
    def save_model(self, path='final_improved_model.pth'):
        """
        Save the complete model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'lookback_days': self.lookback_days,
            'hidden_size': self.hidden_size,
            'coin_symbol': self.coin_symbol
        }, path)
        logger.info(f"Model saved to {path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to train and test the improved model
    """
    print("="*60)
    print("FINAL IMPROVED CRYPTO PREDICTOR")
    print("="*60)
    
    # Initialize predictor
    predictor = FinalImprovedCryptoPredictor(
        coin_symbol='BTC-USD',
        lookback_days=30,
        hidden_size=128  # Increase model size
    )
    
    # Train model
    history = predictor.train(
        epochs=1000,         # Much longer training
        batch_size=256,      # Larger batch size for GPU
        lr=0.001,
        patience=100         # Less sensitive early stopping
    )
    
    # Make prediction
    try:
        prediction = predictor.predict_next(confidence_threshold=0.6)
        print("\n" + "="*60)
        print("PREDICTION FOR TOMORROW:")
        print("="*60)
        print(f"Probability of Price Increase: {prediction['prediction']:.2%}")
        print(f"Confidence Level: {prediction['confidence']:.2%}")
        print(f"Recommended Action: {prediction['decision']}")
    except Exception as e:
        print(f"Prediction error: {e}")
    
    # Save model
    predictor.save_model()
    
    # Plot results
    if history:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', alpha=0.7)
        plt.plot(history['val_loss'], label='Val Loss', alpha=0.7)
        plt.title('Model Loss (Focal Loss)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Training complete! Files saved:")
        print("   - best_model.pth (best checkpoint)")
        print("   - final_improved_model.pth (complete model)")
        print("   - training_results.png (performance plots)")

if __name__ == "__main__":
    main()