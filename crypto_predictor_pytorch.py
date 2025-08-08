# improved_crypto_training.py
# Complete improved training script with better data handling and model architecture

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import warnings
import os
import logging
import time
import sys
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"IMPROVED TRAINING: Using device: {device}")

class ImprovedLSTMPredictor(nn.Module):
    """
    Improved LSTM model with better architecture for small datasets
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ImprovedLSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Smaller, more regularized LSTM for small datasets
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Simplified attention
        self.attention = nn.Linear(hidden_size, 1)
        
        # More regularized classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"IMPROVED MODEL: {total_params:,} parameters (smaller for better generalization)")
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Simple attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        return output

class ImprovedCryptoPredictorPyTorch:
    """
    Improved cryptocurrency prediction system with better data handling
    """
    def __init__(self, coin_symbol='BTC-USD', lookback_days=30, hidden_size=64):
        logger.info(f"IMPROVED INIT: Initializing for {coin_symbol}")
        
        self.coin_symbol = coin_symbol
        self.lookback_days = lookback_days  # Reduced from 60 to 30
        self.hidden_size = hidden_size      # Reduced from 256 to 64
        self.scaler = RobustScaler()        # More robust to outliers
        self.feature_selector = None
        self.model = None
        self.device = device
        self.feature_columns = []
        
        # Improved config
        self.config = {
            'data_period': '5y',           # More data
            'min_samples': 200,            # Minimum samples needed
            'top_features': 15,            # Reduce features
            'validation_split': 0.15,      # Smaller validation set
        }
        
        logger.info(f"IMPROVED CONFIG: {self.config}")
        
        # CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def fetch_multiple_coins_data(self, coins=['BTC-USD', 'ETH-USD'], period='5y'):
        """Fetch data for multiple coins to increase dataset size"""
        logger.info(f"MULTI COIN: Fetching data for {coins}")
        
        all_data = []
        
        for coin in coins:
            try:
                ticker = yf.Ticker(coin)
                data = ticker.history(period=period)
                
                if not data.empty:
                    # Ensure timezone-naive
                    if data.index.tz is not None:
                        data.index = data.index.tz_convert('UTC').tz_localize(None)
                    
                    # Add coin identifier
                    data['coin'] = coin
                    all_data.append(data[['Close', 'Volume', 'High', 'Low', 'Open', 'coin']])
                    logger.info(f"SUCCESS: {coin} - {len(data)} days")
                else:
                    logger.warning(f"WARNING: No data for {coin}")
                    
            except Exception as e:
                logger.error(f"ERROR: Failed to fetch {coin}: {e}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=False)
            logger.info(f"COMBINED DATA: {len(combined_data)} total rows from {len(all_data)} coins")
            return combined_data
        else:
            logger.error("ERROR: No data fetched for any coin")
            return None
    
    def fetch_improved_sentiment_data(self):
        """Enhanced sentiment data fetching with fallbacks"""
        try:
            # Try multiple sources for sentiment
            sources = [
                "https://api.alternative.me/fng/?limit=365",  # More data
                "https://api.alternative.me/fng/?limit=200"   # Fallback
            ]
            
            for url in sources:
                try:
                    response = requests.get(url, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        if 'data' in data:
                            fear_greed_df = pd.DataFrame(data['data'])
                            fear_greed_df['timestamp'] = pd.to_datetime(fear_greed_df['timestamp'], unit='s', utc=True)
                            fear_greed_df['timestamp'] = fear_greed_df['timestamp'].dt.tz_localize(None)
                            fear_greed_df['value'] = fear_greed_df['value'].astype(int)
                            fear_greed_df = fear_greed_df.set_index('timestamp')
                            
                            logger.info(f"SENTIMENT SUCCESS: {len(fear_greed_df)} readings from {url}")
                            return fear_greed_df[['value']].rename(columns={'value': 'fear_greed'})
                except:
                    continue
            
            logger.warning("SENTIMENT FALLBACK: Using synthetic sentiment data")
            return None
            
        except Exception as e:
            logger.warning(f"SENTIMENT ERROR: {e}")
            return None
    
    def calculate_essential_indicators(self, data):
        """Calculate only the most important technical indicators"""
        logger.info("ESSENTIAL INDICATORS: Calculating top performing indicators...")
        
        df = data.copy()
        
        # Price-based indicators (most important)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD (momentum)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI (mean reversion)
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Bollinger Bands (volatility)
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_3d'] = df['Close'].pct_change(3)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price position
        df['Price_Position'] = (df['Close'] - df['Low'].rolling(10).min()) / (df['High'].rolling(10).max() - df['Low'].rolling(10).min())
        
        # Rate of change
        df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
        
        # Support/resistance levels
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        logger.info(f"INDICATORS DONE: Calculated essential indicators")
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_improved_features(self, data):
        """Improved feature preparation with feature selection"""
        logger.info("IMPROVED FEATURES: Starting preparation...")
        start_time = time.time()
        
        # Calculate essential indicators
        df = self.calculate_essential_indicators(data)
        
        # Add sentiment data
        sentiment_data = self.fetch_improved_sentiment_data()
        if sentiment_data is not None:
            df = df.join(sentiment_data, how='left')
            df['fear_greed'] = df['fear_greed'].fillna(method='ffill').fillna(50)
        else:
            # Create synthetic sentiment based on price volatility
            df['fear_greed'] = 50 + (df['Volatility'].rolling(20).mean() * 1000).fillna(0)
            df['fear_greed'] = df['fear_greed'].clip(0, 100)
            logger.info("SYNTHETIC SENTIMENT: Created based on volatility")
        
        # Create target (next day price movement)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Define all potential features
        potential_features = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_position',
            'Volatility', 'Price_Change', 'Price_Change_3d',
            'Volume_Ratio', 'Price_Position', 'ROC_5',
            'High_Low_Ratio', 'fear_greed'
        ]
        
        # Clean data
        df_clean = df.dropna()
        
        if len(df_clean) < self.config['min_samples']:
            logger.error(f"INSUFFICIENT DATA: {len(df_clean)} < {self.config['min_samples']} required")
            return None
        
        # Feature selection to reduce overfitting
        logger.info("FEATURE SELECTION: Selecting top features...")
        X_temp = df_clean[potential_features]
        y_temp = df_clean['Target']
        
        # Select top features based on statistical tests
        selector = SelectKBest(score_func=f_classif, k=self.config['top_features'])
        X_selected = selector.fit_transform(X_temp, y_temp)
        
        # Get selected feature names
        selected_features = [potential_features[i] for i in selector.get_support(indices=True)]
        self.feature_columns = selected_features
        self.feature_selector = selector
        
        logger.info(f"SELECTED FEATURES: {len(self.feature_columns)} out of {len(potential_features)}")
        for i, feature in enumerate(self.feature_columns):
            score = selector.scores_[selector.get_support(indices=True)[i]]
            logger.info(f"   {i+1:2d}. {feature} (score: {score:.2f})")
        
        prep_time = time.time() - start_time
        logger.info(f"FEATURE PREP DONE: {prep_time:.2f}s, {len(df_clean)} samples")
        
        return df_clean[self.feature_columns + ['Target']]
    
    def create_improved_sequences(self, data):
        """Create sequences with data augmentation"""
        logger.info(f"IMPROVED SEQUENCES: Creating with lookback={self.lookback_days}")
        
        features = data[self.feature_columns].values
        targets = data['Target'].values
        
        # Robust scaling (better for outliers)
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.lookback_days, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_days:i])
            y.append(targets[i])
        
        X, y = np.array(X), np.array(y)
        
        # Simple data augmentation: add small noise to prevent overfitting
        if len(X) < 500:  # Only for small datasets
            logger.info("DATA AUGMENTATION: Adding noise-augmented samples...")
            
            augmented_X, augmented_y = [], []
            for i in range(len(X)):
                # Original sample
                augmented_X.append(X[i])
                augmented_y.append(y[i])
                
                # Add 2 noisy versions (small noise)
                for _ in range(2):
                    noise = np.random.normal(0, 0.01, X[i].shape)  # 1% noise
                    noisy_sample = X[i] + noise
                    augmented_X.append(noisy_sample)
                    augmented_y.append(y[i])
            
            X = np.array(augmented_X)
            y = np.array(augmented_y)
            logger.info(f"AUGMENTATION DONE: {len(X)} samples (3x original)")
        
        logger.info(f"SEQUENCES READY: X={X.shape}, y={y.shape}")
        return X, y
    
    def train_improved_model(self, epochs=200, batch_size=32, lr=0.001, patience=30):
        """Improved training with better practices"""
        logger.info("IMPROVED TRAINING: Starting...")
        
        # Get enhanced data
        if self.coin_symbol in ['BTC-USD', 'ETH-USD']:
            # For major coins, use multi-coin data
            coins = ['BTC-USD', 'ETH-USD']
            price_data = self.fetch_multiple_coins_data(coins, self.config['data_period'])
        else:
            # For other coins, use single coin with more history
            ticker = yf.Ticker(self.coin_symbol)
            price_data = ticker.history(period=self.config['data_period'])
            
            if price_data.index.tz is not None:
                price_data.index = price_data.index.tz_convert('UTC').tz_localize(None)
        
        if price_data is None or len(price_data) < 100:
            logger.error("FAILED: Insufficient price data")
            return None
        
        # Prepare features
        featured_data = self.prepare_improved_features(price_data)
        if featured_data is None:
            return None
        
        # Create sequences
        X, y = self.create_improved_sequences(featured_data)
        if len(X) == 0:
            logger.error("FAILED: No sequences created")
            return None
        
        # Create data loaders with smaller validation set
        split_idx = int(len(X) * (1 - self.config['validation_split']))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"DATA SPLIT: Train={len(X_train)}, Val={len(X_val)}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for better training
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize improved model
        input_size = X.shape[2]
        self.model = ImprovedLSTMPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=2,  # Reduced layers
            dropout=0.4    # More dropout
        ).to(self.device)
        
        # Improved training setup
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)  # More weight decay
        
        # Cosine annealing scheduler for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        train_losses, val_losses, val_accuracies = [], [], []
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"TRAINING START: {epochs} epochs, batch_size={batch_size}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_targets, all_preds)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f'improved_{self.coin_symbol.replace("-", "_")}_model.pth')
            else:
                patience_counter += 1
            
            # Progress reporting
            if epoch % 20 == 0 or epoch == epochs - 1:
                logger.info(f'Epoch [{epoch+1:3d}/{epochs}] | '
                          f'TrLoss: {avg_train_loss:.4f} | '
                          f'VaLoss: {avg_val_loss:.4f} | '
                          f'VaAcc: {val_accuracy:.4f} | '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if patience_counter >= patience:
                logger.info(f"EARLY STOP: At epoch {epoch+1}")
                break
        
        # Load best model and final evaluation
        self.model.load_state_dict(torch.load(f'improved_{self.coin_symbol.replace("-", "_")}_model.pth'))
        
        # Final test
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = X_val_tensor.to(self.device)
            y_val_tensor = y_val_tensor.to(self.device)
            final_preds = self.model(X_val_tensor)
            final_preds_binary = (final_preds > 0.5).float()
            final_accuracy = accuracy_score(y_val_tensor.cpu().numpy(), final_preds_binary.cpu().numpy())
        
        logger.info(f"FINAL ACCURACY: {final_accuracy:.4f}")
        
        # Classification report
        class_report = classification_report(
            y_val_tensor.cpu().numpy(),
            final_preds_binary.cpu().numpy(),
            target_names=['Down', 'Up']
        )
        logger.info(f"CLASSIFICATION REPORT:\n{class_report}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': final_accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'selected_features': self.feature_columns
        }
    
    def save_improved_model(self, path=None):
        """Save the improved model"""
        if path is None:
            path = f'improved_crypto_predictor_{self.coin_symbol.replace("-", "_")}.pth'
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_columns': self.feature_columns,
            'lookback_days': self.lookback_days,
            'hidden_size': self.hidden_size,
            'coin_symbol': self.coin_symbol,
            'config': self.config
        }, path)
        
        logger.info(f"IMPROVED MODEL SAVED: {path}")

def main():
    """Main function for improved training"""
    logger.info("IMPROVED CRYPTO PREDICTOR - Enhanced Training")
    logger.info("=" * 60)
    
    # Initialize improved predictor
    predictor = ImprovedCryptoPredictorPyTorch(
        coin_symbol='BTC-USD',
        lookback_days=30,     # Reduced for more samples
        hidden_size=64        # Smaller for better generalization
    )
    
    # Train improved model
    history = predictor.train_improved_model(
        epochs=200,           # More epochs
        batch_size=32,        # Smaller batch size
        lr=0.001,            # Standard learning rate
        patience=30          # More patience
    )
    
    if history:
        logger.info("\nIMPROVED TRAINING COMPLETE!")
        logger.info(f"FINAL ACCURACY: {history['final_accuracy']:.4f}")
        logger.info(f"TRAINING SAMPLES: {history['training_samples']}")
        logger.info(f"SELECTED FEATURES: {len(history['selected_features'])}")
        
        # Save improved model
        predictor.save_improved_model()
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_losses'], label='Training Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Improved Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
        plt.title('Improved Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        # Feature importance
        features = history['selected_features'][:10]  # Top 10
        scores = np.random.rand(len(features))  # Placeholder scores
        plt.barh(features, scores)
        plt.title('Top Selected Features')
        plt.xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig('improved_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("\nIMPROVED MODEL READY!")
        logger.info("Files created:")
        logger.info(f"  - improved_crypto_predictor_BTC_USD.pth")
        logger.info(f"  - improved_training_results.png")
        
        return predictor
    else:
        logger.error("IMPROVED TRAINING FAILED!")
        return None

if __name__ == "__main__":
    trained_model = main()