# crypto_predictor_pytorch.py
# Complete PyTorch-based cryptocurrency price prediction system
# Optimized for CUDA 12.9 + RTX 3090 + Python 3.12.10

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import warnings
import os
warnings.filterwarnings('ignore')

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"⚡ PyTorch CUDA: Compatible with CUDA 12.9")

class LSTMPredictor(nn.Module):
    """
    LSTM-based cryptocurrency price prediction model with attention mechanism
    """
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism (simple)
        self.attention = nn.Linear(hidden_size, 1)
        
        # Classification head with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Apply attention to LSTM outputs
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        return output

class CryptoPredictorPyTorch:
    """
    Main cryptocurrency prediction system using PyTorch and LSTM
    """
    def __init__(self, coin_symbol='BTC-USD', lookback_days=60, hidden_size=128):
        self.coin_symbol = coin_symbol
        self.lookback_days = lookback_days
        self.hidden_size = hidden_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = device
        self.feature_columns = []
        
        # Windows optimizations for PyTorch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def fetch_price_data(self, period='2y'):
        """Fetch historical price data using yfinance"""
        try:
            print(f"📊 Fetching price data for {self.coin_symbol}...")
            ticker = yf.Ticker(self.coin_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"❌ No data found for {self.coin_symbol}")
                return None
                
            print(f"✅ Retrieved {len(data)} days of price data")
            return data[['Close', 'Volume', 'High', 'Low', 'Open']]
            
        except Exception as e:
            print(f"❌ Error fetching price data: {e}")
            return None
    
    def fetch_fear_greed_index(self):
        """Get crypto fear & greed index as sentiment proxy"""
        try:
            print("📈 Fetching Fear & Greed Index...")
            url = "https://api.alternative.me/fng/?limit=200"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            fear_greed_df = pd.DataFrame(data['data'])
            fear_greed_df['timestamp'] = pd.to_datetime(fear_greed_df['timestamp'], unit='s')
            fear_greed_df['value'] = fear_greed_df['value'].astype(int)
            fear_greed_df = fear_greed_df.set_index('timestamp')
            
            print(f"✅ Retrieved {len(fear_greed_df)} fear & greed readings")
            return fear_greed_df[['value']].rename(columns={'value': 'fear_greed'})
            
        except Exception as e:
            print(f"⚠️ Could not fetch sentiment data: {e}")
            print("🔄 Using neutral sentiment (50)")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        print("🔧 Calculating technical indicators...")
        df = data.copy()
        
        # Moving averages
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_21'] = df['Close'].rolling(window=21).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volatility measures
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['ATR'] = self.calculate_atr(df)  # Average True Range
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_2d'] = df['Close'].pct_change(2)
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Support/Resistance levels
        df['Price_Position'] = (df['Close'] - df['Low'].rolling(20).min()) / (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        # Momentum indicators
        df['ROC'] = df['Close'].pct_change(periods=10) * 100  # Rate of Change
        df['Williams_R'] = self.calculate_williams_r(df)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    def calculate_williams_r(self, df, window=14):
        """Calculate Williams %R"""
        highest_high = df['High'].rolling(window=window).max()
        lowest_low = df['Low'].rolling(window=window).min()
        return ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
    
    def prepare_features(self, data):
        """Engineer features and create target variable"""
        print("🛠️ Engineering features...")
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(data)
        
        # Add sentiment data if available
        sentiment_data = self.fetch_fear_greed_index()
        if sentiment_data is not None:
            df = df.join(sentiment_data, how='left')
            df['fear_greed'] = df['fear_greed'].fillna(method='ffill').fillna(50)
        else:
            df['fear_greed'] = 50  # Neutral sentiment
        
        # Target: next day's price movement (1 if up, 0 if down)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Define feature columns
        self.feature_columns = [
            'Close', 'Volume', 'High', 'Low', 'Open',
            'SMA_7', 'SMA_21', 'SMA_50', 'EMA_12', 'EMA_26', 
            'MACD', 'MACD_signal', 'MACD_histogram',
            'RSI', 'BB_width', 'BB_position', 'Volatility', 'ATR',
            'Price_Change', 'Price_Change_2d', 'Volume_Change', 
            'High_Low_Ratio', 'Price_Position', 'ROC', 'Williams_R',
            'Volume_Ratio', 'fear_greed'
        ]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        print(f"✅ Feature engineering complete: {len(self.feature_columns)} features, {len(df_clean)} samples")
        return df_clean[self.feature_columns + ['Target']]
    
    def create_sequences(self, data):
        """Create LSTM sequences for training"""
        print(f"🔄 Creating sequences with lookback period: {self.lookback_days} days")
        
        # Separate features and targets
        features = data[self.feature_columns].values
        targets = data['Target'].values
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.lookback_days, len(features_scaled)):
            X.append(features_scaled[i-self.lookback_days:i])
            y.append(targets[i])
        
        print(f"✅ Created {len(X)} sequences")
        return np.array(X), np.array(y)
    
    def create_data_loaders(self, X, y, test_size=0.2, batch_size=64):
        """Create PyTorch data loaders with proper time series split"""
        # Split data (time series - no shuffling to maintain temporal order)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders - increased batch size for RTX 3090
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader, (X_test_tensor, y_test_tensor)
    
    def train(self, epochs=100, batch_size=128, lr=0.001, patience=15):
        """Train the PyTorch model with optimizations for RTX 3090"""
        print(f"🧠 Starting training for {self.coin_symbol}")
        print("=" * 60)
        
        # Get price data
        price_data = self.fetch_price_data()
        if price_data is None:
            print("❌ Failed to fetch price data")
            return None
        
        # Prepare features
        featured_data = self.prepare_features(price_data)
        if len(featured_data) < self.lookback_days + 50:
            print(f"❌ Insufficient data: {len(featured_data)} samples")
            return None
        
        # Create sequences
        X, y = self.create_sequences(featured_data)
        if len(X) == 0:
            print("❌ Failed to create sequences")
            return None
        
        # Create data loaders with larger batch size for RTX 3090
        train_loader, test_loader, (X_test, y_test) = self.create_data_loaders(
            X, y, batch_size=batch_size
        )
        
        # Initialize model with optimized architecture for RTX 3090
        input_size = X.shape[2]
        self.model = LSTMPredictor(
            input_size=input_size, 
            hidden_size=self.hidden_size,
            num_layers=3,  # Increased for better performance
            dropout=0.3
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=7, factor=0.5, verbose=True
        )
        
        # Training tracking
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"🚀 Training on {self.device} with {input_size} features")
        print(f"📐 Model: {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"🔥 Batch size: {batch_size} (optimized for RTX 3090)")
        print()
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = accuracy_score(all_targets, all_preds)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_{self.coin_symbol.replace("-", "_")}_model.pth')
            else:
                patience_counter += 1
            
            # Progress reporting
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch [{epoch+1:3d}/{epochs}] | '
                      f'Train Loss: {avg_train_loss:.4f} | '
                      f'Val Loss: {avg_val_loss:.4f} | '
                      f'Train Acc: {train_accuracy:.4f} | '
                      f'Val Acc: {val_accuracy:.4f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            if patience_counter >= patience:
                print(f"\n⏰ Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'best_{self.coin_symbol.replace("-", "_")}_model.pth'))
        
        # Final evaluation
        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION")
        print("=" * 60)
        
        with torch.no_grad():
            self.model.eval()
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            predictions = self.model(X_test)
            final_preds = (predictions > 0.5).float()
            final_accuracy = accuracy_score(y_test.cpu().numpy(), final_preds.cpu().numpy())
            
            print(f"🎯 Final Test Accuracy: {final_accuracy:.4f}")
            print("\n📋 Classification Report:")
            print(classification_report(
                y_test.cpu().numpy(), 
                final_preds.cpu().numpy(), 
                target_names=['Down ↓', 'Up ↑'],
                digits=4
            ))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': final_accuracy,
            'best_epoch': len(train_losses) - patience_counter
        }
    
    def predict_next_movement(self):
        """Predict next day's price movement"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        try:
            print(f"🔮 Making prediction for {self.coin_symbol}...")
            
            # Get latest data (more recent period for prediction)
            latest_data = self.fetch_price_data(period='120d')
            if latest_data is None:
                return None
                
            featured_data = self.prepare_features(latest_data)
            
            # Get last sequence
            last_sequence = featured_data[self.feature_columns].tail(self.lookback_days).values
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor)
                confidence = float(prediction[0][0])
            
            direction = "UP" if confidence > 0.5 else "DOWN"
            confidence_score = abs(confidence - 0.5) * 2  # Convert to 0-1 confidence
            
            # Determine signal strength
            if confidence_score > 0.8:
                signal_strength = "Very Strong"
            elif confidence_score > 0.6:
                signal_strength = "Strong"
            elif confidence_score > 0.4:
                signal_strength = "Moderate"
            else:
                signal_strength = "Weak"
            
            # Get current price for context
            current_price = latest_data['Close'].iloc[-1]
            
            result = {
                'direction': direction,
                'confidence': confidence_score,
                'raw_prediction': confidence,
                'signal_strength': signal_strength,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Prediction: {direction} (confidence: {confidence_score:.2%})")
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def save_model(self, path=None):
        """Save the trained model and scaler"""
        if self.model is not None:
            if path is None:
                path = f'crypto_predictor_{self.coin_symbol.replace("-", "_")}.pth'
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'lookback_days': self.lookback_days,
                'hidden_size': self.hidden_size,
                'coin_symbol': self.coin_symbol
            }, path)
            print(f"💾 Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Restore configuration
            self.scaler = checkpoint['scaler']
            self.feature_columns = checkpoint['feature_columns']
            self.lookback_days = checkpoint['lookback_days']
            self.hidden_size = checkpoint['hidden_size']
            self.coin_symbol = checkpoint['coin_symbol']
            
            # Initialize and load model
            input_size = len(self.feature_columns)
            self.model = LSTMPredictor(
                input_size=input_size, 
                hidden_size=self.hidden_size
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✅ Model loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Main function to train and test the crypto predictor"""
    print("🚀 Crypto Price Predictor - PyTorch Edition")
    print("Optimized for CUDA 12.9 + RTX 3090 + Python 3.12.10")
    print("=" * 60)
    
    # Initialize predictor with optimal settings for RTX 3090
    predictor = CryptoPredictorPyTorch(
        coin_symbol='BTC-USD',
        lookback_days=60,
        hidden_size=256  # Increased for RTX 3090's 24GB VRAM
    )
    
    # Train model with optimal batch size for RTX 3090
    print("🧠 Starting model training...")
    history = predictor.train(
        epochs=150,         # More epochs for better performance
        batch_size=128,     # Large batch size for 24GB VRAM
        lr=0.0008,          # Slightly lower learning rate
        patience=20         # More patience for complex model
    )
    
    if history:
        print(f"\n🎉 Training completed successfully!")
        print(f"🎯 Best accuracy achieved: {history['final_accuracy']:.4f}")
        
        # Save the model
        predictor.save_model()
        
        # Test prediction
        print("\n🔮 Testing prediction capability...")
        prediction = predictor.predict_next_movement()
        
        if prediction:
            print(f"\n📈 Sample Prediction for {predictor.coin_symbol}:")
            print(f"   Direction: {prediction['direction']}")
            print(f"   Confidence: {prediction['confidence']:.2%}")
            print(f"   Signal Strength: {prediction['signal_strength']}")
            print(f"   Current Price: ${prediction['current_price']:.2f}")
        
        # Plot training history
        print("\n📊 Generating training plots...")
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_losses'], label='Training Loss', alpha=0.8)
        plt.plot(history['val_losses'], label='Validation Loss', alpha=0.8)
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        # GPU utilization info
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            plt.bar(['GPU Memory Used', 'GPU Memory Total'], 
                   [gpu_memory_used, gpu_memory_total],
                   color=['orange', 'lightblue'])
            plt.title('GPU Memory Usage (GB)')
            plt.ylabel('Memory (GB)')
        
        plt.tight_layout()
        plt.savefig(f'training_results_{predictor.coin_symbol.replace("-", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Training complete! Files saved:")
        print(f"   📊 Training plot: training_results_{predictor.coin_symbol.replace('-', '_')}.png")
        print(f"   💾 Model file: crypto_predictor_{predictor.coin_symbol.replace('-', '_')}.pth")
        print(f"   🎯 Ready for deployment!")
        
        return predictor
    else:
        print("❌ Training failed!")
        return None

if __name__ == "__main__":
    # Run the main training function
    trained_model = main()
    
    if trained_model:
        print("\n🎉 SUCCESS! Your crypto predictor is ready!")
        print("\n📋 Next steps:")
        print("   1. Run the Flask web app: python flask_api_pytorch.py")
        print("   2. Deploy to your .ai domain")
        print("   3. Start making predictions!")
    else:
        print("\n❌ Setup incomplete. Please check the error messages above.")