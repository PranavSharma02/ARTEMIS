import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
import argparse
import os
import glob
warnings.filterwarnings('ignore')

from tradingPerformance import PerformanceEstimator

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Self-attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(attn_output)
        return self.layer_norm(x + output)

class EnhancedGRUModel(nn.Module):
    """Enhanced GRU with Attention"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, use_attention=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads=8, dropout=dropout)
        
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 1, dropout=dropout, batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size//2)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.batch_norm3 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.batch_norm4 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.1)
        
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        # First GRU layer
        gru_out, _ = self.gru(x)
        
        if self.use_attention:
            gru_out = self.attention(gru_out)
        
        # Second GRU layer
        gru_out2, hidden = self.gru2(gru_out)
        
        # Global features
        gru_final = hidden[-1]  # Last hidden state (batch_size, hidden_size//2)
        global_avg = gru_out2.mean(dim=1)  # Global average pooling (batch_size, hidden_size//2)
        
        # Combine representations
        combined = torch.cat([gru_final, global_avg], dim=1)  # (batch_size, hidden_size)
        
        # Dense layers with batch norm
        x = F.relu(self.batch_norm3(self.fc1(combined)))
        x = self.dropout1(x)
        
        x = F.relu(self.batch_norm4(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        return torch.tanh(self.output(x))

class RegimeDetector(nn.Module):
    """Market regime detection model"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(32)
        
        self.lstm = nn.LSTM(32, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 4)  # 4 regimes
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        
        x = x.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        return F.softmax(self.fc2(x), dim=1)

class MetaLearner(nn.Module):
    """Meta-learner for adaptive ensemble"""
    def __init__(self, n_models=5, market_features=10):
        super().__init__()
        
        self.model_processor = nn.Linear(n_models, 16)
        self.market_processor = nn.Linear(market_features, 16)
        
        self.fc1 = nn.Linear(32, 32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.2)
        
        self.prediction_head = nn.Linear(16, 1)
        self.confidence_head = nn.Linear(16, 1)
        
    def forward(self, model_predictions, market_features):
        pred_processed = F.relu(self.model_processor(model_predictions))
        market_processed = F.relu(self.market_processor(market_features))
        
        combined = torch.cat([pred_processed, market_processed], dim=1)
        
        x = F.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        prediction = torch.tanh(self.prediction_head(x))
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return prediction, confidence

class NovelEnsembleLoss(nn.Module):
    """Novel loss function with CVaR and signal consistency"""
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, predictions, returns):
        signals = torch.tanh(predictions)
        portfolio_returns = returns * signals.squeeze()
        
        # Sharpe ratio component
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8
        sharpe_ratio = mean_return / std_return
        
        # CVaR component (simplified for autograd compatibility)
        sorted_returns, _ = torch.sort(portfolio_returns)
        n_samples = sorted_returns.size(0)
        cvar_samples = max(1, int(n_samples * self.alpha))
        cvar = -torch.mean(sorted_returns[:cvar_samples])
        
        # Signal consistency penalty
        if len(signals) > 1:
            signal_changes = torch.mean(torch.abs(signals[1:] - signals[:-1]))
        else:
            signal_changes = torch.tensor(0.0, device=signals.device)
        
        # Combined loss
        risk_adjusted_return = 0.7 * sharpe_ratio
        risk_penalty = 0.1 * cvar + 0.05 * signal_changes
        
        total_loss = -(risk_adjusted_return - risk_penalty)
        
        return total_loss

class PyTorchNovelTradingSystem:
    """
    PyTorch Implementation of Novel Deep Learning Trading System
    
    Features:
    - Dynamic Market Regime-Aware Ensemble (DMRAE)
    - Multi-Horizon Hierarchical Attention Ensemble (MHHAE)
    - Adaptive Risk-Aware Meta-Learning Ensemble (ARAMLE)
    - GPU Optimized for RTX 4070
    """
    
    def __init__(self, sequence_length=60, initial_balance=100000, device=None):
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        
        # GPU Setup for RTX 4070
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"🚀 Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"💎 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🔋 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.scaler = RobustScaler()
        self.models = []
        self.multi_horizon_models = {15: [], 30: [], 60: [], 90: []}
        self.regime_detector = None
        self.meta_learner = None
        self.current_stock = None
        
    def find_stock_data_files(self, stock_symbol):
        """Find available data files for a given stock symbol"""
        data_dir = "Data"
        if not os.path.exists(data_dir):
            print(f"❌ Data directory '{data_dir}' not found!")
            return []
        
        # Search for files containing the stock symbol
        pattern = f"{data_dir}/{stock_symbol}*.csv"
        files = glob.glob(pattern)
        
        if not files:
            # Try case-insensitive search
            all_files = glob.glob(f"{data_dir}/*.csv")
            files = [f for f in all_files if stock_symbol.upper() in f.upper()]
        
        return sorted(files)
    
    def list_available_stocks(self):
        """List all available stock symbols in the data directory"""
        data_dir = "Data"
        if not os.path.exists(data_dir):
            print(f"❌ Data directory '{data_dir}' not found!")
            return []
        
        csv_files = glob.glob(f"{data_dir}/*.csv")
        stocks = set()
        
        for file in csv_files:
            filename = os.path.basename(file)
            # Extract stock symbol (everything before the first underscore or dot)
            stock_symbol = filename.split('_')[0].split('.')[0]
            stocks.add(stock_symbol)
        
        return sorted(list(stocks))
    
    def select_best_data_file(self, stock_symbol):
        """Select the best data file for a stock (prefer longer time series)"""
        files = self.find_stock_data_files(stock_symbol)
        
        if not files:
            return None
        
        # Prefer files with longer date ranges (more data)
        best_file = None
        max_size = 0
        
        for file in files:
            try:
                size = os.path.getsize(file)
                if size > max_size:
                    max_size = size
                    best_file = file
            except:
                continue
        
        return best_file
    
    def load_and_prepare_data(self, file_path=None, stock_symbol=None):
        """Load and prepare data for training - can use file path or stock symbol"""
        if file_path is None and stock_symbol is None:
            raise ValueError("Either file_path or stock_symbol must be provided!")
        
        # If stock_symbol is provided, find the best file
        if stock_symbol is not None:
            self.current_stock = stock_symbol.upper()
            file_path = self.select_best_data_file(stock_symbol)
            
            if file_path is None:
                available_stocks = self.list_available_stocks()
                print(f"❌ No data found for stock: {stock_symbol}")
                print(f"📊 Available stocks: {', '.join(available_stocks[:10])}{'...' if len(available_stocks) > 10 else ''}")
                return None
            
            print(f"📈 Selected data file: {file_path}")
        else:
            # Extract stock symbol from file path
            filename = os.path.basename(file_path)
            self.current_stock = filename.split('_')[0].split('.')[0].upper()
        
        # Load the data
        self.data = pd.read_csv(file_path)
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])
        self.data.set_index('Timestamp', inplace=True)
        self.data = self.data.sort_index()
        
        print(f"📊 Loaded {len(self.data)} data points for {self.current_stock}")
        print(f"   Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
        return self.data

    def create_advanced_technical_indicators(self):
        """Create comprehensive technical indicators with market regime detection"""
        df = self.data.copy()
        
        # Basic price indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multiple timeframe moving averages
        for window in [3, 5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
            
        # Price position indicators
        for sma in [20, 50, 100, 200]:
            df[f'Price_vs_SMA{sma}'] = (df['Close'] - df[f'SMA_{sma}']) / df[f'SMA_{sma}']
            df[f'SMA20_vs_SMA{sma}'] = (df['SMA_20'] - df[f'SMA_{sma}']) / df[f'SMA_{sma}']
        
        # Advanced volatility indicators
        for window in [5, 10, 20, 50]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'ParkinsonVol_{window}'] = np.sqrt(
                252 * (np.log(df['High'] / df['Low'])**2).rolling(window=window).mean()
            )
        
        # Market regime indicators
        df['VIX_Proxy'] = df['Volatility_20'].rolling(window=5).mean()
        df['Regime_Bull'] = ((df['SMA_20'] > df['SMA_50']) & 
                            (df['SMA_50'] > df['SMA_200'])).astype(int)
        df['Regime_Bear'] = ((df['SMA_20'] < df['SMA_50']) & 
                            (df['SMA_50'] < df['SMA_200'])).astype(int)
        
        # Enhanced momentum indicators
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = calculate_rsi(df['Close'], period)
        
        # Multiple MACD configurations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            exp1 = df['Close'].ewm(span=fast).mean()
            exp2 = df['Close'].ewm(span=slow).mean()
            macd = exp1 - exp2
            macd_signal = macd.ewm(span=signal).mean()
            df[f'MACD_{fast}_{slow}'] = macd
            df[f'MACD_Signal_{fast}_{slow}'] = macd_signal
            df[f'MACD_Hist_{fast}_{slow}'] = macd - macd_signal
        
        # Additional indicators (keeping it concise for space)
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        for period in [3, 5, 10, 20, 50]:
            df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'ROC_{period}'] = df['Close'].pct_change(period)
        
        self.features_df = df
        return df
    
    def prepare_sequences(self, target_col='Returns'):
        """Prepare sequences for PyTorch training"""
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['Returns', 'Log_Returns'] and 
                       self.features_df[col].dtype in ['float64', 'int64']]
        
        # Clean data
        self.features_df = self.features_df.dropna()
        
        # Prepare features and target
        X_data = self.features_df[feature_cols].values
        y_data = self.features_df[target_col].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_data)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            y_sequences.append(y_data[i])
        
        self.X = torch.FloatTensor(np.array(X_sequences)).to(self.device)
        self.y = torch.FloatTensor(np.array(y_sequences)).to(self.device)
        
        print(f"Created {len(self.X)} sequences with shape {self.X.shape}")
        print(f"Features used ({len(feature_cols)}): {feature_cols[:10]}...")
        
        self.feature_names = feature_cols
        return self.X, self.y
    
    def train_novel_ensemble(self, n_models=5, epochs=150, batch_size=32):
        """Train the novel ensemble with multiple innovative components"""
        print(f"🚀 Training Novel Multi-Component Ensemble with {n_models} models on {self.device}")
        
        # 1. Train base models with different configurations
        model_configs = [
            {'hidden_size': 128, 'dropout': 0.2, 'use_attention': True},
            {'hidden_size': 96, 'dropout': 0.25, 'use_attention': True},
            {'hidden_size': 112, 'dropout': 0.15, 'use_attention': True},
            {'hidden_size': 80, 'dropout': 0.3, 'use_attention': True},
            {'hidden_size': 144, 'dropout': 0.18, 'use_attention': False}
        ]
        
        self.models = []
        criterion = NovelEnsembleLoss()
        
        # Split data
        dataset = TensorDataset(self.X, self.y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for i, config in enumerate(model_configs[:n_models]):
            print(f"\n  Training Model {i+1}/{n_models} with config: {config}")
            
            # Create model
            model = EnhancedGRUModel(
                input_size=self.X.shape[2],
                **config
            ).to(self.device)
            
            # Optimizer with different learning rates
            optimizer = optim.Adam(model.parameters(), lr=0.001 * (0.9 ** (i // 2)), weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            self.models.append(model)
            print(f"  Model {i+1} final val_loss: {best_val_loss:.6f}")
        
        # 2. Train regime detector
        print("\n🔍 Training Market Regime Detector...")
        self.regime_detector = RegimeDetector(self.X.shape[2]).to(self.device)
        regime_labels = self._create_regime_labels()
        
        optimizer = optim.Adam(self.regime_detector.parameters(), lr=0.0005)
        criterion_regime = nn.CrossEntropyLoss()
        
        for epoch in range(50):
            self.regime_detector.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                regime_probs = self.regime_detector(batch_x)
                # Get corresponding regime labels for this batch
                batch_regime_labels = regime_labels[train_dataset.indices][:len(batch_x)]
                batch_regime_labels = torch.LongTensor(batch_regime_labels).to(self.device)
                
                loss = criterion_regime(regime_probs, batch_regime_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Regime Detector Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.6f}")
        
        # 3. Train meta-learner
        print("\n🧠 Training Meta-Learner...")
        self.meta_learner = MetaLearner(n_models=len(self.models)).to(self.device)
        
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=0.0005)
        
        for epoch in range(40):
            self.meta_learner.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Get base model predictions
                base_predictions = []
                for model in self.models:
                    model.eval()
                    with torch.no_grad():
                        pred = model(batch_x)
                        base_predictions.append(pred.squeeze())
                
                base_predictions = torch.stack(base_predictions, dim=1)
                
                # Market features (use subset of input features)
                market_features = batch_x[:, -1, :10]  # Last timestep, first 10 features
                
                # Meta-learner prediction
                meta_pred, confidence = self.meta_learner(base_predictions, market_features)
                
                # Combined loss
                pred_loss = F.mse_loss(meta_pred.squeeze(), batch_y)
                conf_loss = F.binary_cross_entropy(confidence.squeeze(), 
                                                  torch.ones_like(confidence.squeeze()) * 0.7)  # Target confidence
                
                loss = 0.8 * pred_loss + 0.2 * conf_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                # Set models back to train mode
                for model in self.models:
                    model.train()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Meta-Learner Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.6f}")
        
        print(f"\n✅ Novel Ensemble Training Complete!")
        print(f"   - Base models: {len(self.models)}")
        print(f"   - Regime detector: {'✓' if self.regime_detector else '✗'}")
        print(f"   - Meta-learner: {'✓' if self.meta_learner else '✗'}")
        
        return self.models
    
    def _create_regime_labels(self):
        """Create regime labels for training regime detector"""
        n_samples = len(self.X)
        regime_labels = np.zeros(n_samples)
        
        returns = self.features_df['Returns'].dropna()
        volatility = self.features_df['Volatility_20'].dropna()
        
        # Align lengths
        min_len = min(n_samples, len(returns), len(volatility))
        returns = returns.iloc[-min_len:]
        volatility = volatility.iloc[-min_len:]
        
        vol_threshold = volatility.median()
        
        for i in range(min_len):
            if i < 20:
                regime_labels[i] = 2  # Default to sideways low vol
                continue
                
            recent_returns = returns.iloc[max(0, i-20):i]
            current_vol = volatility.iloc[i]
            
            trend = recent_returns.mean()
            
            if trend > 0.002 and current_vol < vol_threshold:
                regime_labels[i] = 0  # Bull market
            elif trend < -0.002 and current_vol < vol_threshold:
                regime_labels[i] = 1  # Bear market
            elif current_vol >= vol_threshold:
                regime_labels[i] = 3  # High volatility
            else:
                regime_labels[i] = 2  # Low volatility
        
        return regime_labels[-n_samples:]
    
    def generate_novel_ensemble_signals(self, start_idx=None, end_idx=None):
        """Generate signals using novel ensemble method"""
        if not self.models:
            raise ValueError("Models must be trained first!")
        
        if start_idx is None:
            start_idx = self.sequence_length
        if end_idx is None:
            end_idx = len(self.features_df)
        
        # Prepare data
        feature_cols = self.feature_names
        data_subset = self.features_df.iloc[start_idx-self.sequence_length:end_idx]
        
        X_data = data_subset[feature_cols].values
        X_scaled = self.scaler.transform(X_data)
        
        # Create sequences
        X_sequences = []
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
        
        X_sequences = torch.FloatTensor(np.array(X_sequences)).to(self.device)
        
        # 1. Get base model predictions
        base_predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                predictions = model(X_sequences)
                base_predictions.append(predictions.cpu().numpy().flatten())
        
        base_predictions = np.array(base_predictions).T
        
        # 2. Detect market regimes
        if self.regime_detector:
            self.regime_detector.eval()
            with torch.no_grad():
                regime_probs = self.regime_detector(X_sequences)
                dominant_regime = torch.argmax(regime_probs, dim=1).cpu().numpy()
        else:
            dominant_regime = np.zeros(len(base_predictions))
        
        # 3. Dynamic ensemble weighting based on regime
        final_signals = np.zeros(len(base_predictions))
        
        for i in range(len(base_predictions)):
            regime = dominant_regime[i]
            
            # Regime-based weighting
            if regime == 0:  # Bull market
                regime_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
            elif regime == 1:  # Bear market
                regime_weights = np.array([0.15, 0.2, 0.25, 0.3, 0.1])
            elif regime == 2:  # Sideways low vol
                regime_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            else:  # High volatility
                regime_weights = np.array([0.1, 0.15, 0.3, 0.35, 0.1])
            
            # Ensure weights match number of models
            regime_weights = regime_weights[:len(self.models)]
            regime_weights = regime_weights / regime_weights.sum()
            
            # Base prediction
            base_signal = np.average(base_predictions[i], weights=regime_weights)
            
            # Meta-learner refinement
            if self.meta_learner and i < len(base_predictions):
                self.meta_learner.eval()
                with torch.no_grad():
                    try:
                        market_features = X_sequences[i:i+1, -1, :10]  # Last timestep, first 10 features
                        model_preds = torch.FloatTensor(base_predictions[i:i+1]).to(self.device)
                        
                        meta_pred, confidence = self.meta_learner(model_preds, market_features)
                        
                        confidence_weight = confidence.cpu().item()
                        meta_prediction = meta_pred.cpu().item()
                        
                        final_signal = (1 - confidence_weight) * base_signal + confidence_weight * meta_prediction
                    except:
                        final_signal = base_signal
            else:
                final_signal = base_signal
            
            final_signals[i] = final_signal
        
        return final_signals 

    def run_complete_novel_trading_system(self, stock_symbol=None, data_file=None):
        """Complete pipeline for the novel trading system"""
        # Determine what we're trading
        display_name = stock_symbol.upper() if stock_symbol else (self.current_stock or "Unknown")
        
        print("=" * 70)
        print("🚀 NOVEL PYTORCH DEEP LEARNING TRADING SYSTEM")
        print(f"   Multi-Component Ensemble with RTX 4070 GPU Acceleration")
        print(f"   📈 Trading Stock: {display_name}")
        print("=" * 70)
        
        try:
            # 1. Load and prepare data
            print("\n📊 Phase 1: Data Loading and Preparation")
            if stock_symbol:
                result = self.load_and_prepare_data(stock_symbol=stock_symbol)
            elif data_file:
                result = self.load_and_prepare_data(file_path=data_file)
            else:
                # Default to AAPL if nothing specified
                result = self.load_and_prepare_data(stock_symbol="AAPL")
            
            if result is None:
                return None
            
            # 2. Create technical indicators
            print(f"\n🔧 Phase 2: Advanced Technical Indicators Creation for {self.current_stock}")
            features_df = self.create_advanced_technical_indicators()
            print(f"   Created {len(features_df.columns)} features")
            
            # 3. Prepare sequences
            print(f"\n🔄 Phase 3: Sequence Preparation for Deep Learning")
            X, y = self.prepare_sequences()
            
            # 4. Train novel ensemble
            print(f"\n🎯 Phase 4: Novel Ensemble Training for {self.current_stock}")
            self.train_novel_ensemble(n_models=5, epochs=150, batch_size=32)
            
            # 5. Generate signals
            print(f"\n📈 Phase 5: Signal Generation for {self.current_stock}")
            signals = self.generate_novel_ensemble_signals()
            
            # 6. Backtest
            print(f"\n💰 Phase 6: Backtesting Performance for {self.current_stock}")
            results = self.backtest_novel_signals(signals)
            
            # 7. Performance analysis
            print(f"\n📊 Phase 7: Performance Analysis for {self.current_stock}")
            self.analyze_performance(results)
            
            print("\n" + "=" * 70)
            print(f"✅ NOVEL TRADING SYSTEM EXECUTION COMPLETE FOR {self.current_stock}!")
            print("=" * 70)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error in novel trading system for {display_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def backtest_novel_signals(self, signals):
        """Backtest the novel ensemble signals"""
        if len(signals) == 0:
            print("No signals to backtest")
            return None
        
        # Align signals with price data
        price_data = self.features_df['Close'].iloc[self.sequence_length:self.sequence_length + len(signals)]
        returns_data = self.features_df['Returns'].iloc[self.sequence_length:self.sequence_length + len(signals)]
        
        # Get date range for proper annual return calculation
        start_date = self.features_df.index[self.sequence_length]
        end_date = self.features_df.index[self.sequence_length + len(signals) - 1]
        trading_days = len(signals)
        years = trading_days / 252.0  # Keep 252 trading days per year
        
        # Trading simulation
        positions = np.tanh(signals)  # Convert to position sizes [-1, 1]
        portfolio_returns = positions * returns_data.values
        
        # Calculate cumulative returns (compound growth)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns[-1] - 1
        
        # Calculate annualized return using TDQN's formula structure but with 252 days
        if years > 0 and total_return > -1:
            annual_return = ((1 + total_return) ** (1 / years)) - 1
        else:
            annual_return = 0
        
        # Calculate Sharpe ratio (annualized)
        excess_returns = portfolio_returns - 0.02/252  # Assume 2% risk-free rate
        if np.std(portfolio_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(portfolio_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        rolling_max = pd.Series(cumulative_returns).expanding().max()
        drawdowns = (pd.Series(cumulative_returns) / rolling_max) - 1
        max_drawdown = drawdowns.min()
        
        # Win rate
        winning_trades = np.sum(portfolio_returns > 0)
        total_trades = len(portfolio_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate buy & hold performance for comparison
        buy_hold_total_return = (price_data.iloc[-1] / price_data.iloc[0]) - 1
        if years > 0 and buy_hold_total_return > -1:
            buy_hold_annual_return = ((1 + buy_hold_total_return) ** (1 / years)) - 1
        else:
            buy_hold_annual_return = 0
        
        results = {
            'signals': signals,
            'positions': positions,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'price_data': price_data,
            'returns_data': returns_data,
            'trading_days': trading_days,
            'years': years,
            'start_date': start_date,
            'end_date': end_date,
            'buy_hold_total_return': buy_hold_total_return,
            'buy_hold_annual_return': buy_hold_annual_return
        }
        
        return results
    
    def analyze_performance(self, results):
        """Analyze and print performance metrics"""
        if results is None:
            print("No results to analyze")
            return
        
        stock_name = self.current_stock or "Unknown"
        
        print(f"\n{'='*50}")
        print(f"🏆 NOVEL ENSEMBLE PERFORMANCE RESULTS - {stock_name}")
        print(f"{'='*50}")
        
        print(f"📊 Portfolio Metrics:")
        print(f"   Total Return:        {results['total_return']:.2%}")
        print(f"   Cumulative Return:   {results['total_return']:.2%}")
        print(f"   Annual Return:       {results['annual_return']:.2%}")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown:        {results['max_drawdown']:.2%}")
        print(f"   Win Rate:            {results['win_rate']:.1%}")
        print(f"   Total Trades:        {results['total_trades']:,}")
        
        # Time period information
        print(f"\n📅 Trading Period:")
        print(f"   Start Date:          {results['start_date'].strftime('%Y-%m-%d')}")
        print(f"   End Date:            {results['end_date'].strftime('%Y-%m-%d')}")
        print(f"   Trading Days:        {results['trading_days']:,}")
        print(f"   Years:               {results['years']:,.2f}")
        
        # Daily metrics
        daily_returns = results['portfolio_returns']
        print(f"\n📈 Daily Performance:")
        print(f"   Avg Daily Return:    {daily_returns.mean():.4f}")
        print(f"   Daily Volatility:    {daily_returns.std():.4f}")
        print(f"   Best Day:            {daily_returns.max():.4f}")
        print(f"   Worst Day:           {daily_returns.min():.4f}")
        
        # Compare to buy-and-hold
        print(f"\n🏪 Benchmark Comparison:")
        print(f"   Buy & Hold Total:    {results['buy_hold_total_return']:.2%}")
        print(f"   Buy & Hold Annual:   {results['buy_hold_annual_return']:.2%}")
        print(f"   Strategy vs B&H:     {results['total_return'] - results['buy_hold_total_return']:.2%}")
        print(f"   Annual Alpha:        {results['annual_return'] - results['buy_hold_annual_return']:.2%}")
        
        return results

def main():
    """Main execution function with command-line argument support"""
    parser = argparse.ArgumentParser(description='Novel PyTorch Deep Learning Trading System')
    parser.add_argument('--stock', '-s', type=str, default=None,
                       help='Stock symbol to trade (e.g., AAPL, MSFT, GOOGL)')
    parser.add_argument('--file', '-f', type=str, default=None,
                       help='Specific data file path to use')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available stock symbols')
    parser.add_argument('--epochs', '-e', type=int, default=150,
                       help='Number of training epochs (default: 150)')
    parser.add_argument('--models', '-m', type=int, default=5,
                       help='Number of ensemble models (default: 5)')
    
    args = parser.parse_args()
    
    print("🚀 Initializing PyTorch Novel Trading System...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"🔋 VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    # Initialize system
    system = PyTorchNovelTradingSystem(
        sequence_length=60,
        initial_balance=100000
    )
    
    # List available stocks if requested
    if args.list:
        available_stocks = system.list_available_stocks()
        print(f"\n📊 Available Stock Symbols ({len(available_stocks)}):")
        for i, stock in enumerate(available_stocks):
            if i % 10 == 0:
                print()
            print(f"{stock:>8}", end="")
        print("\n")
        return
    
    # Determine what to trade
    if args.stock and args.file:
        print("⚠️  Both --stock and --file specified. Using --stock.")
        args.file = None
    
    if not args.stock and not args.file:
        # Default behavior - show available stocks and ask user
        available_stocks = system.list_available_stocks()
        print(f"\n📊 Available stocks: {', '.join(available_stocks[:10])}{'...' if len(available_stocks) > 10 else ''}")
        print("💡 Use --stock SYMBOL to specify a stock (e.g., --stock AAPL)")
        print("📋 Use --list to see all available stocks")
        args.stock = "AAPL"  # Default to AAPL
        print(f"🔄 Defaulting to {args.stock}")
    
    # Run trading system
    print(f"\n🎯 Starting trading system...")
    if args.stock:
        print(f"📈 Target Stock: {args.stock.upper()}")
    elif args.file:
        print(f"📁 Data File: {args.file}")
    
    results = system.run_complete_novel_trading_system(
        stock_symbol=args.stock,
        data_file=args.file
    )
    
    if results:
        print(f"\n🎉 Trading system executed successfully for {system.current_stock}!")
        
        # Save model with stock-specific name
        model_filename = f'pytorch_novel_trading_models_{system.current_stock}.pth'
        print(f"\n💾 Saving trained models to {model_filename}...")
        torch.save({
            'stock_symbol': system.current_stock,
            'models': [model.state_dict() for model in system.models],
            'regime_detector': system.regime_detector.state_dict() if system.regime_detector else None,
            'meta_learner': system.meta_learner.state_dict() if system.meta_learner else None,
            'scaler': system.scaler,
            'feature_names': system.feature_names
        }, model_filename)
        print(f"✅ Models saved successfully!")
    else:
        print(f"❌ Trading system execution failed")

if __name__ == "__main__":
    main() 