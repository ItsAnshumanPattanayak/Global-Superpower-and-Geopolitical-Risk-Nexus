import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, csv_path='global.csv'):
        self.csv_path = csv_path
        self.df = None
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
    def load_data(self):
        """Load CSV dataset"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
            self.df.sort_index(inplace=True)
            print(f"✅ Data Loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def handle_missing_values(self):
        """Handle missing values using forward fill (max 3 days) as per dataset spec"""
        if self.df is None:
            return None
        
        # Forward fill with limit of 3 days (aligns weekend market data with Wikipedia logs)
        self.df = self.df.ffill(limit=3)
        
        # Backward fill for any remaining NaNs at the start of the series
        self.df = self.df.bfill()
        
        # Drop any truly persistent NaNs (safety check)
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        dropped = initial_rows - len(self.df)
        
        if dropped > 0:
            print(f"⚠️ Dropped {dropped} rows with persistent NaNs after imputation")
        
        remaining_nans = self.df.isnull().sum().sum()
        print(f"✅ Missing values handled: {remaining_nans} nulls remaining")
        return self.df
    
    def get_financial_features(self):
        """Extract financial features (OHLCV)"""
        financial_cols = [col for col in self.df.columns 
                         if any(x in col for x in ['_Open', '_High', '_Low', '_Close', '_Volume'])]
        return self.df[financial_cols]
    
    def get_geopolitical_features(self):
        """Extract geopolitical features (Views, Momentum, Shock)"""
        geopolitical_cols = [col for col in self.df.columns 
                           if any(x in col for x in ['_Views', '_Momentum', '_Shock'])]
        return self.df[geopolitical_cols]
    
    def normalize_data(self, method='standard'):
        """Normalize features"""
        if method == 'standard':
            df_normalized = pd.DataFrame(
                self.scaler.fit_transform(self.df),
                index=self.df.index,
                columns=self.df.columns
            )
        else:
            df_normalized = pd.DataFrame(
                self.minmax_scaler.fit_transform(self.df),
                index=self.df.index,
                columns=self.df.columns
            )
        return df_normalized
    
    def create_lagged_features(self, lags=[1, 5, 10, 20]):
        """Create lagged features for time series"""
        df_lagged = self.df.copy()
        
        for lag in lags:
            for col in self.df.columns:
                df_lagged[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        df_lagged.dropna(inplace=True)
        print(f"✅ Lagged features created: {df_lagged.shape[1]} features")
        return df_lagged
    
    def get_train_test_split(self, test_size=0.2):
        """Split data into train and test sets"""
        split_idx = int(len(self.df) * (1 - test_size))
        train = self.df[:split_idx]
        test = self.df[split_idx:]
        print(f"✅ Train: {train.shape[0]}, Test: {test.shape[0]}")
        return train, test
    
    def get_summary_stats(self):
        """Get data summary statistics"""
        return {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.to_dict(),
            'missing': self.df.isnull().sum().to_dict(),
            'stats': self.df.describe().to_dict()
        }