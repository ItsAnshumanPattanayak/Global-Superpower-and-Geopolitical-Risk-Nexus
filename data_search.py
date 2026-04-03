import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataSearchEngine:
    """Advanced data search and manual input functionality"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.data = data_processor.df
        self.search_history = []
        
    def get_available_assets(self):
        """Get list of available financial assets"""
        close_cols = [col for col in self.data.columns if '_Close' in col]
        return [col.replace('_Close', '') for col in close_cols]
    
    def get_available_geopolitical_terms(self):
        """Get list of available geopolitical terms"""
        views_cols = [col for col in self.data.columns if '_Views' in col]
        return [col.replace('_Views', '') for col in views_cols]
    
    def get_date_range(self):
        """Get available date range"""
        return {
            'start': self.data.index.min(),
            'end': self.data.index.max()
        }
    
    def search_by_date(self, date):
        """Search data by specific date"""
        try:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            if date in self.data.index:
                result = self.data.loc[date].to_dict()
                self.search_history.append({
                    'type': 'date_search',
                    'query': str(date.date()),
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
                return {'success': True, 'data': result, 'date': date}
            else:
                # Find nearest date
                nearest = self.data.index[self.data.index.get_indexer([date], method='nearest')[0]]
                result = self.data.loc[nearest].to_dict()
                self.search_history.append({
                    'type': 'date_search',
                    'query': str(date.date()),
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'note': f'Nearest date used: {nearest.date()}'
                })
                return {'success': True, 'data': result, 'date': nearest, 'nearest': True}
        except Exception as e:
            self.search_history.append({
                'type': 'date_search',
                'query': str(date),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            return {'success': False, 'error': str(e)}
    
    def search_by_date_range(self, start_date, end_date):
        """Search data by date range"""
        try:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            result = self.data.loc[start_date:end_date]
            
            self.search_history.append({
                'type': 'date_range_search',
                'query': f'{start_date.date()} to {end_date.date()}',
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'records_found': len(result)
            })
            
            return {
                'success': True,
                'data': result,
                'start_date': start_date,
                'end_date': end_date,
                'records': len(result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def search_by_asset(self, asset_name, start_date=None, end_date=None):
        """Search data by asset name"""
        try:
            # Find relevant columns
            asset_cols = [col for col in self.data.columns if asset_name in col]
            
            if not asset_cols:
                return {'success': False, 'error': f'Asset "{asset_name}" not found'}
            
            if start_date and end_date:
                result = self.data.loc[start_date:end_date, asset_cols]
            else:
                result = self.data[asset_cols]
            
            self.search_history.append({
                'type': 'asset_search',
                'query': asset_name,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'columns_found': asset_cols
            })
            
            return {
                'success': True,
                'data': result,
                'asset': asset_name,
                'columns': asset_cols
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def search_by_geopolitical_term(self, term, start_date=None, end_date=None):
        """Search data by geopolitical term"""
        try:
            term_cols = [col for col in self.data.columns if term in col]
            
            if not term_cols:
                return {'success': False, 'error': f'Term "{term}" not found'}
            
            if start_date and end_date:
                result = self.data.loc[start_date:end_date, term_cols]
            else:
                result = self.data[term_cols]
            
            self.search_history.append({
                'type': 'geopolitical_search',
                'query': term,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'columns_found': term_cols
            })
            
            return {
                'success': True,
                'data': result,
                'term': term,
                'columns': term_cols
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def search_by_value_threshold(self, column, operator, value):
        """Search data by value threshold"""
        try:
            if column not in self.data.columns:
                return {'success': False, 'error': f'Column "{column}" not found'}
            
            if operator == '>':
                result = self.data[self.data[column] > value]
            elif operator == '<':
                result = self.data[self.data[column] < value]
            elif operator == '>=':
                result = self.data[self.data[column] >= value]
            elif operator == '<=':
                result = self.data[self.data[column] <= value]
            elif operator == '==':
                result = self.data[self.data[column] == value]
            else:
                return {'success': False, 'error': f'Invalid operator: {operator}'}
            
            self.search_history.append({
                'type': 'threshold_search',
                'query': f'{column} {operator} {value}',
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'records_found': len(result)
            })
            
            return {
                'success': True,
                'data': result,
                'query': f'{column} {operator} {value}',
                'records': len(result)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def search_anomalies(self, column, std_threshold=2):
        """Search for anomalies in a specific column"""
        try:
            if column not in self.data.columns:
                return {'success': False, 'error': f'Column "{column}" not found'}
            
            col_data = self.data[column]
            mean = col_data.mean()
            std = col_data.std()
            
            upper_bound = mean + (std_threshold * std)
            lower_bound = mean - (std_threshold * std)
            
            anomalies = self.data[(col_data > upper_bound) | (col_data < lower_bound)]
            
            return {
                'success': True,
                'data': anomalies,
                'column': column,
                'mean': mean,
                'std': std,
                'threshold': std_threshold,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'anomalies_count': len(anomalies)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def search_correlation(self, column1, column2):
        """Find correlation between two columns"""
        try:
            if column1 not in self.data.columns:
                return {'success': False, 'error': f'Column "{column1}" not found'}
            if column2 not in self.data.columns:
                return {'success': False, 'error': f'Column "{column2}" not found'}
            
            correlation = self.data[column1].corr(self.data[column2])
            
            return {
                'success': True,
                'column1': column1,
                'column2': column2,
                'correlation': correlation,
                'strength': 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def free_text_search(self, query):
        """Perform free text search across column names"""
        query_lower = query.lower()
        matching_columns = [col for col in self.data.columns if query_lower in col.lower()]
        
        if not matching_columns:
            return {'success': False, 'error': f'No columns matching "{query}"'}
        
        return {
            'success': True,
            'query': query,
            'matching_columns': matching_columns,
            'data': self.data[matching_columns].tail(30)  # Return last 30 rows
        }
    
    def get_statistics(self, column):
        """Get detailed statistics for a column"""
        try:
            if column not in self.data.columns:
                return {'success': False, 'error': f'Column "{column}" not found'}
            
            col_data = self.data[column].dropna()
            
            return {
                'success': True,
                'column': column,
                'statistics': {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_search_history(self):
        """Get search history"""
        return self.search_history
    
    def clear_search_history(self):
        """Clear search history"""
        self.search_history = []
        return True


class ManualDataInput:
    """Handle manual data input and custom analysis"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.custom_data = pd.DataFrame()
        self.custom_analyses = []
    
    def add_custom_data_point(self, date, column, value):
        """Add a custom data point for what-if analysis"""
        try:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            if column not in self.data_processor.df.columns:
                return {'success': False, 'error': f'Column "{column}" not found in dataset'}
            
            # Store in custom data
            if date not in self.custom_data.index:
                self.custom_data.loc[date] = {}
            
            self.custom_data.loc[date, column] = value
            
            return {
                'success': True,
                'message': f'Added {column}={value} for {date.date()}'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_scenario(self, scenario_name, changes):
        """Create a what-if scenario with multiple changes"""
        try:
            scenario_data = self.data_processor.df.copy()
            
            for change in changes:
                date = pd.to_datetime(change['date'])
                column = change['column']
                value = change['value']
                
                if date in scenario_data.index and column in scenario_data.columns:
                    scenario_data.loc[date, column] = value
            
            self.custom_analyses.append({
                'name': scenario_name,
                'changes': changes,
                'data': scenario_data,
                'created_at': datetime.now().isoformat()
            })
            
            return {
                'success': True,
                'scenario_name': scenario_name,
                'changes_applied': len(changes)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def compare_scenarios(self, scenario1_name, scenario2_name, columns):
        """Compare two scenarios"""
        scenario1 = None
        scenario2 = None
        
        for analysis in self.custom_analyses:
            if analysis['name'] == scenario1_name:
                scenario1 = analysis['data']
            if analysis['name'] == scenario2_name:
                scenario2 = analysis['data']
        
        if scenario1 is None:
            return {'success': False, 'error': f'Scenario "{scenario1_name}" not found'}
        if scenario2 is None:
            return {'success': False, 'error': f'Scenario "{scenario2_name}" not found'}
        
        comparison = {}
        for col in columns:
            if col in scenario1.columns and col in scenario2.columns:
                comparison[col] = {
                    'scenario1_mean': scenario1[col].mean(),
                    'scenario2_mean': scenario2[col].mean(),
                    'difference': scenario2[col].mean() - scenario1[col].mean()
                }
        
        return {
            'success': True,
            'comparison': comparison
        }
    
    def get_custom_data(self):
        """Get all custom data"""
        return self.custom_data
    
    def get_scenarios(self):
        """Get all scenarios"""
        return [{'name': a['name'], 'created_at': a['created_at'], 'changes': len(a['changes'])} 
                for a in self.custom_analyses]
    
    def clear_custom_data(self):
        """Clear custom data"""
        self.custom_data = pd.DataFrame()
        self.custom_analyses = []
        return True