import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class GeopoliticalAIAgent:
    def __init__(self, data, ml_models, data_processor):
        self.data = data
        self.ml_models = ml_models
        self.data_processor = data_processor
        self.memory = []
        self.insights = []
        
    def analyze_geopolitical_trends(self, lookback_days=30):
        """Agent: Analyze geopolitical trends"""
        recent_data = self.data.tail(lookback_days)
        geopolitical_cols = self.data_processor.get_geopolitical_features()
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'lookback_days': lookback_days,
            'trends': {},
            'alerts': []
        }
        
        for col in geopolitical_cols.columns:
            if '_Shock' in col:
                latest = recent_data[col].iloc[-1]
                mean = recent_data[col].mean()
                std = recent_data[col].std()
                
                z_score = (latest - mean) / std if std > 0 else 0
                
                analysis['trends'][col] = {
                    'latest': float(latest),
                    'mean': float(mean),
                    'std': float(std),
                    'z_score': float(z_score),
                    'status': 'HIGH' if z_score > 2 else 'MEDIUM' if z_score > 1 else 'LOW'
                }
                
                if z_score > 2:
                    analysis['alerts'].append(f"🚨 High geopolitical shock detected in {col}: Z-score = {z_score:.2f}")
        
        self.memory.append(analysis)
        return analysis
    
    def detect_market_financial_linkage(self):
        """Agent: Detect correlation between geopolitical events and market movements"""
        financial_cols = self.data_processor.get_financial_features()
        geopolitical_cols = self.data_processor.get_geopolitical_features()
        
        # Focus on Close prices and Shock metrics
        close_cols = [col for col in financial_cols.columns if '_Close' in col]
        shock_cols = [col for col in geopolitical_cols.columns if '_Shock' in col]
        
        linkage_analysis = {
            'timestamp': datetime.now().isoformat(),
            'correlations': {}
        }
        
        for geopolitical in shock_cols:
            for financial in close_cols:
                try:
                    corr = self.data[geopolitical].corr(self.data[financial])
                    if abs(corr) > 0.3:  # Significant correlation
                        linkage_analysis['correlations'][f"{geopolitical} → {financial}"] = {
                            'correlation': float(corr),
                            'strength': 'STRONG' if abs(corr) > 0.6 else 'MODERATE'
                        }
                except:
                    pass
        
        self.insights.append(linkage_analysis)
        return linkage_analysis
    
    def predict_market_stress(self, days_ahead=5):
        """Agent: Predict market stress scenarios"""
        recent_shock_cols = [col for col in self.data.columns if '_Shock' in col]
        recent_data = self.data[recent_shock_cols].tail(20)
        
        stress_prediction = {
            'timestamp': datetime.now().isoformat(),
            'days_ahead': days_ahead,
            'stress_scenarios': [],
            'risk_level': 'LOW'
        }
        
        # Simple heuristic: if average shock is high
        avg_shock = recent_data.mean().mean()
        max_shock = recent_data.max().max()
        
        if max_shock > recent_data.std().mean() * 3:
            stress_prediction['risk_level'] = 'CRITICAL'
            stress_prediction['stress_scenarios'].append({
                'scenario': 'Extreme Geopolitical Volatility',
                'probability': 0.75,
                'impact': 'HIGH',
                'recommendation': 'Reduce exposure to Defense & Commodity sectors'
            })
        elif avg_shock > recent_data.mean().mean():
            stress_prediction['risk_level'] = 'HIGH'
            stress_prediction['stress_scenarios'].append({
                'scenario': 'Elevated Geopolitical Tension',
                'probability': 0.55,
                'impact': 'MEDIUM',
                'recommendation': 'Monitor NATO, Taiwan Strait, Cyberwarfare metrics'
            })
        else:
            stress_prediction['risk_level'] = 'LOW'
            stress_prediction['stress_scenarios'].append({
                'scenario': 'Stable Geopolitical Environment',
                'probability': 0.85,
                'impact': 'LOW',
                'recommendation': 'Neutral positioning acceptable'
            })
        
        self.insights.append(stress_prediction)
        return stress_prediction
    
    def anomaly_alert_system(self):
        """Agent: Real-time anomaly detection and alerts"""
        anomalies = self.ml_models.detect_geopolitical_anomalies(self.data)
        
        alert_system = {
            'timestamp': datetime.now().isoformat(),
            'total_anomalies_detected': anomalies['anomaly_scores'].count(-1),
            'recent_anomalies': [],
            'action_items': []
        }
        
        # Get last 5 anomalies
        if len(anomalies['anomaly_dates']) > 0:
            recent_anomaly_dates = anomalies['anomaly_dates'][-5:]
            for date in recent_anomaly_dates:
                alert_system['recent_anomalies'].append({
                    'date': str(date),
                    'type': 'Geopolitical Shock Spike',
                    'severity': 'HIGH'
                })
                alert_system['action_items'].append(f"Review portfolio exposure on {date}")
        
        return alert_system
    
    def sector_recommendation_engine(self):
        """Agent: Recommend sectors based on geopolitical analysis"""
        recent_geo = self.data_processor.get_geopolitical_features().tail(1)
        
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'defense_sector': {'rating': 'NEUTRAL', 'reasoning': ''},
            'commodities': {'rating': 'NEUTRAL', 'reasoning': ''},
            'indices': {'rating': 'NEUTRAL', 'reasoning': ''}
        }
        
        # Analyze NATO, Cyberwarfare for Defense
        if 'NATO_Shock' in recent_geo.columns and recent_geo['NATO_Shock'].values[0] > 0:
            recommendations['defense_sector'] = {
                'rating': 'BUY',
                'reasoning': 'NATO tensions rising - Defense contractor tailwinds'
            }
        
        # Analyze OPEC, Economic Sanctions for Commodities
        if 'OPEC_Shock' in recent_geo.columns and recent_geo['OPEC_Shock'].values[0] > 0:
            recommendations['commodities'] = {
                'rating': 'BUY',
                'reasoning': 'Oil market volatility - Energy commodity spike likely'
            }
        
        # Analyze Recession, Inflation for Indices
        if 'Recession_Shock' in recent_geo.columns:
            if recent_geo['Recession_Shock'].values[0] > 1:
                recommendations['indices'] = {
                    'rating': 'SELL',
                    'reasoning': 'Recession signals - Market downside risk'
                }
        
        return recommendations
    
    def executive_summary(self):
        """Agent: Generate executive summary"""
        summary = {
            'report_date': datetime.now().isoformat(),
            'data_coverage': f"{self.data.index[0].date()} to {self.data.index[-1].date()}",
            'total_features': self.data.shape[1],
            'total_observations': self.data.shape[0],
            'memory_events': len(self.memory),
            'insights_generated': len(self.insights),
            'key_findings': [],
            'top_risks': [],
            'opportunities': []
        }
        
        # Compile findings
        latest_analysis = self.analyze_geopolitical_trends(30)
        summary['key_findings'].append(f"Analyzed {len(latest_analysis['trends'])} geopolitical metrics")
        summary['key_findings'].extend(latest_analysis['alerts'][:3])
        
        # Identify top risks
        for metric, data in latest_analysis['trends'].items():
            if data['status'] == 'HIGH':
                summary['top_risks'].append(f"{metric}: Z-score = {data['z_score']:.2f}")
        
        # Identify opportunities
        recommendations = self.sector_recommendation_engine()
        for sector, data in recommendations.items():
            if data['rating'] == 'BUY':
                summary['opportunities'].append(f"{sector.upper()}: {data['reasoning']}")
        
        return summary
    
    def get_memory_log(self):
        """Retrieve agent memory log"""
        return self.memory
    
    def get_insights_log(self):
        """Retrieve insights log"""
        return self.insights