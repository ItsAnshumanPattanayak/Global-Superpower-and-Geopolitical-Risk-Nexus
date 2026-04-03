import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json

class GeopoliticalChatbot:
    """Interactive AI Chatbot for Geopolitical Risk Analysis"""
    
    def __init__(self, data_processor, ml_models, agent):
        self.data_processor = data_processor
        self.ml_models = ml_models
        self.agent = agent
        self.data = data_processor.df
        self.conversation_history = []
        self.context = {}
        
        # Define intent patterns
        self.intents = {
            'greeting': r'\b(hi|hello|hey|greetings|good morning|good afternoon)\b',
            'help': r'\b(help|what can you do|commands|options|guide)\b',
            'risk_analysis': r'\b(risk|danger|threat|alert|warning)\b',
            'market_analysis': r'\b(market|stock|price|financial|trading|asset)\b',
            'geopolitical': r'\b(geopolitical|nato|china|russia|taiwan|war|conflict|sanction)\b',
            'prediction': r'\b(predict|forecast|future|next|upcoming|expect)\b',
            'anomaly': r'\b(anomaly|unusual|strange|outlier|abnormal)\b',
            'recommendation': r'\b(recommend|suggest|advice|should|strategy)\b',
            'data_query': r'\b(show|display|get|find|search|data|value|price on)\b',
            'comparison': r'\b(compare|versus|vs|difference|between)\b',
            'trend': r'\b(trend|pattern|movement|direction|going)\b',
            'summary': r'\b(summary|overview|brief|report|status)\b',
            'specific_date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}',
            'time_period': r'\b(today|yesterday|last week|last month|this week|this month|year)\b',
            'asset_query': r'\b(gold|oil|sp500|nasdaq|dji|defense|lockheed|raytheon|northrop|boeing)\b',
            'goodbye': r'\b(bye|goodbye|exit|quit|thanks|thank you)\b'
        }
        
        # Asset name mappings
        self.asset_mappings = {
            'gold': 'Gold',
            'oil': 'Oil',
            'sp500': 'SP500',
            's&p': 'SP500',
            'nasdaq': 'Nasdaq',
            'dji': 'DJI',
            'dow': 'DJI',
            'lockheed': 'LMT',
            'raytheon': 'RTX',
            'northrop': 'NOC',
            'boeing': 'BA',
            'general dynamics': 'GD'
        }
        
        # Geopolitical term mappings
        self.geopolitical_mappings = {
            'nato': 'NATO',
            'china': 'China_US_Trade_War',
            'russia': 'Russia_Ukraine_War',
            'taiwan': 'Taiwan_Strait',
            'opec': 'OPEC',
            'sanctions': 'Economic_Sanctions',
            'cyber': 'Cyberwarfare',
            'nuclear': 'Nuclear_Proliferation',
            'recession': 'Recession',
            'inflation': 'Inflation'
        }
    
    def detect_intent(self, user_input):
        """Detect user intent from input"""
        user_input_lower = user_input.lower()
        detected_intents = []
        
        for intent, pattern in self.intents.items():
            if re.search(pattern, user_input_lower):
                detected_intents.append(intent)
        
        return detected_intents if detected_intents else ['general']
    
    def extract_entities(self, user_input):
        """Extract entities like dates, assets, geopolitical terms"""
        entities = {
            'dates': [],
            'assets': [],
            'geopolitical_terms': [],
            'time_periods': []
        }
        
        user_input_lower = user_input.lower()
        
        # Extract dates (YYYY-MM-DD format)
        date_patterns = re.findall(r'\d{4}-\d{2}-\d{2}', user_input)
        entities['dates'].extend(date_patterns)
        
        # Extract date patterns (MM/DD/YYYY)
        date_patterns2 = re.findall(r'\d{2}/\d{2}/\d{4}', user_input)
        for dp in date_patterns2:
            try:
                parsed = datetime.strptime(dp, '%m/%d/%Y').strftime('%Y-%m-%d')
                entities['dates'].append(parsed)
            except:
                pass
        
        # Extract assets
        for key, value in self.asset_mappings.items():
            if key in user_input_lower:
                entities['assets'].append(value)
        
        # Extract geopolitical terms
        for key, value in self.geopolitical_mappings.items():
            if key in user_input_lower:
                entities['geopolitical_terms'].append(value)
        
        # Extract time periods
        time_patterns = ['today', 'yesterday', 'last week', 'last month', 'this week', 'this month', 'this year', 'last year']
        for tp in time_patterns:
            if tp in user_input_lower:
                entities['time_periods'].append(tp)
        
        return entities
    
    def get_date_range(self, time_period):
        """Convert time period to date range"""
        today = self.data.index[-1]  # Use last date in dataset
        
        if time_period == 'today':
            return today, today
        elif time_period == 'yesterday':
            return today - timedelta(days=1), today - timedelta(days=1)
        elif time_period == 'last week':
            return today - timedelta(days=7), today
        elif time_period == 'this week':
            return today - timedelta(days=today.weekday()), today
        elif time_period == 'last month':
            return today - timedelta(days=30), today
        elif time_period == 'this month':
            return today.replace(day=1), today
        elif time_period == 'this year':
            return today.replace(month=1, day=1), today
        elif time_period == 'last year':
            return today - timedelta(days=365), today
        else:
            return today - timedelta(days=30), today
    
    def process_message(self, user_input):
        """Process user message and generate response"""
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'message': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Detect intent and extract entities
        intents = self.detect_intent(user_input)
        entities = self.extract_entities(user_input)
        
        # Store context
        self.context['last_intents'] = intents
        self.context['last_entities'] = entities
        
        # Generate response based on intent
        response = self.generate_response(intents, entities, user_input)
        
        # Add response to history
        self.conversation_history.append({
            'role': 'assistant',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def generate_response(self, intents, entities, user_input):
        """Generate response based on detected intents and entities"""
        
        # Handle greeting
        if 'greeting' in intents:
            return self.handle_greeting()
        
        # Handle goodbye
        if 'goodbye' in intents:
            return self.handle_goodbye()
        
        # Handle help
        if 'help' in intents:
            return self.handle_help()
        
        # Handle data query
        if 'data_query' in intents:
            return self.handle_data_query(entities, user_input)
        
        # Handle risk analysis
        if 'risk_analysis' in intents:
            return self.handle_risk_analysis(entities)
        
        # Handle market analysis
        if 'market_analysis' in intents:
            return self.handle_market_analysis(entities)
        
        # Handle geopolitical queries
        if 'geopolitical' in intents:
            return self.handle_geopolitical_query(entities)
        
        # Handle predictions
        if 'prediction' in intents:
            return self.handle_prediction(entities)
        
        # Handle anomaly detection
        if 'anomaly' in intents:
            return self.handle_anomaly_query()
        
        # Handle recommendations
        if 'recommendation' in intents:
            return self.handle_recommendation()
        
        # Handle comparison
        if 'comparison' in intents:
            return self.handle_comparison(entities, user_input)
        
        # Handle trend analysis
        if 'trend' in intents:
            return self.handle_trend_analysis(entities)
        
        # Handle summary
        if 'summary' in intents:
            return self.handle_summary()
        
        # Handle specific asset query
        if 'asset_query' in intents and entities['assets']:
            return self.handle_asset_query(entities['assets'])
        
        # Default response
        return self.handle_unknown()
    
    def handle_greeting(self):
        """Handle greeting intent"""
        return """👋 **Hello! Welcome to the Geopolitical Risk Analysis Chatbot!**

I'm your AI assistant for analyzing global superpower dynamics and geopolitical risks. Here's what I can help you with:

🔹 **Risk Analysis** - Current geopolitical risk levels
🔹 **Market Data** - Financial asset prices and trends
🔹 **Predictions** - AI-powered forecasts
🔹 **Anomaly Detection** - Unusual market/geopolitical events
🔹 **Recommendations** - Strategic sector recommendations

Type **'help'** for a full list of commands, or just ask me anything about geopolitical risks and markets!"""
    
    def handle_goodbye(self):
        """Handle goodbye intent"""
        return """👋 **Thank you for using the Geopolitical Risk Analysis Chatbot!**

Stay informed and stay ahead of global risks. See you next time!

📊 *Remember: Geopolitical awareness is your strategic advantage.*"""
    
    def handle_help(self):
        """Handle help intent"""
        return """📚 **Chatbot Command Guide**

**📊 Data Queries:**
• "Show me gold price on 2024-01-15"
• "Get SP500 data for last week"
• "What's the current oil price?"

**🌐 Geopolitical Analysis:**
• "What's the NATO risk level?"
• "Analyze China-US trade tensions"
• "Show Russia-Ukraine conflict impact"

**📈 Market Analysis:**
• "How is the defense sector performing?"
• "Compare gold vs oil trends"
• "Show market volatility"

**🔮 Predictions:**
• "Predict market stress for next 5 days"
• "What's the risk forecast?"

**🚨 Anomalies:**
• "Detect any anomalies"
• "Show unusual market events"

**💡 Recommendations:**
• "What sectors should I invest in?"
• "Give me strategic recommendations"

**📋 Reports:**
• "Generate summary report"
• "Show executive overview"

**💬 Example Questions:**
• "What are the top geopolitical risks right now?"
• "How did sanctions affect oil prices?"
• "Is there any unusual activity in defense stocks?"

Just type your question naturally - I'll understand! 🤖"""
    
    def handle_data_query(self, entities, user_input):
        """Handle data query intent"""
        response = "📊 **Data Query Results:**\n\n"
        
        # If specific dates requested
        if entities['dates']:
            date_str = entities['dates'][0]
            try:
                query_date = pd.to_datetime(date_str)
                if query_date in self.data.index:
                    row = self.data.loc[query_date]
                    
                    # If specific assets mentioned
                    if entities['assets']:
                        for asset in entities['assets']:
                            close_col = f"{asset}_Close"
                            if close_col in row.index:
                                response += f"**{asset}** on {date_str}:\n"
                                response += f"  • Close: ${row[close_col]:,.2f}\n"
                                if f"{asset}_Volume" in row.index:
                                    response += f"  • Volume: {row[f'{asset}_Volume']:,.0f}\n"
                    else:
                        # Show summary of all close prices
                        close_cols = [col for col in row.index if '_Close' in col][:5]
                        response += f"**Market Snapshot on {date_str}:**\n"
                        for col in close_cols:
                            asset = col.replace('_Close', '')
                            response += f"  • {asset}: ${row[col]:,.2f}\n"
                else:
                    response += f"⚠️ No data available for {date_str}. The date might be a weekend or holiday."
            except Exception as e:
                response += f"⚠️ Error processing date: {str(e)}"
        
        # If time period requested
        elif entities['time_periods']:
            period = entities['time_periods'][0]
            start_date, end_date = self.get_date_range(period)
            period_data = self.data.loc[start_date:end_date]
            
            if not period_data.empty:
                response += f"**Data for {period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}):**\n\n"
                
                if entities['assets']:
                    for asset in entities['assets']:
                        close_col = f"{asset}_Close"
                        if close_col in period_data.columns:
                            start_price = period_data[close_col].iloc[0]
                            end_price = period_data[close_col].iloc[-1]
                            change = ((end_price - start_price) / start_price) * 100
                            response += f"**{asset}:**\n"
                            response += f"  • Start: ${start_price:,.2f}\n"
                            response += f"  • End: ${end_price:,.2f}\n"
                            response += f"  • Change: {change:+.2f}%\n\n"
                else:
                    response += "Please specify an asset (e.g., 'gold', 'oil', 'SP500')"
            else:
                response += "⚠️ No data available for the specified period."
        
        # If only assets mentioned
        elif entities['assets']:
            response += f"**Latest Data (as of {self.data.index[-1].strftime('%Y-%m-%d')}):**\n\n"
            latest = self.data.iloc[-1]
            
            for asset in entities['assets']:
                close_col = f"{asset}_Close"
                if close_col in latest.index:
                    response += f"**{asset}:**\n"
                    response += f"  • Close: ${latest[close_col]:,.2f}\n"
                    
                    # Calculate 7-day change
                    if len(self.data) > 7:
                        week_ago = self.data.iloc[-7][close_col]
                        change = ((latest[close_col] - week_ago) / week_ago) * 100
                        response += f"  • 7-Day Change: {change:+.2f}%\n"
                    response += "\n"
        else:
            response += "Please specify what data you'd like to see. For example:\n"
            response += "• 'Show gold price for last week'\n"
            response += "• 'Get SP500 data on 2024-06-15'\n"
            response += "• 'What's the current oil price?'"
        
        return response
    
    def handle_risk_analysis(self, entities):
        """Handle risk analysis intent"""
        analysis = self.agent.analyze_geopolitical_trends(30)
        
        response = "🚨 **Geopolitical Risk Analysis**\n\n"
        
        # Count risk levels
        high_risk = sum(1 for t in analysis['trends'].values() if t['status'] == 'HIGH')
        medium_risk = sum(1 for t in analysis['trends'].values() if t['status'] == 'MEDIUM')
        low_risk = sum(1 for t in analysis['trends'].values() if t['status'] == 'LOW')
        
        response += f"**Risk Distribution (30-Day Analysis):**\n"
        response += f"  🔴 High Risk: {high_risk} metrics\n"
        response += f"  🟠 Medium Risk: {medium_risk} metrics\n"
        response += f"  🟢 Low Risk: {low_risk} metrics\n\n"
        
        # Show alerts
        if analysis['alerts']:
            response += "**⚠️ Active Alerts:**\n"
            for alert in analysis['alerts'][:5]:
                response += f"  • {alert}\n"
        else:
            response += "✅ No critical alerts at this time.\n"
        
        # If specific geopolitical terms mentioned
        if entities['geopolitical_terms']:
            response += "\n**Specific Risk Metrics:**\n"
            for term in entities['geopolitical_terms']:
                shock_col = f"{term}_Shock"
                if shock_col in analysis['trends']:
                    trend = analysis['trends'][shock_col]
                    response += f"\n**{term}:**\n"
                    response += f"  • Current Level: {trend['latest']:.2f}\n"
                    response += f"  • Z-Score: {trend['z_score']:.2f}\n"
                    response += f"  • Status: {trend['status']}\n"
        
        return response
    
    def handle_market_analysis(self, entities):
        """Handle market analysis intent"""
        response = "📈 **Market Analysis**\n\n"
        
        financial_data = self.data_processor.get_financial_features()
        close_cols = [col for col in financial_data.columns if '_Close' in col]
        
        # Get last 30 days performance
        recent_data = financial_data[close_cols].tail(30)
        
        response += "**30-Day Market Performance:**\n\n"
        
        # If specific assets mentioned
        assets_to_show = entities['assets'] if entities['assets'] else ['SP500', 'Gold', 'Oil', 'DJI', 'Nasdaq']
        
        for asset in assets_to_show:
            close_col = f"{asset}_Close"
            if close_col in recent_data.columns:
                start = recent_data[close_col].iloc[0]
                end = recent_data[close_col].iloc[-1]
                change = ((end - start) / start) * 100
                volatility = recent_data[close_col].std() / recent_data[close_col].mean() * 100
                
                trend_emoji = "📈" if change > 0 else "📉"
                response += f"{trend_emoji} **{asset}:**\n"
                response += f"  • Current: ${end:,.2f}\n"
                response += f"  • 30-Day Change: {change:+.2f}%\n"
                response += f"  • Volatility: {volatility:.2f}%\n\n"
        
        return response
    
    def handle_geopolitical_query(self, entities):
        """Handle geopolitical query intent"""
        response = "🌐 **Geopolitical Intelligence Report**\n\n"
        
        geo_data = self.data_processor.get_geopolitical_features()
        
        # If specific terms mentioned
        if entities['geopolitical_terms']:
            for term in entities['geopolitical_terms']:
                views_col = f"{term}_Views"
                shock_col = f"{term}_Shock"
                momentum_col = f"{term}_Momentum"
                
                response += f"**{term.replace('_', ' ')}:**\n"
                
                if views_col in geo_data.columns:
                    recent_views = geo_data[views_col].tail(7)
                    avg_views = recent_views.mean()
                    trend = "Rising 📈" if recent_views.iloc[-1] > recent_views.iloc[0] else "Declining 📉"
                    response += f"  • 7-Day Avg Views: {avg_views:,.0f}\n"
                    response += f"  • Trend: {trend}\n"
                
                if shock_col in geo_data.columns:
                    latest_shock = geo_data[shock_col].iloc[-1]
                    response += f"  • Current Shock Level: {latest_shock:.2f}\n"
                
                response += "\n"
        else:
            # Show top geopolitical metrics
            response += "**Top Geopolitical Attention (Last 7 Days):**\n\n"
            
            views_cols = [col for col in geo_data.columns if '_Views' in col]
            recent_avg = geo_data[views_cols].tail(7).mean().sort_values(ascending=False)
            
            for col in recent_avg.head(5).index:
                term = col.replace('_Views', '')
                response += f"  • {term}: {recent_avg[col]:,.0f} avg views/day\n"
        
        return response
    
    def handle_prediction(self, entities):
        """Handle prediction intent"""
        stress = self.agent.predict_market_stress(days_ahead=5)
        
        response = "🔮 **Market Stress Prediction (5-Day Forecast)**\n\n"
        
        risk_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
        
        response += f"**Overall Risk Level:** {risk_emoji.get(stress['risk_level'], '❓')} {stress['risk_level']}\n\n"
        
        response += "**Scenario Analysis:**\n"
        for scenario in stress['stress_scenarios']:
            response += f"\n📌 **{scenario['scenario']}**\n"
            response += f"  • Probability: {scenario['probability']:.0%}\n"
            response += f"  • Impact: {scenario['impact']}\n"
            response += f"  • 💡 Recommendation: {scenario['recommendation']}\n"
        
        return response
    
    def handle_anomaly_query(self):
        """Handle anomaly detection query"""
        anomalies = self.agent.anomaly_alert_system()
        
        response = "🚨 **Anomaly Detection Report**\n\n"
        
        response += f"**Detection Summary:**\n"
        response += f"  • Total Anomalies Detected: {anomalies['total_anomalies_detected']}\n"
        response += f"  • Recent Anomalies: {len(anomalies['recent_anomalies'])}\n\n"
        
        if anomalies['recent_anomalies']:
            response += "**Recent Anomalous Events:**\n"
            for anomaly in anomalies['recent_anomalies']:
                response += f"  ⚠️ {anomaly['date']} - {anomaly['type']} (Severity: {anomaly['severity']})\n"
            
            response += "\n**Recommended Actions:**\n"
            for action in anomalies['action_items'][:3]:
                response += f"  ✅ {action}\n"
        else:
            response += "✅ No significant anomalies detected in recent data."
        
        return response
    
    def handle_recommendation(self):
        """Handle recommendation intent"""
        recommendations = self.agent.sector_recommendation_engine()
        
        response = "💡 **Strategic Sector Recommendations**\n\n"
        
        rating_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'NEUTRAL': '⚪'}
        
        for sector, data in recommendations.items():
            if sector != 'timestamp':
                emoji = rating_emoji.get(data['rating'], '❓')
                response += f"{emoji} **{sector.replace('_', ' ').title()}:** {data['rating']}\n"
                if data['reasoning']:
                    response += f"   📝 {data['reasoning']}\n"
                response += "\n"
        
        response += "\n*Recommendations based on current geopolitical analysis and market conditions.*"
        
        return response
    
    def handle_comparison(self, entities, user_input):
        """Handle comparison intent"""
        response = "📊 **Comparison Analysis**\n\n"
        
        if len(entities['assets']) >= 2:
            assets = entities['assets'][:2]
            recent_data = self.data.tail(30)
            
            response += f"**{assets[0]} vs {assets[1]} (30-Day Comparison):**\n\n"
            
            for asset in assets:
                close_col = f"{asset}_Close"
                if close_col in recent_data.columns:
                    start = recent_data[close_col].iloc[0]
                    end = recent_data[close_col].iloc[-1]
                    change = ((end - start) / start) * 100
                    volatility = recent_data[close_col].std() / recent_data[close_col].mean() * 100
                    
                    response += f"**{asset}:**\n"
                    response += f"  • Change: {change:+.2f}%\n"
                    response += f"  • Volatility: {volatility:.2f}%\n\n"
            
            # Correlation
            col1 = f"{assets[0]}_Close"
            col2 = f"{assets[1]}_Close"
            if col1 in recent_data.columns and col2 in recent_data.columns:
                corr = recent_data[col1].corr(recent_data[col2])
                response += f"**Correlation:** {corr:.2f}\n"
                if corr > 0.7:
                    response += "↗️ Strong positive correlation - assets move together\n"
                elif corr < -0.7:
                    response += "↘️ Strong negative correlation - assets move inversely\n"
                else:
                    response += "↔️ Weak correlation - assets move independently\n"
        else:
            response += "Please specify two assets to compare. For example:\n"
            response += "'Compare gold vs oil' or 'Gold versus SP500'"
        
        return response
    
    def handle_trend_analysis(self, entities):
        """Handle trend analysis intent"""
        response = "📈 **Trend Analysis**\n\n"
        
        assets = entities['assets'] if entities['assets'] else ['SP500', 'Gold', 'Oil']
        geo_terms = entities['geopolitical_terms'] if entities['geopolitical_terms'] else []
        
        # Financial trends
        if assets:
            response += "**Financial Asset Trends (30-Day):**\n"
            recent_data = self.data.tail(30)
            
            for asset in assets[:3]:
                close_col = f"{asset}_Close"
                if close_col in recent_data.columns:
                    prices = recent_data[close_col]
                    
                    # Simple trend detection
                    sma_10 = prices.rolling(10).mean().iloc[-1]
                    sma_20 = prices.rolling(20).mean().iloc[-1]
                    current = prices.iloc[-1]
                    
                    if current > sma_10 > sma_20:
                        trend = "📈 Strong Uptrend"
                    elif current > sma_10:
                        trend = "↗️ Mild Uptrend"
                    elif current < sma_10 < sma_20:
                        trend = "📉 Strong Downtrend"
                    elif current < sma_10:
                        trend = "↘️ Mild Downtrend"
                    else:
                        trend = "↔️ Sideways"
                    
                    response += f"  • {asset}: {trend}\n"
            
            response += "\n"
        
        # Geopolitical trends
        if geo_terms:
            response += "**Geopolitical Attention Trends:**\n"
            geo_data = self.data_processor.get_geopolitical_features()
            recent_geo = geo_data.tail(30)
            
            for term in geo_terms:
                views_col = f"{term}_Views"
                if views_col in recent_geo.columns:
                    views = recent_geo[views_col]
                    recent_avg = views.tail(7).mean()
                    earlier_avg = views.head(7).mean()
                    
                    if recent_avg > earlier_avg * 1.2:
                        trend = "📈 Rising Attention"
                    elif recent_avg < earlier_avg * 0.8:
                        trend = "📉 Declining Attention"
                    else:
                        trend = "↔️ Stable"
                    
                    response += f"  • {term.replace('_', ' ')}: {trend}\n"
        
        return response
    
    def handle_summary(self):
        """Handle summary intent"""
        summary = self.agent.executive_summary()
        
        response = "📋 **Executive Summary**\n\n"
        
        response += f"**Report Date:** {summary['report_date'][:10]}\n"
        response += f"**Data Coverage:** {summary['data_coverage']}\n\n"
        
        response += "**📊 Key Metrics:**\n"
        response += f"  • Total Observations: {summary['total_observations']:,}\n"
        response += f"  • Features Analyzed: {summary['total_features']}\n"
        response += f"  • Memory Events: {summary['memory_events']}\n"
        response += f"  • Insights Generated: {summary['insights_generated']}\n\n"
        
        if summary['key_findings']:
            response += "**🔍 Key Findings:**\n"
            for finding in summary['key_findings'][:3]:
                response += f"  • {finding}\n"
            response += "\n"
        
        if summary['top_risks']:
            response += "**⚠️ Top Risks:**\n"
            for risk in summary['top_risks'][:3]:
                response += f"  • {risk}\n"
            response += "\n"
        
        if summary['opportunities']:
            response += "**🚀 Opportunities:**\n"
            for opp in summary['opportunities'][:3]:
                response += f"  • {opp}\n"
        
        return response
    
    def handle_asset_query(self, assets):
        """Handle specific asset query"""
        response = "📊 **Asset Analysis**\n\n"
        
        latest = self.data.iloc[-1]
        week_ago = self.data.iloc[-7] if len(self.data) > 7 else self.data.iloc[0]
        month_ago = self.data.iloc[-30] if len(self.data) > 30 else self.data.iloc[0]
        
        for asset in assets:
            close_col = f"{asset}_Close"
            if close_col in latest.index:
                current = latest[close_col]
                weekly_change = ((current - week_ago[close_col]) / week_ago[close_col]) * 100
                monthly_change = ((current - month_ago[close_col]) / month_ago[close_col]) * 100
                
                response += f"**{asset}**\n"
                response += f"  💰 Current Price: ${current:,.2f}\n"
                response += f"  📅 7-Day Change: {weekly_change:+.2f}%\n"
                response += f"  📆 30-Day Change: {monthly_change:+.2f}%\n"
                
                if f"{asset}_Volume" in latest.index:
                    response += f"  📊 Volume: {latest[f'{asset}_Volume']:,.0f}\n"
                
                response += "\n"
        
        return response
    
    def handle_unknown(self):
        """Handle unknown intent"""
        return """🤔 I'm not sure I understood that. Here are some things you can ask me:

**📊 Data Questions:**
• "What's the gold price?"
• "Show SP500 data for last week"

**🌐 Geopolitical Analysis:**
• "What are the current risk levels?"
• "Analyze NATO tensions"

**📈 Market Insights:**
• "How are markets performing?"
• "Compare gold vs oil"

**🔮 Predictions:**
• "Predict market stress"
• "What's the risk forecast?"

**💡 Recommendations:**
• "What sectors should I invest in?"

Type **'help'** for a complete list of commands!"""
    
    def get_conversation_history(self):
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.context = {}
        return "🔄 Conversation history cleared."
    
    def get_quick_stats(self):
        """Get quick statistics for chatbot context"""
        return {
            'latest_date': self.data.index[-1].strftime('%Y-%m-%d'),
            'total_records': len(self.data),
            'features': len(self.data.columns),
            'conversations': len(self.conversation_history)
        }