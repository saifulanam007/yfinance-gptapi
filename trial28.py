from flask import Flask, request, jsonify, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import openai
import json

app = Flask(__name__)

# Your OpenAI API key
openai.api_key = 'sk-proj-CAJi4ugGY48gGc4anYsYT3BlbkFJUcFsI7sznRrYeptkjc63'  # Replace with your actual API key

def get_historical_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cashflow = stock.cashflow
    quarterly_financials = stock.quarterly_financials
    quarterly_balance_sheet = stock.quarterly_balance_sheet
    quarterly_cashflow = stock.quarterly_cashflow
    earnings = stock.earnings_dates
    historical_prices = stock.history(period=period)
    return financials, balance_sheet, cashflow, quarterly_financials, quarterly_balance_sheet, quarterly_cashflow, earnings, historical_prices

def calculate_ratios(financials, balance_sheet, cashflow):
    ratios = {}

    def safe_get(data, key):
        if isinstance(data, pd.Series):
            return pd.to_numeric(data.get(key, np.nan), errors='coerce')
        elif isinstance(data, pd.DataFrame):
            return pd.to_numeric(data.loc[key, data.columns[0]], errors='coerce')
        else:
            return np.nan

    net_income = safe_get(financials, 'Net Income')
    total_revenue = safe_get(financials, 'Total Revenue')
    total_assets = safe_get(balance_sheet, 'Total Assets')
    total_equity = safe_get(balance_sheet, 'Stockholders Equity')
    total_liabilities = safe_get(balance_sheet, 'Total Liabilities Net Minority Interest')
    current_assets = safe_get(balance_sheet, 'Current Assets')
    current_liabilities = safe_get(balance_sheet, 'Current Liabilities')
    operating_cash_flow = safe_get(cashflow, 'Operating Cash Flow')

    if not pd.isna(net_income) and not pd.isna(total_revenue):
        ratios['Profit Margin'] = net_income / total_revenue
    if not pd.isna(net_income) and not pd.isna(total_assets):
        ratios['ROA'] = net_income / total_assets
    if not pd.isna(net_income) and not pd.isna(total_equity):
        ratios['ROE'] = net_income / total_equity
    if not pd.isna(total_liabilities) and not pd.isna(total_equity):
        ratios['Debt to Equity'] = total_liabilities / total_equity
    if not pd.isna(current_assets) and not pd.isna(current_liabilities):
        ratios['Current Ratio'] = current_assets / current_liabilities
    if not pd.isna(operating_cash_flow) and not pd.isna(net_income):
        ratios['Operating Cash Flow to Net Income'] = operating_cash_flow / net_income
    if not pd.isna(total_revenue) and not pd.isna(total_assets):
        ratios['Asset Turnover'] = total_revenue / total_assets

    inventory = safe_get(balance_sheet, 'Inventory')
    if not pd.isna(current_assets) and not pd.isna(inventory) and not pd.isna(current_liabilities):
        ratios['Quick Ratio'] = (current_assets - inventory) / current_liabilities

    gross_profit = safe_get(financials, 'Gross Profit')
    if not pd.isna(gross_profit) and not pd.isna(total_revenue):
        ratios['Gross Margin'] = gross_profit / total_revenue

    return ratios

def calculate_historical_ratios(financials, balance_sheet, cashflow):
    historical_ratios = {}
    for date in financials.columns:
        year_financials = financials[date]
        year_balance_sheet = balance_sheet[date] if date in balance_sheet.columns else pd.Series()
        year_cashflow = cashflow[date] if date in cashflow.columns else pd.Series()
        year_ratios = calculate_ratios(year_financials, year_balance_sheet, year_cashflow)
        if year_ratios:
            historical_ratios[str(date.date())] = year_ratios
    return historical_ratios if historical_ratios else None

def calculate_ratio_trends(historical_ratios):
    trends = {}
    
    if historical_ratios is None or len(historical_ratios) < 2:
        return None
    
    dates = sorted(historical_ratios.keys())
    for ratio in historical_ratios[dates[0]].keys():
        values = [historical_ratios[date].get(ratio, np.nan) for date in dates]
        values = [v for v in values if not pd.isna(v)]
        
        if len(values) < 2:
            continue
        
        x = range(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        trends[ratio] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }
    
    return trends

def analyze_stock(ticker, period="1y"):
    financials, balance_sheet, cashflow, quarterly_financials, quarterly_balance_sheet, quarterly_cashflow, earnings, historical_prices = get_historical_data(ticker, period)
    
    latest_ratios = calculate_ratios(financials, balance_sheet, cashflow)
    historical_ratios = calculate_historical_ratios(financials, balance_sheet, cashflow)
    ratio_trends = calculate_ratio_trends(historical_ratios)
    
    return financials, balance_sheet, cashflow, quarterly_financials, quarterly_balance_sheet, quarterly_cashflow, earnings, historical_prices, latest_ratios, historical_ratios, ratio_trends

def interpret_trends(ratio_trends):
    interpretations = {}
    for ratio, trend in ratio_trends.items():
        if trend['p_value'] < 0.05:
            direction = "increasing" if trend['slope'] > 0 else "decreasing"
            strength = "strong" if trend['r_squared'] > 0.7 else "moderate" if trend['r_squared'] > 0.3 else "weak"
            interpretations[ratio] = f"{ratio} shows a statistically significant {strength} {direction} trend with an R-squared value of {trend['r_squared']:.2f} and a slope of {trend['slope']:.2f}."
        else:
            interpretations[ratio] = f"No significant trend detected for {ratio}."
    return interpretations

def call_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a financial analyst with expertise in stock analysis and forecasting."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content'].strip()

def summarize_dataframe(df, max_rows=5):
    if len(df) > max_rows:
        return df.head(max_rows).to_string() + f"\n... [truncated, showing {max_rows} out of {len(df)} rows]"
    return df.to_string()

def summarize_dict(d, max_items=5):
    if len(d) > max_items:
        return json.dumps(dict(list(d.items())[:max_items]), indent=2) + f"\n... [truncated, showing {max_items} out of {len(d)} items]"
    return json.dumps(d, indent=2)

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stock Analysis API</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
            .container { max-width: 800px; }
            h1 { color: #2c3e50; margin-top: 30px; }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <h1 class="text-center mb-4">Stock Analysis API</h1>
            <form action="/analyze_earnings" method="get" class="mb-4">
                <div class="input-group">
                    <input type="text" name="ticker" class="form-control" placeholder="Enter ticker symbol (e.g., AAPL)" required>
                    <button type="submit" class="btn btn-primary">Analyze</button>
                </div>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/analyze_earnings', methods=['GET'])
def analyze_earnings():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol"}), 400
    
    try:
        financials, balance_sheet, cashflow, quarterly_financials, quarterly_balance_sheet, quarterly_cashflow, earnings, historical_prices, latest_ratios, historical_ratios, ratio_trends = analyze_stock(ticker)

        summary_financials = summarize_dataframe(financials)
        summary_balance_sheet = summarize_dataframe(balance_sheet)
        summary_cashflow = summarize_dataframe(cashflow)
        summary_latest_ratios = summarize_dict(latest_ratios)
        summary_historical_ratios = summarize_dict(historical_ratios)
        summary_ratio_trends = summarize_dict(ratio_trends)
        
        interpretations = interpret_trends(ratio_trends)
        summary_interpretations = summarize_dict(interpretations)

        if not historical_prices.empty:
            price_data = historical_prices['Close']
            price_trend = stats.linregress(range(len(price_data)), price_data)
            price_interpretation = interpret_trends({'Stock Price': {'slope': price_trend.slope, 'r_squared': price_trend.rvalue**2, 'p_value': price_trend.pvalue}})
        else:
            price_interpretation = {"Stock Price": "No historical price data available."}
        
        prompt = f"""
        Given the following financial data and ratios for the company with ticker symbol {ticker}:
        
        Financials:
        {summary_financials}
        
        Balance Sheet:
        {summary_balance_sheet}
        
        Cashflow:
        {summary_cashflow}
        
        Latest Ratios:
        {summary_latest_ratios}
        
        Historical Ratios:
        {summary_historical_ratios}
        
        Ratio Trends:
        {summary_ratio_trends}
        
        Interpretations:
        {summary_interpretations}
        
        Stock Price Trend:
        {json.dumps(price_interpretation, indent=2)}
        
        Please provide a detailed analysis including key points and insights, focusing on numerical data to enhance understanding.
        Please provide a comprehensive analysis including:

        1. Profitability Analysis
        2. Liquidity and Solvency Analysis
        3. Efficiency Analysis
        4. Growth Analysis
        5. Earnings Analysis
        6. Trend Analysis
        7. Financial Health Assessment
        8. Historical Price analysis
        9. Risk Assessment
        10. Future Outlook and Forecasting
        
        Format your response as HTML, with each section as an <h3> heading followed by <p> paragraphs for the analysis points.
        Focus on the most recent data and significant trends. Provide detailed explanations and insights, supporting your analysis with specific data points from the provided financial information. Pay special attention to the trend interpretations and incorporate these insights throughout your analysis.
        """
        
        analysis = call_chatgpt(prompt)

        # Get the first date from historical_ratios to use as a reference for column names
        first_date = list(historical_ratios.keys())[0] if historical_ratios else None
        ratio_names = list(historical_ratios[first_date].keys()) if first_date else []
        
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stock Analysis: {{ ticker }}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
                .container { max-width: 1200px; }
                h1, h2 { color: #2c3e50; margin-top: 30px; }
                .table { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="text-center mb-4">Stock Analysis: {{ ticker }}</h1>
                
                <h2>Financial Analysis</h2>
                <div>{{ analysis | safe }}</div>
                
                <h2>Latest Ratios</h2>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Ratio</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for ratio, value in latest_ratios.items() %}
                        <tr>
                            <td>{{ ratio }}</td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                
                <h2>Historical Ratios</h2>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Date</th>
                            {% for ratio in ratio_names %}
                            <th>{{ ratio }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for date, ratios in historical_ratios.items() %}
                        <tr>
                            <td>{{ date }}</td>
                            {% for ratio in ratio_names %}
                            <td>{{ ratios.get(ratio, '') }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                
                <h2>Ratio Trend Interpretations</h2>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Ratio</th>
                            <th>Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for ratio, interpretation in interpretations.items() %}
                        <tr>
                            <td>{{ ratio }}</td>
                            <td>{{ interpretation }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        ''', ticker=ticker, analysis=analysis, latest_ratios=latest_ratios, historical_ratios=historical_ratios, interpretations=interpretations, ratio_names=ratio_names)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)