import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from django.conf import settings
from django.shortcuts import render, redirect
from neuralprophet import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATABASE = "/content/ecommerce_data.sqlite3"
BACKTEST_MODEL = "/content/sales_forecast_model_backtest.np"
PRODUCTION_MODEL = "/content/sales_forecast_model_production.np"

# Global configuration for simulated current date and data ranges
CURRENT_DATE = datetime(2018, 6, 30)  # Simulated "today"
FUTURE_START_DATE = CURRENT_DATE + timedelta(days=1)  # Predictions start the next day
BACKTEST_DAYS = 30  # Number of days held out for backtesting
HISTORICAL_START_DATE = datetime(2017, 1, 1)  # Beginning of available historical data
MAX_FUTURE_DAYS = 60  # Maximum days allowed for future predictions
FUTURE_WARNING_THRESHOLD = 30  # Warn if predicting beyond this many days


def generate_historical_sales_chart(start_date, end_date):
    """Generate historical daily sales chart"""
    con = sqlite3.connect(DATABASE)
    df = pd.read_sql_query(f"""
        SELECT
            DATE(order_purchase_timestamp) AS ds, 
            SUM(payment_value) AS y
        FROM orders
        JOIN order_payments
            ON orders.order_id = order_payments.order_id
        WHERE DATE(order_purchase_timestamp) BETWEEN '{start_date}' AND '{end_date}'
          AND order_status != 'cancelled'
        GROUP BY DATE(order_purchase_timestamp)
        ORDER BY ds
    """, con, parse_dates=['ds'])
    con.close()

    if df.empty:
        return None

    df = df[df["y"] <= 100000]

    # Create Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Sales', line=dict(color='#007bff'),
        hovertemplate='$%{y:,.2f}<br>Date: %{x}<extra></extra>'))

    fig.update_layout(title='Daily Sales Totals', xaxis_title='Date', yaxis_title='Sales ($)', xaxis=dict(tickangle=45),
        template='plotly_white', showlegend=False)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)')

    return pio.to_html(fig, full_html=False)


def generate_backtest_chart(start_date, end_date):
    """Generate backtest chart and metrics"""
    con = sqlite3.connect(DATABASE)
    df = pd.read_sql_query(f"""
        SELECT
            DATE(order_purchase_timestamp) AS ds, 
            SUM(payment_value) AS y
        FROM orders
        JOIN order_payments
            ON orders.order_id = order_payments.order_id
        WHERE DATE(order_purchase_timestamp) BETWEEN '{start_date}' AND '{end_date}'
          AND order_status != 'cancelled'
        GROUP BY DATE(order_purchase_timestamp)
        ORDER BY ds
    """, con, parse_dates=['ds'])
    con.close()

    if df.empty:
        return None, None, None

    df = df[df["y"] <= 100000]
    model = load(BACKTEST_MODEL)
    forecast = model.predict(df)

    y_true = df['y'].values
    y_pred = forecast['yhat1'].values

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mpe = np.mean((y_true - y_pred) / y_true) * 100
    metrics = {"RMSE": f"${rmse:,.2f}", "MAE": f"${mae:,.2f}", "R2": f"{r2:.4f}", "MAPE": f"{mape:.2f}%",
               "MPE": f"{mpe:.2f}%"}

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=y_true, mode='lines+markers', name='Actual', line=dict(color='blue'),
                             hovertemplate='$%{y:,.2f}<br>Date: %{x}<extra></extra>'))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=y_pred, mode='lines+markers', name='Predicted', line=dict(color='red'),
                             hovertemplate='$%{y:,.2f}<br>Date: %{x}<extra></extra>'))

    fig.update_layout(title='Actual vs Predicted Sales', xaxis_title='Date', yaxis_title='Sales ($)')
    chart_div = pio.to_html(fig, full_html=False)

    # Generate residuals histogram
    residuals = y_true - y_pred
    residuals_chart = generate_residuals_histogram(residuals)

    return chart_div, metrics, residuals_chart


def generate_residuals_histogram(residuals):
    fig = go.Figure()

    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals',
        marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
        hovertemplate='Error: $%{x:,.2f}<br>Count: %{y}<extra></extra>'))

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error", annotation_position="top")

    mean_residual = np.mean(residuals)
    fig.add_vline(x=mean_residual, line_dash="dot", line_color="green")

    fig.add_annotation(x=mean_residual, y=1.05, xref="x", yref="paper", text=f"Mean: ${mean_residual:,.2f}",
        showarrow=False, font=dict(size=12, color="green"))

    fig.update_layout(title='Residuals Distribution (Actual - Predicted)', xaxis_title='Residual ($)',
        yaxis_title='Frequency', showlegend=False)

    return pio.to_html(fig, full_html=False)


def generate_future_chart(num_days):
    """Generate future prediction chart starting from the day after CURRENT_DATE"""
    start_date = FUTURE_START_DATE.strftime("%Y-%m-%d")
    end_date = (FUTURE_START_DATE + timedelta(days=num_days - 1)).strftime("%Y-%m-%d")

    future_dates = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date, freq='D'), 'y': np.nan})

    model = load(PRODUCTION_MODEL)
    future_forecast = model.predict(future_dates)

    fig_future = go.Figure()
    fig_future.add_trace(
        go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat1'], mode='lines+markers', name='Predicted',
                   line=dict(color='green'), hovertemplate='$%{y:,.2f}<br>Date: %{x}<extra></extra>'))

    fig_future.update_layout(title='Future Sales Predictions', xaxis_title='Date', yaxis_title='Predicted Sales ($)')
    return pio.to_html(fig_future, full_html=False)


def dashboard(request):
    # Calculate dates based on global configuration
    historical_min_date = HISTORICAL_START_DATE.strftime("%Y-%m-%d")
    historical_max_date = CURRENT_DATE.strftime("%Y-%m-%d")

    # Default: show last 90 days of historical data
    default_historical_start = HISTORICAL_START_DATE.strftime("%Y-%m-%d")
    default_historical_end = CURRENT_DATE.strftime("%Y-%m-%d")

    # Backtest constraints: last BACKTEST_DAYS before CURRENT_DATE
    backtest_min_date = (CURRENT_DATE - timedelta(days=BACKTEST_DAYS - 1)).strftime("%Y-%m-%d")
    backtest_max_date = CURRENT_DATE.strftime("%Y-%m-%d")

    # Default: show full backtest period
    default_backtest_start = backtest_min_date
    default_backtest_end = backtest_max_date

    # Future prediction configuration
    future_start_date = FUTURE_START_DATE.strftime("%Y-%m-%d")
    max_future_days = MAX_FUTURE_DAYS
    future_warning_threshold = FUTURE_WARNING_THRESHOLD

    # Default: 30 days of predictions
    default_future_days = 30

    # Initialize from session or defaults
    current_historical_start = request.session.get('historical_start', default_historical_start)
    current_historical_end = request.session.get('historical_end', default_historical_end)
    current_backtest_start = request.session.get('backtest_start', default_backtest_start)
    current_backtest_end = request.session.get('backtest_end', default_backtest_end)
    current_future_days = request.session.get('future_days', default_future_days)

    # Get cached charts from session
    historical_chart_div = request.session.get('historical_chart_div')
    chart_div = request.session.get('chart_div')
    metrics = request.session.get('metrics')
    residuals_chart = request.session.get('residuals_chart')
    future_chart_div = request.session.get('future_chart_div')
    future_warning = request.session.get('future_warning')
    historical_error = None
    backtest_error = None

    if request.method == "POST":
        # Handle historical sales form
        if "historical_submit" in request.POST:
            start_date = request.POST.get("historical_start_date")
            end_date = request.POST.get("historical_end_date")

            # Validate date range is within allowed historical period
            if start_date and end_date:
                if start_date < historical_min_date:
                    historical_error = f"Start date cannot be before {historical_min_date}."
                elif end_date > historical_max_date:
                    historical_error = f"End date cannot be after {historical_max_date}."
                elif start_date > end_date:
                    historical_error = "Start date must be before or equal to end date."
                else:
                    # Update current values and session
                    current_historical_start = start_date
                    current_historical_end = end_date
                    request.session['historical_start'] = start_date
                    request.session['historical_end'] = end_date

                    # Generate new historical chart
                    historical_chart_div = generate_historical_sales_chart(start_date, end_date)

                    # Store in session
                    request.session['historical_chart_div'] = historical_chart_div

        # Handle backtesting form
        elif "backtest_submit" in request.POST:
            start_date = request.POST.get("start_date")
            end_date = request.POST.get("end_date")

            # Validate date range is within allowed backtest period
            if start_date and end_date:
                if start_date < backtest_min_date or start_date > backtest_max_date:
                    backtest_error = f"Start date must be between {backtest_min_date} and {backtest_max_date}."
                elif end_date < backtest_min_date or end_date > backtest_max_date:
                    backtest_error = f"End date must be between {backtest_min_date} and {backtest_max_date}."
                elif start_date > end_date:
                    backtest_error = "Start date must be before or equal to end date."
                else:
                    # Update current values and session
                    current_backtest_start = start_date
                    current_backtest_end = end_date
                    request.session['backtest_start'] = start_date
                    request.session['backtest_end'] = end_date

                    # Generate new backtest chart and residuals
                    chart_div, metrics, residuals_chart = generate_backtest_chart(start_date, end_date)

                    # Store in session
                    request.session['chart_div'] = chart_div
                    request.session['metrics'] = metrics
                    request.session['residuals_chart'] = residuals_chart

        # Handle future prediction form
        elif "future_submit" in request.POST:
            future_days_str = request.POST.get("future_days")

            if future_days_str:
                try:
                    future_days = int(future_days_str)

                    # Validate number of days
                    if future_days < 1:
                        future_warning = "Please select at least 1 day for predictions."
                    elif future_days > max_future_days:
                        future_warning = f"Maximum {max_future_days} days allowed for predictions."
                    else:
                        # Update current values and session
                        current_future_days = future_days
                        request.session['future_days'] = future_days

                        # Check if more than threshold days (show warning)
                        if future_days > future_warning_threshold:
                            future_warning = f"Warning: Predicting more than {future_warning_threshold} days into the future can significantly decrease model accuracy."
                        else:
                            future_warning = None

                        # Store warning in session
                        request.session['future_warning'] = future_warning

                        # Generate new future chart
                        future_chart_div = generate_future_chart(future_days)

                        # Store in session
                        request.session['future_chart_div'] = future_chart_div

                except ValueError:
                    future_warning = "Please enter a valid number of days."

    # Generate default charts on first load (no session data)
    else:
        if not historical_chart_div:
            historical_chart_div = generate_historical_sales_chart(default_historical_start, default_historical_end)
            request.session['historical_chart_div'] = historical_chart_div
            request.session['historical_start'] = default_historical_start
            request.session['historical_end'] = default_historical_end

        if not chart_div:
            chart_div, metrics, residuals_chart = generate_backtest_chart(default_backtest_start, default_backtest_end)
            request.session['chart_div'] = chart_div
            request.session['metrics'] = metrics
            request.session['residuals_chart'] = residuals_chart
            request.session['backtest_start'] = default_backtest_start
            request.session['backtest_end'] = default_backtest_end

        if not future_chart_div:
            future_chart_div = generate_future_chart(default_future_days)
            request.session['future_chart_div'] = future_chart_div
            request.session['future_days'] = default_future_days

    return render(request, "app/dashboard.html",
                  {"historical_chart_div": historical_chart_div, "historical_min_date": historical_min_date,
                   "historical_max_date": historical_max_date, "current_historical_start": current_historical_start,
                   "current_historical_end": current_historical_end, "historical_error": historical_error,
                   "chart_div": chart_div, "metrics": metrics, "residuals_chart": residuals_chart,
                   "backtest_min_date": backtest_min_date, "backtest_max_date": backtest_max_date,
                   "backtest_error": backtest_error, "future_chart_div": future_chart_div,
                   "future_start_date": future_start_date, "max_future_days": max_future_days,
                   "future_warning": future_warning, "current_backtest_start": current_backtest_start,
                   "current_backtest_end": current_backtest_end, "current_future_days": current_future_days,
                   "simulated_current_date": CURRENT_DATE.strftime("%B %d, %Y")})


def root_redirect(request):
    return redirect('/home')


def home(request):
    return render(request, 'app/home.html')
