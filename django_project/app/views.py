import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
from django.conf import settings
from django.shortcuts import render, redirect
from neuralprophet import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATABASE = "/content/ecommerce_data.db"  # or settings.BASE_DIR / "ecommerce_data.db"
MODEL = "/content/sales_forecast_model.np"  # put your model in project root


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
        return None, None

    df = df[df["y"] <= 100000]
    model = load(MODEL)
    forecast = model.predict(df)

    y_true = df['y'].values
    y_pred = forecast['yhat1'].values

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mpe = np.mean((y_true - y_pred) / y_true) * 100
    metrics = {"RMSE": f"${rmse:,.2f}", "MAE": f"${mae:,.2f}", "R2": f"{r2:.4f}", 
              "MAPE": f"{mape:.2f}%", "MPE": f"{mpe:.2f}%"}

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df['ds'],
            y=y_true,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue'),
            hovertemplate='$%{y:,.2f}<br>Date: %{x}<extra></extra>'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=y_pred,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red'),
            hovertemplate='$%{y:,.2f}<br>Date: %{x}<extra></extra>'
        )
    )
    
    fig.update_layout(title='Actual vs Predicted Sales', xaxis_title='Date', yaxis_title='Sales ($)')
    chart_div = pio.to_html(fig, full_html=False)
    
    return chart_div, metrics


def generate_future_chart(start_date, end_date):
    """Generate future prediction chart"""
    future_dates = pd.DataFrame({
        'ds': pd.date_range(start=start_date, end=end_date, freq='D'),
        'y': np.nan
    })
    
    model = load(MODEL)
    future_forecast = model.predict(future_dates)
    
    fig_future = go.Figure()
    fig_future.add_trace(
        go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat1'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='green'),
            hovertemplate='$%{y:,.2f}<br>Date: %{x}<extra></extra>'
        )
    )
    
    fig_future.update_layout(
        title='Future Sales Predictions',
        xaxis_title='Date',
        yaxis_title='Predicted Sales ($)'
    )
    return pio.to_html(fig_future, full_html=False)


def dashboard(request):
    min_date = "2017-01-01"
    max_date = "2018-06-30"
    
    # Default backtest dates
    default_backtest_start = "2018-06-01"
    default_backtest_end = "2018-06-30"
    
    # Future prediction variables
    future_min_date = "2018-07-01"
    
    # Default future prediction dates
    default_future_start = "2018-07-01"
    default_future_end = "2018-07-31"
    
    # Initialize from session or defaults
    current_backtest_start = request.session.get('backtest_start', default_backtest_start)
    current_backtest_end = request.session.get('backtest_end', default_backtest_end)
    current_future_start = request.session.get('future_start', default_future_start)
    current_future_end = request.session.get('future_end', default_future_end)
    
    # Get cached charts from session
    chart_div = request.session.get('chart_div')
    metrics = request.session.get('metrics')
    future_chart_div = request.session.get('future_chart_div')
    future_warning = request.session.get('future_warning')
    date_error = None

    if request.method == "POST":
        # Handle backtesting form
        if "backtest_submit" in request.POST:
            start_date = request.POST.get("start_date")
            end_date = request.POST.get("end_date")
            
            # Validate date range
            if start_date and end_date:
                if start_date < min_date:
                    start_date = min_date
                if end_date > max_date:
                    end_date = max_date

                # Update current values and session
                current_backtest_start = start_date
                current_backtest_end = end_date
                request.session['backtest_start'] = start_date
                request.session['backtest_end'] = end_date
                
                # Generate new backtest chart
                chart_div, metrics = generate_backtest_chart(start_date, end_date)
                
                # Store in session
                request.session['chart_div'] = chart_div
                request.session['metrics'] = metrics

        # Handle future prediction form
        elif "future_submit" in request.POST:
            future_start = request.POST.get("future_start_date")
            future_end = request.POST.get("future_end_date")

            if future_start and future_end:
                # Validate dates
                start_dt = datetime.strptime(future_start, "%Y-%m-%d")
                end_dt = datetime.strptime(future_end, "%Y-%m-%d")
                cutoff_dt = datetime.strptime("2018-06-30", "%Y-%m-%d")
                
                # Check if dates are after cutoff
                if start_dt <= cutoff_dt or end_dt <= cutoff_dt:
                    date_error = "Future predictions must be for dates after June 30, 2018."
                else:
                    # Check date range (max 31 days)
                    date_diff = (end_dt - start_dt).days + 1
                    if date_diff > 31:
                        date_error = "Date range cannot exceed 31 days."
                    else:
                        # Update current values and session
                        current_future_start = future_start
                        current_future_end = future_end
                        request.session['future_start'] = future_start
                        request.session['future_end'] = future_end
                        
                        # Check if range is more than 30 days after cutoff
                        days_from_cutoff = (start_dt - cutoff_dt).days
                        if days_from_cutoff > 30:
                            future_warning = "Warning: The model is intended to be retrained regularly. Prediction error can increase significantly far into the future."
                        else:
                            future_warning = None
                        
                        # Store warning in session
                        request.session['future_warning'] = future_warning
                        
                        # Generate new future chart
                        future_chart_div = generate_future_chart(future_start, future_end)
                        
                        # Store in session
                        request.session['future_chart_div'] = future_chart_div
    
    # Generate default charts on first load (no session data)
    else:
        if not chart_div:
            chart_div, metrics = generate_backtest_chart(default_backtest_start, default_backtest_end)
            request.session['chart_div'] = chart_div
            request.session['metrics'] = metrics
            request.session['backtest_start'] = default_backtest_start
            request.session['backtest_end'] = default_backtest_end
        
        if not future_chart_div:
            future_chart_div = generate_future_chart(default_future_start, default_future_end)
            request.session['future_chart_div'] = future_chart_div
            request.session['future_start'] = default_future_start
            request.session['future_end'] = default_future_end

    return render(request, "app/dashboard.html", {
        "chart_div": chart_div,
        "metrics": metrics,
        "min_date": min_date,
        "max_date": max_date,
        "future_chart_div": future_chart_div,
        "future_min_date": future_min_date,
        "future_warning": future_warning,
        "date_error": date_error,
        "current_backtest_start": current_backtest_start,
        "current_backtest_end": current_backtest_end,
        "current_future_start": current_future_start,
        "current_future_end": current_future_end
    })


def root_redirect(request):
    return redirect('/home')


def home(request):
    return render(request, 'app/home.html')