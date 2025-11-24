import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from django.shortcuts import render, redirect
from neuralprophet import load
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Relative paths to data and models
CSV_FILE = "/content/train.csv"
BACKTEST_MODEL = "/content/model_train.np"
PRODUCTION_MODEL = "/content/model_production.np"

# Cache for loaded data
_cached_data = None
_cached_current_date = None


def load_sales_data():
    """
    Loads and caches sales data from CSV.

    Arguments:
        None

    Returns:
        daily_sales (pd.DataFrame): Processed daily sales data.
        actual_last_date (pd.Timestamp): The last date in the dataset.
    """
    global _cached_data, _cached_current_date

    if _cached_data is not None:
        return _cached_data, _cached_current_date

    df = pd.read_csv(CSV_FILE)
    df['date'] = pd.to_datetime(df['date'])
    actual_last_date = df['date'].max()

    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    daily_sales['y'] = daily_sales['y'].round(2)
    daily_sales = daily_sales.sort_values('ds').reset_index(drop=True)

    _cached_data = daily_sales
    _cached_current_date = actual_last_date

    return daily_sales, actual_last_date


def generate_backtest_chart():
    """
    Generates backtest chart and metrics.

    Arguments:
        None

    Returns:
        chart_div_train (str): HTML div for training chart.
        chart_div_test (str): HTML div for testing chart.
        metrics_dict (dict): Dictionary of performance metrics.
        residuals_chart (str): HTML div for residuals histogram.
    """
    daily_sales, _ = load_sales_data()
    model = load(BACKTEST_MODEL, map_location='cpu')

    # Split data (1.8% for test = ~30 days)
    df_train, df_test = model.split_df(daily_sales, freq="D", valid_p=0.018)

    # Generate predictions
    forecast_train = model.predict(df_train)
    forecast_test = model.predict(df_test)

    # Calculate metrics
    train_merged = forecast_train[['ds', 'yhat1']].merge(df_train[['ds', 'y']], on='ds').dropna()
    test_merged = forecast_test[['ds', 'yhat1']].merge(df_test[['ds', 'y']], on='ds').dropna()

    y_true_train, y_pred_train = train_merged['y'].values, train_merged['yhat1'].values
    y_true_test, y_pred_test = test_merged['y'].values, test_merged['yhat1'].values

    metrics_dict = {'R2_train': f"{r2_score(y_true_train, y_pred_train):.4f}",
                    'R2_test': f"{r2_score(y_true_test, y_pred_test):.4f}",
                    'RMSE_train': f"${np.sqrt(mean_squared_error(y_true_train, y_pred_train)):,.2f}",
                    'RMSE_test': f"${np.sqrt(mean_squared_error(y_true_test, y_pred_test)):,.2f}",
                    'MAPE_train': f"{mean_absolute_percentage_error(y_true_train, y_pred_train) * 100:.2f}",
                    'MAPE_test': f"{mean_absolute_percentage_error(y_true_test, y_pred_test) * 100:.2f}",
                    'MPE_train': f"{np.mean((y_true_train - y_pred_train) / y_true_train) * 100:.2f}",
                    'MPE_test': f"{np.mean((y_true_test - y_pred_test) / y_true_test) * 100:.2f}", }

    # Generate charts
    model.set_plotting_backend("plotly")
    chart_train = model.plot(forecast_train)
    chart_train.update_layout(autosize=True, width=None, height=500  # or whatever height you want
    )

    chart_test = model.plot(forecast_test[-30:])
    chart_test.update_layout(autosize=True, width=None, height=500  # or whatever height you want
    )

    chart_div_train = pio.to_html(chart_train, full_html=False)
    chart_div_test = pio.to_html(chart_test, full_html=False)
    residuals_chart = generate_residuals_histogram(y_true_test - y_pred_test)

    return chart_div_train, chart_div_test, metrics_dict, residuals_chart


def generate_residuals_histogram(residuals):
    """
    Generates residuals histogram.

    Arguments:
       None

    Returns:
        chart_div (str): HTML div for residuals histogram.
    """
    mean_residual = np.mean(residuals)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals',
                               marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
                               hovertemplate='Error: $%{x:,.2f}<br>Count: %{y}<extra></extra>'))

    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error", annotation_position="top")
    fig.add_vline(x=mean_residual, line_dash="dot", line_color="green")
    fig.add_annotation(x=mean_residual, y=1.05, xref="x", yref="paper", text=f"Mean: ${mean_residual:,.2f}",
                       showarrow=False, font=dict(size=12, color="green"))

    fig.update_layout(title='Residuals Distribution (Actual - Predicted)', xaxis_title='Residual ($)',
                      yaxis_title='Frequency', showlegend=False, template='plotly_white')

    chart_div = pio.to_html(fig, full_html=False)
    return chart_div


def generate_prediction_chart(target_days=30):
    """
    Generates recursive forecast using production model.

    Arguments:
        target_days (int): Number of days to forecast.

    Returns:
        chart_div (str): HTML div for prediction chart.
    """
    daily_sales, _ = load_sales_data()
    model = load(PRODUCTION_MODEL, map_location='cpu')

    n_forecasts = model.n_forecasts
    n_iterations = (target_days + n_forecasts - 1) // n_forecasts

    current_df = daily_sales.copy()
    all_predictions = []

    # Recursive forecasting loop
    for iteration in range(n_iterations):
        future_df = model.make_future_dataframe(df=current_df, periods=n_forecasts,
                                                n_historic_predictions=len(current_df))
        forecast = model.predict(future_df)

        # Extract future predictions and combine yhat columns to fill gaps
        future_forecast = forecast[forecast['ds'] > current_df['ds'].max()].copy()
        if len(future_forecast) == 0:
            break

        yhat_cols = sorted([col for col in future_forecast.columns if col.startswith('yhat') and '%' not in col])
        future_forecast['yhat'] = future_forecast[yhat_cols].bfill(axis=1).iloc[:, 0]

        all_predictions.append(future_forecast[['ds', 'yhat']])

        # Append predictions as "actuals" for next iteration
        current_df = pd.concat([current_df, future_forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})],
                               ignore_index=True)

        # Keep only recent history
        if len(current_df) > model.n_lags + 100:
            current_df = current_df.iloc[-(model.n_lags + 100):].reset_index(drop=True)

    if not all_predictions:
        return "<p>Unable to generate predictions</p>"

    # Combine and limit to target days
    combined_forecast = pd.concat(all_predictions, ignore_index=True).sort_values('ds').reset_index(drop=True).head(
        target_days)

    # Create chart
    fig = go.Figure()

    # Historical data (strictly before forecast starts)
    historical = daily_sales[daily_sales['ds'] < combined_forecast['ds'].min()].tail(60)
    fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'], mode='lines+markers', name='Historical Sales',
                             line=dict(color='#0072B2', width=2), marker=dict(size=4),
                             hovertemplate='Date: %{x|%Y-%m-%d}<br>Sales: $%{y:,.2f}<extra></extra>'))

    # Predictions
    fig.add_trace(
        go.Scatter(x=combined_forecast['ds'], y=combined_forecast['yhat'], mode='lines+markers', name='Predicted Sales',
                   line=dict(color='#D55E00', width=2, dash='dash'), marker=dict(size=6, symbol='diamond'),
                   hovertemplate='Date: %{x|%Y-%m-%d}<br>Forecast: $%{y:,.2f}<extra></extra>'))

    # Forecast start line
    forecast_start = daily_sales['ds'].max()
    fig.add_shape(type="line", x0=forecast_start, x1=forecast_start, y0=0, y1=1, yref="paper",
                  line=dict(color="gray", width=2, dash="dot"))
    fig.add_annotation(x=forecast_start, y=1.05, yref="paper", text="Forecast Start", showarrow=False,
                       font=dict(size=12, color="gray"))

    fig.update_layout(title=f'{target_days}-Day Sales Forecast ({len(combined_forecast)} days predicted)',
                      xaxis_title='Date', yaxis_title='Sales ($)', template='plotly_white', hovermode='closest',
                      height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    chart_div = pio.to_html(fig, full_html=False)
    return chart_div, combined_forecast


def dashboard(request):
    """
    Renders the dashboard with backtest and prediction charts.

    Arguments:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: Rendered dashboard page.
    """
    daily_sales, current_date = load_sales_data()

    # Handle forecast days input from form
    if request.method == 'POST':
        try:
            forecast_days = int(request.POST.get('forecast_days', 30))
            forecast_days = max(7, min(30, forecast_days))  # Clamp between 7 and 30
        except (ValueError, TypeError):
            forecast_days = 30

        # Regenerate prediction chart with new days
        chart_div_prediction, forecast_table = generate_prediction_chart(forecast_days)
        request.session['chart_div_prediction'] = chart_div_prediction
        fc = forecast_table.copy()
        fc['ds'] = fc['ds'].astype(str)
        request.session['forecast_table'] = fc.to_dict('records')
        request.session['forecast_days'] = forecast_days
    else:
        forecast_days = request.session.get('forecast_days', 30)

    # Retrieve from session or generate
    if not (chart_div_train := request.session.get('chart_div_train')):
        chart_div_train, chart_div_test, metrics, residuals_chart = generate_backtest_chart()
        request.session.update(
            {'chart_div_train': chart_div_train, 'chart_div_test': chart_div_test, 'metrics': metrics,
             'residuals_chart': residuals_chart})
    else:
        chart_div_test = request.session.get('chart_div_test')
        metrics = request.session.get('metrics')
        residuals_chart = request.session.get('residuals_chart')

    if not (chart_div_prediction := request.session.get('chart_div_prediction')):
        chart_div_prediction, forecast_table = generate_prediction_chart(forecast_days)
        request.session['chart_div_prediction'] = chart_div_prediction
        fc = forecast_table.copy()
        fc['ds'] = fc['ds'].astype(str)
        request.session['forecast_table'] = fc.to_dict('records')

    return render(request, "app/dashboard.html",
                  {"chart_div_train": chart_div_train, "chart_div_test": chart_div_test, "metrics": metrics,
                   "residuals_chart": residuals_chart, "chart_div_prediction": chart_div_prediction,
                   "simulated_current_date": current_date.strftime("%B %d, %Y"), "forecast_days": forecast_days,
                   "forecast_table": request.session.get('forecast_table')})


def root_redirect(request):
    """
    Redirects root URL to home page.

    Arguments:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: Redirect response to home page.
    """
    return redirect('/home')


def home(request):
    """
    Renders the home page.

    Arguments:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: Rendered home page.
    """
    return render(request, 'app/home.html')
