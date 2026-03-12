from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# ── Period aliases ──────────────────────────────────────────────────────────
PERIOD_ALIASES: dict[str, str] = {
    # human-readable keys  →  pandas offset alias
    "hourly":   "h",
    "hour":     "h",
    "h":        "h",
    "daily":    "D",
    "day":      "D",
    "d":        "D",
    "weekly":   "W",
    "week":     "W",
    "w":        "W",
    "monthly":  "ME",   # month-end (pandas ≥ 2.2)
    "month":    "ME",
    "m":        "ME",
    "me":       "ME",
}

# ── Colour palette ──────────────────────────────────────────────────────────
BLUE_FULL  = "#2563EB"   # single-series line
BLUE_TRAIN = "#1D4ED8"   # train line  (darker blue)
BLUE_TEST  = "#60A5FA"   # test line   (lighter blue)
BLUE_FILL  = "rgba(37,99,235,0.08)"

PREDICTOR_COLOURS = [
    "#2563EB", "#0EA5E9", "#6366F1", "#8B5CF6", "#06B6D4",
    "#3B82F6", "#A78BFA", "#38BDF8", "#818CF8", "#7C3AED",
]

# ── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_period(period: str) -> str:
    """Return a pandas-compatible offset alias from a human-readable string."""
    key = period.strip().lower()
    alias = PERIOD_ALIASES.get(key)
    if alias is None:
        raise ValueError(
            f"Unknown period '{period}'. "
            f"Valid options: {sorted(PERIOD_ALIASES.keys())}"
        )
    return alias


def _prepare(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period_alias: str,
    agg: str,
) -> pd.DataFrame:
    """
    Parse dates, set index, resample, and return a clean DataFrame
    with columns [date_col, value_col].
    """
    df = df[[date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Resample
    agg_func = getattr(df[value_col].resample(period_alias), agg)
    resampled = agg_func().dropna().reset_index()
    resampled.columns = [date_col, value_col]
    return resampled


def _base_layout(title: str, period: str, value_col: str) -> dict:
    """Return a shared Plotly layout dictionary."""
    return dict(
        title=dict(
            text=title,
            font=dict(size=20, color="#1E293B"),
            x=0.02,
        ),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True, thickness=0.06),
            type="date",
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            showspikes=True,
            spikecolor="#94A3B8",
            spikethickness=1,
        ),
        yaxis=dict(
            title=value_col,
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            showspikes=True,
            spikecolor="#94A3B8",
            spikethickness=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        hovermode="x unified",
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        font=dict(family="'IBM Plex Sans', Arial, sans-serif", color="#334155"),
        margin=dict(l=60, r=30, t=70, b=80),
        # ── Zoom / pan buttons ──────────────────────────────────────────
        xaxis_rangeselector=dict(
            buttons=[
                dict(count=7,  label="1W",  step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#EFF6FF",
            activecolor="#2563EB",
            font=dict(color="#1E40AF"),
        ),
        dragmode="zoom",
        annotations=[
            dict(
                text=f"Period: <b>{period.upper()}</b>",
                xref="paper", yref="paper",
                x=1, y=1.06,
                showarrow=False,
                font=dict(size=11, color="#64748B"),
            )
        ],
    )


def plot_time_series(
    df: Optional[pd.DataFrame] = None,
    *,
    df_train: Optional[pd.DataFrame] = None,
    df_test:  Optional[pd.DataFrame] = None,
    date_col:  str = "ds",
    value_col: str = "y",
    period: str = "D",
    agg: str = "mean",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Parameters
    ----------
    df : pd.DataFrame, optional
        Full dataset (date + value columns). Mutually exclusive with
        df_train / df_test.
    df_train : pd.DataFrame, optional
        Training portion of the split.
    df_test : pd.DataFrame, optional
        Test portion of the split.
    date_col : str
        Name of the datetime column.  Default ``"ds"``.
    value_col : str
        Name of the target value column.  Default ``"y"``.
    period : str
        Resampling period. Accepts human-readable strings:
        ``'hourly'``, ``'daily'``, ``'weekly'``, ``'monthly'``
        (or short forms ``'h'``, ``'d'``, ``'w'``, ``'m'``).
    agg : str
        Aggregation function applied during resampling.
        Default ``"mean"``; also accepts ``"sum"``, ``"last"``, etc.
    title : str, optional
        Chart title.  Auto-generated when omitted.
    """

    # ── Validate inputs ─────────────────────────────────────────────────────
    has_full  = df is not None
    has_split = (df_train is not None) or (df_test is not None)

    if has_full and has_split:
        raise ValueError(
            "Provide either `df` (full dataset) OR `df_train`/`df_test`, not both."
        )
    if not has_full and not has_split:
        raise ValueError(
            "You must provide either `df` or at least one of `df_train` / `df_test`."
        )

    period_alias = _resolve_period(period)
    fig = go.Figure()

    # ── Case 1: single DataFrame ─────────────────────────────────────────
    if has_full:
        data = _prepare(df, date_col, value_col, period_alias, agg)

        fig.add_trace(
            go.Scatter(
                x=data[date_col],
                y=data[value_col],
                mode="lines",
                name=value_col,
                line=dict(color=BLUE_FULL, width=2),
                fill="tozeroy",
                fillcolor=BLUE_FILL,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.4g}</b><extra></extra>",
            )
        )
        fig.update_xaxes(
            range=[data[date_col].iloc[0], data[date_col].iloc[-1]],
            rangeslider=dict(visible=True, thickness=0.06),
            tick0=data[date_col].iloc[0],
            tickformat="%Y-%m-%d",
        )
        chart_title = title or f"{value_col} — {period.capitalize()} view"

    # ── Case 2: train / test split ───────────────────────────────────────
    else:
        if df_train is not None:
            train_data = _prepare(df_train, date_col, value_col, period_alias, agg)
            fig.add_trace(
                go.Scatter(
                    x=train_data[date_col],
                    y=train_data[value_col],
                    mode="lines",
                    name="Train",
                    line=dict(color=BLUE_TRAIN, width=2),
                    fill="tozeroy",
                    fillcolor=BLUE_FILL,
                    hovertemplate="Train  %{x|%Y-%m-%d}<br><b>%{y:.4g}</b><extra></extra>",
                )
            )

        if df_test is not None:
            test_data = _prepare(df_test, date_col, value_col, period_alias, agg)
            fig.add_trace(
                go.Scatter(
                    x=test_data[date_col],
                    y=test_data[value_col],
                    mode="lines",
                    name="Test",
                    line=dict(color=BLUE_TEST, width=2.5, dash="dot"),
                    hovertemplate="Test  %{x|%Y-%m-%d}<br><b>%{y:.4g}</b><extra></extra>",
                )
            )

            # Vertical dashed separator at the split point
            if df_train is not None:
                split_date = test_data[date_col].iloc[0]
                fig.add_vline(
                    x=split_date,
                    line=dict(color="#94A3B8", dash="dash", width=1.5),
                    annotation_text="split",
                    annotation_position="top right",
                    annotation_font=dict(color="#64748B", size=11),
                )

        chart_title = title or f"{value_col} — Train / Test  ({period.capitalize()})"

    # ── Apply layout ─────────────────────────────────────────────────────
    fig.update_layout(**_base_layout(chart_title, period, value_col))

    return fig


def plot_time_series_predictors(
    df: pd.DataFrame,
    predictors: list[str],
    *,
    date_col: str = "date",
    period: str = "D",
    agg: str = "mean",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot all predictors over time with individual legend toggles.

    Parameters
    ----------
    df          : DataFrame containing the date column + predictor columns.
    predictors  : List of column names to plot.
    date_col    : Name of the datetime column / index.  Default ``"date"``.
    period      : Resampling period – ``'hourly'``, ``'daily'``,
                  ``'weekly'``, ``'monthly'`` (or short forms).
    agg         : Aggregation applied on resample.  Default ``"mean"``.
    title       : Chart title.  Auto-generated when omitted.
    """
    period_alias = _resolve_period(period)

    # ── Ensure date_col is a real column ─────────────────────────────────
    if date_col not in df.columns:
        df = df.reset_index()

    missing = [p for p in predictors if p not in df.columns]
    if missing:
        raise ValueError(f"Predictors not found in DataFrame: {missing}")

    fig = go.Figure()

    for i, predictor in enumerate(predictors):
        data = _prepare(df, date_col, predictor, period_alias, agg)
        colour = PREDICTOR_COLOURS[i % len(PREDICTOR_COLOURS)]

        fig.add_trace(
            go.Scatter(
                x=data[date_col],
                y=data[predictor],
                mode="lines",
                name=predictor,
                line=dict(color=colour, width=1.8),
                visible=True,          # all ON by default
                hovertemplate=(
                    f"<b>{predictor}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "%{y:.4g}<extra></extra>"
                ),
            )
        )

    # ── Range selector buttons ────────────────────────────────────────────
    rangeselector = dict(
        buttons=[
            dict(count=7,  label="1W",  step="day",   stepmode="backward"),
            dict(count=1,  label="1M",  step="month", stepmode="backward"),
            dict(count=3,  label="3M",  step="month", stepmode="backward"),
            dict(count=6,  label="6M",  step="month", stepmode="backward"),
            dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#EFF6FF",
        activecolor="#2563EB",
        font=dict(color="#1E40AF"),
    )

    chart_title = title or f"Predictors over time — {period.capitalize()} view"

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=20, color="#1E293B"), x=0.02),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=rangeselector,
            type="date",
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            showspikes=True,
            spikecolor="#94A3B8",
            spikethickness=1,
        ),
        yaxis=dict(
            title="Value",
            showgrid=True,
            gridcolor="#E2E8F0",
            linecolor="#CBD5E1",
            showspikes=True,
            spikecolor="#94A3B8",
            spikethickness=1,
        ),
        # ── Legend as interactive selector ───────────────────────────────
        legend=dict(
            title=dict(text="Predictors — click to toggle", font=dict(size=11, color="#64748B")),
            orientation="v",
            x=1.01,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=12),
            itemclick="toggle",           # single click → hide/show one
            itemdoubleclick="toggleothers",  # double click → isolate one
        ),
        hovermode="x unified",
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        font=dict(family="'IBM Plex Sans', Arial, sans-serif", color="#334155"),
        margin=dict(l=60, r=160, t=70, b=80),
        dragmode="zoom",
        annotations=[
            dict(
                text=f"Period: <b>{period.upper()}</b>",
                xref="paper", yref="paper",
                x=1, y=1.06,
                showarrow=False,
                font=dict(size=11, color="#64748B"),
            )
        ],
    )

    return fig

def plot_time_series_decomposition(df, y_col, title='Time Series Decomposition', period=12, model='additive'):
    """
    Create an interactive time series decomposition plot with range selectors.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index
    y_col : str or list
        Column name to decompose (if list, uses first element)
    title : str
        Main plot title
    period : int
        Seasonal period (default: 12 for monthly data)
    model : str
        'additive' or 'multiplicative'
    """
    
    
    # Handle if y_col is a list
    if isinstance(y_col, list):
        y_col = y_col[0]
    
    # Create a copy to avoid modifying the original
    df_plot = df.copy()
    
    # Convert index to DatetimeIndex if needed
    if isinstance(df_plot.index, pd.PeriodIndex):
        df_plot.index = df_plot.index.to_timestamp()
    elif not isinstance(df_plot.index, pd.DatetimeIndex):
        df_plot.index = pd.to_datetime(df_plot.index)
    
    # Perform decomposition
    decomposed = seasonal_decompose(
        df_plot[y_col],
        model=model,
        period=period,
        extrapolate_trend='freq'
    )
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            'Original Series',
            'Trend Component',
            'Seasonal Component',
            'Residual Component'
        )
    )
    
    # Color scheme
    colors = {
        'observed': '#1f77b4',
        'trend': '#ff7f0e',
        'seasonal': '#2ca02c',
        'resid': '#d62728'
    }
    
    # Add Observed trace
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=decomposed.observed,
            mode='lines',
            name='Observed',
            line=dict(color=colors['observed'], width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                          '<b>Observed</b>: %{y:.2f}<br>' +
                          '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add Trend trace
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=decomposed.trend,
            mode='lines',
            name='Trend',
            line=dict(color=colors['trend'], width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                          '<b>Trend</b>: %{y:.2f}<br>' +
                          '<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add Seasonal trace
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=decomposed.seasonal,
            mode='lines',
            name='Seasonal',
            line=dict(color=colors['seasonal'], width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                          '<b>Seasonal</b>: %{y:.2f}<br>' +
                          '<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Add Residual trace as SCATTER POINTS
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=decomposed.resid,
            mode='markers',  # Changed to markers only
            name='Residual',
            marker=dict(
                color=colors['resid'],
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color='darkred')
            ),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br>' +
                          '<b>Residual</b>: %{y:.2f}<br>' +
                          '<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Range selector buttons
    range_buttons = [
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=2, label="2y", step="year", stepmode="backward"),
        dict(count=5, label="5y", step="year", stepmode="backward"),
        dict(step="all", label="All")
    ]
    
    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'y': 0.98,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        height=900,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Date",
        row=4, col=1,
        rangeselector=dict(
            buttons=range_buttons,
            bgcolor='#ecf0f1',
            activecolor='#3498db',
            x=0.0,
            y=1.05
        ),
        rangeslider=dict(
            visible=True,
            bgcolor='#ecf0f1',
            thickness=0.05
        ),
        type='date'
    )
    
    # Update y-axes labels
    y_labels = ['Value', 'Trend', 'Seasonal', 'Residual']
    for i, label in enumerate(y_labels, 1):
        fig.update_yaxes(
            title_text=label,
            row=i, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='#ecf0f1'
        )
    
    # Add zero reference line for residuals
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="gray", 
        opacity=0.5,
        row=4, col=1
    )
    
    # Update subplot titles styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color='#34495e')
    
    fig.show()
    
    return fig, decomposed

def plot_acf_pacf(series, lags=40, title='ACF & PACF', confidence=0.05):
    """
    Interactive ACF/PACF plot using Plotly.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    lags : int
        Number of lags to display
    title : str
        Main plot title
    confidence : float
        Significance level (default 0.05 for 95% CI)
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
    sig_df : pd.DataFrame with significant lags
    """

    series = series.astype(float).dropna()
    n = len(series)

    # ACF with Bartlett's CI, PACF with 1/sqrt(n) CI
    acf_vals, acf_ci = acf(series, nlags=lags, fft=True, alpha=confidence)
    pacf_vals, pacf_ci = pacf(series, nlags=lags, method='ywm', alpha=confidence)

    # Extract upper/lower bounds relative to zero
    acf_upper = acf_ci[:, 1] - acf_vals
    acf_lower = acf_ci[:, 0] - acf_vals
    pacf_upper = pacf_ci[:, 1] - pacf_vals
    pacf_lower = pacf_ci[:, 0] - pacf_vals
    
    # Individual ACF p-values via t-test 
    acf_se = np.sqrt((1 + 2 * np.cumsum(acf_vals[1:]**2)) / n)  # Bartlett's SE
    acf_tstat = acf_vals[1:] / acf_se
    acf_pvalues = 2 * (1 - stats.norm.cdf(np.abs(acf_tstat)))

    # PACF p-values via t-test
    pacf_tstat = pacf_vals[1:] * np.sqrt(n)
    pacf_pvalues = 2 * (1 - stats.t.cdf(np.abs(pacf_tstat), df=n - 1))

    # Build significance DataFrame
    records = []
    for lag in range(1, lags + 1):
        is_acf_sig = acf_vals[lag] > acf_upper[lag] or acf_vals[lag] < acf_lower[lag]
        is_pacf_sig = pacf_vals[lag] > pacf_upper[lag] or pacf_vals[lag] < pacf_lower[lag]
        if is_acf_sig or is_pacf_sig:
            records.append({
                'Lag': lag,
                'ACF_Value': round(acf_vals[lag], 4),
                'ACF_p_value': round(acf_pvalues[lag - 1], 6),
                'ACF_Significant': is_acf_sig,
                'PACF_Value': round(pacf_vals[lag], 4),
                'PACF_p_value': round(pacf_pvalues[lag - 1], 6),
                'PACF_Significant': is_pacf_sig
            })
    sig_df = pd.DataFrame(records)

    lag_range = list(range(1, lags + 1))
    acf_vals = acf_vals[1:]
    pacf_vals = pacf_vals[1:]
    acf_upper = acf_upper[1:]
    acf_lower = acf_lower[1:]
    pacf_upper = pacf_upper[1:]
    pacf_lower = pacf_lower[1:]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=('Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)')
    )

    colors = {'acf': '#2980b9', 'pacf': '#e74c3c', 'ci': 'rgba(52,152,219,0.15)'}

    ci_bounds = [
        (acf_upper, acf_lower),
        (pacf_upper, pacf_lower)
    ]

    for row, (vals, color, (upper, lower)) in enumerate(
        zip(
            [acf_vals, pacf_vals],
            [colors['acf'], colors['pacf']],
            ci_bounds
        ), 1
    ):
        # Stem lines
        for lag, val in zip(lag_range, vals):
            fig.add_trace(go.Scatter(
                x=[lag, lag], y=[0, val],
                mode='lines', line=dict(color=color, width=1.5),
                showlegend=False, hoverinfo='skip'
            ), row=row, col=1)

        # Markers
        fig.add_trace(go.Scatter(
            x=lag_range, y=vals,
            mode='markers',
            marker=dict(color=color, size=6),
            name='ACF' if row == 1 else 'PACF',
            hovertemplate='Lag %{x}<br>Value: %{y:.4f}<extra></extra>'
        ), row=row, col=1)

        # Confidence bands (correct CI from statsmodels)
        fig.add_trace(go.Scatter(
            x=lag_range + lag_range[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself', fillcolor=colors['ci'],
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=(row == 1), name=f'{int((1 - confidence) * 100)}% CI',
            hoverinfo='skip'
        ), row=row, col=1)

        # Zero line
        fig.add_hline(y=0, line_dash='solid', line_color='gray', opacity=0.4, row=row, col=1)

    fig.update_layout(
        title={'text': title, 'x': 0.5, 'y': 0.97, 'xanchor': 'center',
               'font': {'size': 18, 'color': '#2c3e50'}},
        height=600, template='plotly_white',
        margin=dict(t=80, b=40, l=60, r=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    fig.update_xaxes(title_text='Lag', row=2, col=1, dtick=5)
    fig.update_yaxes(title_text='ACF', row=1, col=1)
    fig.update_yaxes(title_text='PACF', row=2, col=1)

    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=13, color='#34495e')

    fig.show()
    return fig, sig_df
