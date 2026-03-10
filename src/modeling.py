class TimeSeriesAnalyzer:
    """
    Time Series Forecast Analyzer.
    
    Computes error metrics, generates diagnostic plots, and identifies
    weak prediction periods with heatmaps and detailed breakdowns.
    
    Parameters:
    -----------
    train : pd.DataFrame
        Training data with PeriodIndex or DatetimeIndex
    test : pd.DataFrame
        Test data with PeriodIndex or DatetimeIndex
    predictions : pd.Series
        Forecasted values aligned with test index
    y_col : str
        Target column name
    model_name : str
        Name for display purposes (default: 'SARIMA')
    
    Usage:
    ------
    analyzer = TimeSeriesAnalyzer(df_train, df_test, predictions, y_col='gaspriceinflation')
    analyzer.summary()
    analyzer.plot_forecast()
    analyzer.plot_residual_diagnostics()
    analyzer.plot_error_heatmap()
    analyzer.plot_monthly_accuracy()
    analyzer.plot_cumulative_error()
    analyzer.plot_all()
    """

    # ── Color palette ──────────────────────────────────────────────────
    COLORS = {
        'train': '#2c3e50',
        'test': '#2980b9',
        'pred': '#e74c3c',
        'ci': 'rgba(231,76,60,0.15)',
        'residual_pos': '#e74c3c',
        'residual_neg': '#27ae60',
        'grid': '#ecf0f1',
        'accent': '#f39c12',
        'bg': '#fafafa',
    }

    LAYOUT_DEFAULTS = dict(
        template='plotly_white',
        font=dict(family='Segoe UI, Roboto, sans-serif', size=12, color='#2c3e50'),
        margin=dict(t=80, b=50, l=60, r=30),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

    def __init__(self, train, test, predictions, y_col, model_name='SARIMA'):
        self.train = train.copy()
        self.test = test.copy()
        self.y_col = y_col
        self.model_name = model_name

        # Align predictions with test
        self.actual = test[y_col].astype(float)
        self.predictions = predictions.astype(float)

        # Ensure same index
        common_idx = self.actual.index.intersection(self.predictions.index)
        self.actual = self.actual.loc[common_idx]
        self.predictions = self.predictions.loc[common_idx]

        # Compute core errors
        self.errors = self.actual - self.predictions
        self.abs_errors = np.abs(self.errors)
        self.sq_errors = self.errors ** 2
        self.pct_errors = np.where(
            self.actual != 0,
            np.abs(self.errors / self.actual) * 100,
            np.nan
        )

        # Compute all metrics
        self.metrics = self._compute_metrics()

    # ══════════════════════════════════════════════════════════════════
    #  METRICS
    # ══════════════════════════════════════════════════════════════════

    def _compute_metrics(self):
        n = len(self.actual)
        actual = self.actual.values
        pred = self.predictions.values
        errors = self.errors.values
        abs_errors = self.abs_errors.values
        sq_errors = self.sq_errors.values

        # Naive forecast (random walk)
        naive_errors = np.abs(np.diff(self.train[self.y_col].astype(float).values))
        mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0

        # Seasonal naive forecast
        m = 12  # monthly
        train_vals = self.train[self.y_col].astype(float).values
        if len(train_vals) > m:
            seasonal_naive_errors = np.abs(train_vals[m:] - train_vals[:-m])
            mae_seasonal_naive = np.mean(seasonal_naive_errors) if len(seasonal_naive_errors) > 0 else 1.0
        else:
            mae_seasonal_naive = mae_naive

        mae = np.mean(abs_errors)
        mse = np.mean(sq_errors)
        rmse = np.sqrt(mse)
        max_error = np.max(abs_errors)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Max Error': max_error,
            'N Observations': n,
        }

    def summary(self):
        """Print a formatted summary of all metrics."""
        print(f"\n{'═' * 55}")
        print(f"  📊 Forecast Evaluation: {self.model_name}")
        print(f"{'═' * 55}")
        print(f"  Test Period: {self.actual.index[0]} → {self.actual.index[-1]}")
        print(f"  Observations: {int(self.metrics['N Observations'])}")
        print(f"{'─' * 55}")

        sections = {
            'Scale-Dependent': ['MAE', 'MSE', 'RMSE', 'Max Error']
        }

        for section, keys in sections.items():
            print(f"\n  ▸ {section}")
            for key in keys:
                val = self.metrics[key]
                print(f"    {key:<20s} {val:>10.4f}")

        print(f"\n{'═' * 55}")
        return pd.DataFrame([self.metrics]).T.rename(columns={0: 'Value'})

    # ══════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════

    def _base_layout(self, title, height=500):
        layout = dict(**self.LAYOUT_DEFAULTS)
        layout['title'] = {'text': title, 'x': 0.5, 'y': 0.97, 'xanchor': 'center',
                        'font': {'size': 18, 'color': '#2c3e50'}}
        layout['height'] = height
        return layout

    def _to_timestamp(self, index):
        """Convert PeriodIndex to timestamps for Plotly compatibility."""
        if isinstance(index, pd.PeriodIndex):
            return index.to_timestamp()
        return index

    # ── 1. Forecast vs Actual ──────────────────────────────────────────

    def plot_forecast(self, show_train=True, last_n_train=None):
        """
        Plot actual vs predicted with optional training data.
        
        Parameters:
        -----------
        show_train : bool
            Whether to show training data
        last_n_train : int, optional
            Only show last N training observations (for clarity)
        """
        fig = go.Figure()

        # Training data
        if show_train:
            train_series = self.train[self.y_col].astype(float)
            if last_n_train:
                train_series = train_series.iloc[-last_n_train:]
            fig.add_trace(go.Scatter(
                x=self._to_timestamp(train_series.index),
                y=train_series.values,
                mode='lines', name='Train',
                line=dict(color=self.COLORS['train'], width=1.5),
                hovertemplate='%{x|%Y-%m}<br>Train: %{y:.2f}<extra></extra>'
            ))

        # Actual test
        fig.add_trace(go.Scatter(
            x=self._to_timestamp(self.actual.index),
            y=self.actual.values,
            mode='lines+markers', name='Actual',
            line=dict(color=self.COLORS['test'], width=2),
            marker=dict(size=4),
            hovertemplate='%{x|%Y-%m}<br>Actual: %{y:.2f}<extra></extra>'
        ))

        # Predictions
        fig.add_trace(go.Scatter(
            x=self._to_timestamp(self.predictions.index),
            y=self.predictions.values,
            mode='lines+markers', name='Predicted',
            line=dict(color=self.COLORS['pred'], width=2, dash='dash'),
            marker=dict(size=4),
            hovertemplate='%{x|%Y-%m}<br>Predicted: %{y:.2f}<extra></extra>'
        ))

        # Vertical split line
        # Vertical split line
        split_x = self._to_timestamp(self.actual.index)[0]
        fig.add_shape(
            type='line',
            x0=split_x, x1=split_x,
            y0=0, y1=1, yref='paper',
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5
        )
        fig.add_annotation(
            x=split_x, y=1.05, yref='paper',
            text='Train/Test Split',
            showarrow=False, font=dict(size=10, color='gray')
        )
        fig.update_layout(
            **self._base_layout(f'{self.model_name} — Forecast vs Actual', 450),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
            xaxis_title='Date', yaxis_title=self.y_col,
        )
        fig.show()
        return fig

    # ── 2. Residual Diagnostics ────────────────────────────────────────

    def plot_residual_diagnostics(self):
        """4-panel residual analysis: residuals, distribution, QQ, ACF."""
        
        errors = self.errors.values
        n = len(errors)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residuals Over Time',
                'Residual Distribution',
                'Q-Q Plot',
                'Residual ACF'
            ),
            vertical_spacing=0.14, horizontal_spacing=0.1
        )

        # ── Panel 1: Residuals over time
        colors = [self.COLORS['residual_pos'] if e > 0 else self.COLORS['residual_neg'] for e in errors]
        fig.add_trace(go.Bar(
            x=list(range(len(errors))), y=errors,
            marker_color=colors, name='Residuals', showlegend=False,
            hovertemplate='Step %{x}<br>Error: %{y:.2f}<extra></extra>'
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash='solid', line_color='gray', opacity=0.5, row=1, col=1)

        # ── Panel 2: Distribution
        fig.add_trace(go.Histogram(
            x=errors, nbinsx=20,
            marker_color=self.COLORS['test'], opacity=0.7,
            name='Distribution', showlegend=False,
            hovertemplate='Bin: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ), row=1, col=2)

        # Normal curve overlay
        x_range = np.linspace(errors.min(), errors.max(), 100)
        pdf = stats.norm.pdf(x_range, np.mean(errors), np.std(errors))
        pdf_scaled = pdf * len(errors) * (errors.max() - errors.min()) / 20
        fig.add_trace(go.Scatter(
            x=x_range, y=pdf_scaled,
            mode='lines', line=dict(color=self.COLORS['pred'], width=2),
            name='Normal Fit', showlegend=False
        ), row=1, col=2)

        # ── Panel 3: Q-Q Plot
        sorted_errors = np.sort(errors)
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, n))
        fig.add_trace(go.Scatter(
            x=theoretical_q, y=sorted_errors,
            mode='markers', marker=dict(color=self.COLORS['test'], size=5),
            name='Q-Q', showlegend=False,
            hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
        ), row=2, col=1)
        # Reference line
        min_val = min(theoretical_q.min(), sorted_errors.min())
        max_val = max(theoretical_q.max(), sorted_errors.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color=self.COLORS['pred'], dash='dash'),
            showlegend=False
        ), row=2, col=1)

        # ── Panel 4: Residual ACF
        max_lags = min(20, n // 3)
        if max_lags > 1:
            acf_vals = acf(errors, nlags=max_lags, fft=True)
            ci_bound = 1.96 / np.sqrt(n)
            for lag, val in enumerate(acf_vals):
                color = self.COLORS['pred'] if abs(val) > ci_bound else self.COLORS['test']
                fig.add_trace(go.Scatter(
                    x=[lag, lag], y=[0, val],
                    mode='lines', line=dict(color=color, width=2),
                    showlegend=False, hoverinfo='skip'
                ), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=list(range(max_lags + 1)), y=acf_vals,
                mode='markers', marker=dict(color=self.COLORS['test'], size=5),
                showlegend=False,
                hovertemplate='Lag %{x}<br>ACF: %{y:.4f}<extra></extra>'
            ), row=2, col=2)
            fig.add_hline(y=ci_bound, line_dash='dash', line_color='gray', opacity=0.5, row=2, col=2)
            fig.add_hline(y=-ci_bound, line_dash='dash', line_color='gray', opacity=0.5, row=2, col=2)
            fig.add_hline(y=0, line_dash='solid', line_color='gray', opacity=0.3, row=2, col=2)

        fig.update_layout(
            **self._base_layout(f'{self.model_name} — Residual Diagnostics', 650),
        )
        fig.update_xaxes(title_text='Step', row=1, col=1)
        fig.update_xaxes(title_text='Error', row=1, col=2)
        fig.update_xaxes(title_text='Theoretical Quantile', row=2, col=1)
        fig.update_xaxes(title_text='Lag', row=2, col=2)
        fig.update_yaxes(title_text='Error', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=2)
        fig.update_yaxes(title_text='Sample Quantile', row=2, col=1)
        fig.update_yaxes(title_text='ACF', row=2, col=2)

        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=13, color='#34495e')

        fig.show()
        return fig

    # ── 3. Error Heatmap by Month/Year ─────────────────────────────────

    def plot_error_heatmap(self, metric='absolute_error'):
        """
        Heatmap of errors by month and year.
        
        Parameters:
        -----------
        metric : str
            'absolute_error', 'squared_error', 'percentage_error', 'raw_error'
        """
        idx = self._to_timestamp(self.actual.index)

        metric_map = {
            'absolute_error': ('Absolute Error', self.abs_errors.values),
            'squared_error': ('Squared Error', self.sq_errors.values),
            'percentage_error': ('Percentage Error (%)', self.pct_errors),
            'raw_error': ('Raw Error', self.errors.values),
        }

        metric_name, values = metric_map.get(metric, metric_map['absolute_error'])

        df_err = pd.DataFrame({
            'year': idx.year,
            'month': idx.month,
            'error': values
        })

        pivot = df_err.pivot_table(index='month', columns='year', values='error', aggfunc='mean')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.index = [month_names[m - 1] for m in pivot.index]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index,
            colorscale='RdYlGn_r' if metric != 'raw_error' else 'RdBu_r',
            text=np.round(pivot.values, 2),
            texttemplate='%{text:.2f}',
            textfont=dict(size=11),
            hovertemplate='Year: %{x}<br>Month: %{y}<br>Error: %{z:.3f}<extra></extra>',
            colorbar=dict(title=metric_name)
        ))

        fig.update_layout(
            **self._base_layout(f'{self.model_name} — {metric_name} Heatmap', 450),
            xaxis_title='Year', yaxis_title='Month',
            yaxis=dict(autorange='reversed'),
        )
        fig.show()
        return fig

    # ── 4. Monthly Accuracy Breakdown ──────────────────────────────────

    def plot_monthly_accuracy(self):
        """Bar chart of MAE, RMSE, and sMAPE grouped by calendar month."""
        idx = self._to_timestamp(self.actual.index)
        actual = self.actual.values
        pred = self.predictions.values

        df_err = pd.DataFrame({
            'month': idx.month,
            'abs_error': self.abs_errors.values,
            'sq_error': self.sq_errors.values,
            'actual': actual,
            'pred': pred,
        })

        monthly = df_err.groupby('month').agg(
            MAE=('abs_error', 'mean'),
            RMSE=('sq_error', lambda x: np.sqrt(np.mean(x))),
        ).reset_index()

        # sMAPE per month
        smape_monthly = df_err.groupby('month').apply(
            lambda g: np.mean(2 * g['abs_error'] / (np.abs(g['actual']) + np.abs(g['pred']) + 1e-10)) * 100
        ).reset_index(name='sMAPE')
        monthly = monthly.merge(smape_monthly, on='month')

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly['month_name'] = monthly['month'].apply(lambda m: month_names[m - 1])

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('MAE by Month', 'RMSE by Month', 'sMAPE (%) by Month'),
            horizontal_spacing=0.08
        )

        colors_bar = ['#e74c3c', '#2980b9', '#f39c12']
        for col, (metric, color) in enumerate(zip(['MAE', 'RMSE', 'sMAPE'], colors_bar), 1):
            vals = monthly[metric].values
            # Highlight worst month
            bar_colors = [color] * len(vals)
            worst_idx = np.argmax(vals)
            bar_colors[worst_idx] = '#c0392b'

            fig.add_trace(go.Bar(
                x=monthly['month_name'], y=vals,
                marker_color=bar_colors, name=metric, showlegend=False,
                hovertemplate='%{x}<br>' + metric + ': %{y:.3f}<extra></extra>',
                text=np.round(vals, 2), textposition='outside', textfont=dict(size=9),
            ), row=1, col=col)

        fig.update_layout(
            **self._base_layout(f'{self.model_name} — Monthly Accuracy Breakdown', 420),
        )

        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=13, color='#34495e')

        fig.show()
        return fig

    # ── 5. Cumulative Error Over Time ──────────────────────────────────

    def plot_cumulative_error(self):
        """Track how error accumulates over the forecast horizon."""
        cum_mae = np.cumsum(self.abs_errors.values) / np.arange(1, len(self.abs_errors) + 1)
        cum_rmse = np.sqrt(np.cumsum(self.sq_errors.values) / np.arange(1, len(self.sq_errors) + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self._to_timestamp(self.actual.index), y=cum_mae,
            mode='lines+markers', name='Cumulative MAE',
            line=dict(color=self.COLORS['test'], width=2),
            marker=dict(size=4),
            hovertemplate='%{x|%Y-%m}<br>Cum MAE: %{y:.3f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=self._to_timestamp(self.actual.index), y=cum_rmse,
            mode='lines+markers', name='Cumulative RMSE',
            line=dict(color=self.COLORS['pred'], width=2),
            marker=dict(size=4),
            hovertemplate='%{x|%Y-%m}<br>Cum RMSE: %{y:.3f}<extra></extra>'
        ))

        # Overall MAE reference
        fig.add_hline(y=self.metrics['MAE'], line_dash='dash', line_color='gray',
                      opacity=0.5, annotation_text=f"MAE = {self.metrics['MAE']:.3f}")

        fig.update_layout(
            **self._base_layout(f'{self.model_name} — Cumulative Error', 400),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_title='Date', yaxis_title='Error',
            hovermode='x unified',
        )
        fig.show()
        return fig

    # ── 6. Prediction Interval Coverage ────────────────────────────────

    def plot_error_over_time(self):
        """Plot absolute error at each step with a rolling average."""
        abs_err = self.abs_errors.values
        window = max(3, len(abs_err) // 10)
        rolling_mae = pd.Series(abs_err).rolling(window=window, min_periods=1).mean().values

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=self._to_timestamp(self.actual.index), y=abs_err,
            marker_color=self.COLORS['test'], opacity=0.5,
            name='|Error|',
            hovertemplate='%{x|%Y-%m}<br>|Error|: %{y:.3f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=self._to_timestamp(self.actual.index), y=rolling_mae,
            mode='lines', name=f'Rolling MAE ({window}-step)',
            line=dict(color=self.COLORS['pred'], width=2.5),
            hovertemplate='%{x|%Y-%m}<br>Rolling MAE: %{y:.3f}<extra></extra>'
        ))

        fig.update_layout(
            **self._base_layout(f'{self.model_name} — Error Over Time', 400),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_title='Date', yaxis_title='Absolute Error',
            hovermode='x unified',
        )
        fig.show()
        return fig

    # ── 7. Scatter: Actual vs Predicted ────────────────────────────────

    def plot_actual_vs_predicted(self):
        """Scatter plot of actual vs predicted with perfect prediction line."""
        actual = self.actual.values
        pred = self.predictions.values

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actual, y=pred,
            mode='markers',
            marker=dict(color=self.COLORS['test'], size=8, opacity=0.7,
                        line=dict(width=1, color='white')),
            name='Predictions',
            hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
        ))

        # Perfect prediction line
        min_val = min(actual.min(), pred.min())
        max_val = max(actual.max(), pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Perfect Prediction',
            line=dict(color=self.COLORS['pred'], dash='dash', width=2)
        ))


        fig.update_layout(
            **self._base_layout(f'{self.model_name} — Actual vs Predicted', 500),
            xaxis_title='Actual', yaxis_title='Predicted',
            xaxis=dict(scaleanchor='y', scaleratio=1),
        )
        fig.show()
        return fig

    # ── 8. Worst Predictions Table ─────────────────────────────────────

    def worst_predictions(self, n=10):
        """Return the N worst predictions ranked by absolute error."""
        df = pd.DataFrame({
            'Date': self.actual.index,
            'Actual': self.actual.values,
            'Predicted': self.predictions.values,
            'Error': self.errors.values,
            'Abs_Error': self.abs_errors.values,
            'Pct_Error': self.pct_errors,
        }).sort_values('Abs_Error', ascending=False).head(n).reset_index(drop=True)

        df.index = range(1, len(df) + 1)
        df.index.name = 'Rank'
        return df

    # ── PLOT ALL ───────────────────────────────────────────────────────

    def plot_all(self, show_train=True, last_n_train=60):
        """Generate all diagnostic plots at once."""
        print(f"\n{'█' * 55}")
        print(f"  {self.model_name} — Full Diagnostic Report")
        print(f"{'█' * 55}\n")

        metrics_df = self.summary()
        print()

        self.plot_forecast(show_train=show_train, last_n_train=last_n_train)
        self.plot_actual_vs_predicted()
        self.plot_residual_diagnostics()
        self.plot_error_heatmap(metric='absolute_error')
        self.plot_error_heatmap(metric='raw_error')
        self.plot_monthly_accuracy()
        self.plot_cumulative_error()
        self.plot_error_over_time()

        print(f"\n{'─' * 55}")
        print("  Top 10 Worst Predictions:")
        print(self.worst_predictions(10).to_string())
        print(f"{'─' * 55}\n")

        return metrics_df

    # ══════════════════════════════════════════════════════════════════
    #  COMPARISON
    # ══════════════════════════════════════════════════════════════════
    
    @staticmethod
    def compare(analyzers: list, show_train=True, last_n_train=60):
        """
        Compare multiple TimeSeriesAnalyzer instances side by side.
        
        Parameters:
        -----------
        analyzers : list of TimeSeriesAnalyzer
        show_train : bool
            Show training data in forecast overlay
        last_n_train : int
            Only show last N training observations
        
        Returns:
        --------
        comparison_df : pd.DataFrame
        fig_metrics : plotly.Figure (bar chart comparison)
        fig_forecast : plotly.Figure (overlay forecast)
        """
        

        colors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#1abc9c',
                '#e67e22', '#16a085', '#c0392b', '#2c3e50']

        # ── Metrics comparison table ──────────────────────────────────
        records = []
        for a in analyzers:
            row = {'Model': a.model_name}
            row.update(a.metrics)
            records.append(row)
        comp_df = pd.DataFrame(records).set_index('Model')

        # ── Bar chart comparison ──────────────────────────────────────
        key_metrics = ['MAE', 'RMSE', 'MSE']
        available = [m for m in key_metrics if m in comp_df.columns]

        fig_metrics = make_subplots(
            rows=1, cols=len(available),
            subplot_titles=available,
            horizontal_spacing=0.08
        )

        for col, metric in enumerate(available, 1):
            vals = [a.metrics[metric] for a in analyzers]
            best_idx = np.argmin(vals) if metric != 'R²' else np.argmax(vals)

            for i, a in enumerate(analyzers):
                border = dict(width=3, color='gold') if i == best_idx else dict(width=0)
                fig_metrics.add_trace(go.Bar(
                    x=[a.model_name],
                    y=[a.metrics[metric]],
                    marker_color=colors[i % len(colors)],
                    marker_line=border,
                    name=a.model_name,
                    legendgroup=a.model_name,  # ← Add this
                    showlegend=(col == 1),
                    hovertemplate=f'{a.model_name}<br>{metric}: ' + '%{y:.4f}<extra></extra>',
                    text=[f"{a.metrics[metric]:.3f}"],
                    textposition='outside', textfont=dict(size=10),
                ), row=1, col=col)

        fig_metrics.update_layout(
            title={'text': 'Model Comparison — Key Metrics', 'x': 0.5, 'y': 1,
                'xanchor': 'center', 'font': {'size': 18, 'color': '#2c3e50'}},
            height=500, template='plotly_white',
            margin=dict(t=80, b=50, l=50, r=30),
            legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
            barmode='group',
        )
        for ann in fig_metrics['layout']['annotations']:
            ann['font'] = dict(size=12, color='#34495e')

        fig_metrics.show()

        # ── Overlay forecast plot ─────────────────────────────────────
        ref = analyzers[0]  # Use first analyzer for train/actual data
        fig_forecast = go.Figure()

        # Training data
        if show_train:
            train_series = ref.train[ref.y_col].astype(float)
            if last_n_train:
                train_series = train_series.iloc[-last_n_train:]
            fig_forecast.add_trace(go.Scatter(
                x=ref._to_timestamp(train_series.index),
                y=train_series.values,
                mode='lines', name='Train',
                line=dict(color='#95a5a6', width=1.5),
                hovertemplate='%{x|%Y-%m}<br>Train: %{y:.2f}<extra></extra>'
            ))

        # Actual test values
        fig_forecast.add_trace(go.Scatter(
            x=ref._to_timestamp(ref.actual.index),
            y=ref.actual.values,
            mode='lines+markers', name='Actual',
            line=dict(color='#2c3e50', width=2.5),
            marker=dict(size=5),
            hovertemplate='%{x|%Y-%m}<br>Actual: %{y:.2f}<extra></extra>'
        ))

        # Each model's predictions
        dash_styles = ['dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
        for i, a in enumerate(analyzers):
            fig_forecast.add_trace(go.Scatter(
                x=ref._to_timestamp(a.predictions.index),
                y=a.predictions.values,
                mode='lines+markers', name=f'{a.model_name} (MAE={a.metrics["MAE"]:.2f})',
                line=dict(color=colors[i % len(colors)], width=2, dash=dash_styles[i % len(dash_styles)]),
                marker=dict(size=4),
                hovertemplate='%{x|%Y-%m}<br>' + a.model_name + ': %{y:.2f}<extra></extra>'
            ))

        # Train/test split line
        split_x = ref._to_timestamp(ref.actual.index)[0]
        fig_forecast.add_shape(
            type='line', x0=split_x, x1=split_x,
            y0=0, y1=1, yref='paper',
            line=dict(color='gray', width=1, dash='dot'), opacity=0.5
        )
        fig_forecast.add_annotation(
            x=split_x, y=1.05, yref='paper',
            text='Train/Test Split',
            showarrow=False, font=dict(size=10, color='gray')
        )

        fig_forecast.update_layout(
            title={'text': 'Model Comparison — Forecast Overlay', 'x': 0.5, 'y': 1,
                'xanchor': 'center', 'font': {'size': 18, 'color': '#2c3e50'}},
            height=500, template='plotly_white',
            margin=dict(t=80, b=50, l=60, r=30),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
            xaxis_title='Date', yaxis_title=ref.y_col,
        )

        fig_forecast.show()

        # ── Print summary table ───────────────────────────────────────
        print(f"\n{'═' * 70}")
        print("  Model Comparison Summary")
        print(f"{'═' * 70}")
        display_cols = ['MAE', 'RMSE', 'MSE', 'MASE', 'Seasonal MASE', 'sMAPE (%)']
        available_cols = [c for c in display_cols if c in comp_df.columns]
        print(comp_df[available_cols].round(4).to_string())
        print(f"{'─' * 70}")

        # Best model per metric
        for metric in available_cols:
            best = comp_df[metric].idxmin() if metric != 'R²' else comp_df[metric].idxmax()
            print(f"Best {metric}: {best} ({comp_df.loc[best, metric]:.4f})")
        print(f"{'═' * 70}\n")

        return comp_df, fig_metrics, fig_forecast
