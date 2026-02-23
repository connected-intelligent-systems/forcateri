import plotly.graph_objects as go
from typing import List
import pandas as pd
from forcateri import TimeSeries


def plot_metric(
    metric_name: str, metric_list: List[pd.DataFrame], model_name: str
) -> "go.Figure":
    fig = go.Figure()
    xlabel = "Index"

    for i, df in enumerate(metric_list):
        if not isinstance(df, pd.DataFrame) or len(df) <= 1:
            continue

        # Determine x-axis
        if isinstance(df.index, pd.MultiIndex):
            level = 1 if len(df.index.names) > 1 else 0
            x = df.index.get_level_values(level)
            xlabel = df.index.names[level]
        else:
            x = df.index

        # Add traces
        for col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col],
                    mode="lines",
                    name=f"Test series id: {i} - {col}",
                )
            )

    # Set layout
    fig.update_layout(
        title=f"{metric_name} for {model_name}",
        xaxis_title=xlabel,
        yaxis_title="Metric Value",
        legend_title="Series",
        template="plotly_white",
        autosize=True,
    )
    return fig


def plot_quantile_predictions(
    quantiles: List[float],
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    offset: pd.Timedelta,
    model_name: str,
    test_series_id: int,
):

    lower_q, upper_q = quantiles[0], quantiles[-1]
    median_q = min(quantiles, key=lambda q: abs(q - 0.5))
    fig = go.Figure()

    # Lower quantile
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df[lower_q],
            mode="lines",
            name=f"Lower q={lower_q:.2f}",
            line=dict(dash="dash", color="blue", width=1),
            opacity=0.7,
        )
    )

    # Upper quantile
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df[upper_q],
            mode="lines",
            name=f"Upper q={upper_q:.2f}",
            line=dict(dash="dash", color="blue", width=1),
            opacity=0.7,
        )
    )

    # Median quantile
    if median_q in pred_df.columns:
        fig.add_trace(
            go.Scatter(
                x=pred_df.index,
                y=pred_df[median_q],
                mode="lines",
                name=f"Median q={median_q:.2f}",
                line=dict(color="blue", width=2),
            )
        )

    # Ground truth
    gt_df.columns = ["Ground Truth"]
    fig.add_trace(
        go.Scatter(
            x=gt_df.index,
            y=gt_df["Ground Truth"],
            mode="lines",
            name="Ground Truth",
            line=dict(color="black", dash="dash", width=2),
        )
    )

    # Layout
    fig.update_layout(
        title=f"{model_name} — Test Series {test_series_id} — Offset {offset}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Series",
        template="plotly_white",
    )
    return fig


def plot_determ_predictions(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    offset: pd.Timedelta,
    model_name: str,
    test_series_id: int,
):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pred_df.index,
            y=pred_df.iloc[:, 0],
            mode="lines",
            name="Prediction",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gt_df.index,
            y=gt_df.iloc[:, 0],
            mode="lines",
            name="Ground Truth",
            line=dict(color="black", dash="dash", width=2),
        )
    )

    fig.update_layout(
        title=f"{model_name} — Test Series {test_series_id} — Offset {offset}",
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Series",
        template="plotly_white",
    )
    return fig
