import plotly.express as px
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import plotly.graph_objects as go


#%% create correlation
# corr = X.corr()
# corr


def plot_test_data_roc_curve(false_positive_rate: np.ndarray, 
                             true_positive_rate: np.ndarray,
                             title='ROC Curve', width=700, height=500,
                             template='plotly_dark'
                             ):
    fig = px.area(x=false_positive_rate, y=true_positive_rate,
                    title=f'{title} (AUC={auc(false_positive_rate, true_positive_rate):.4f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                    width=width, height=height, template=template
                )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    return fig


def plot_test_data_precision_recall(recall, precision,
                                    title: str='Precision-Recall Curve', 
                                    width=700, height=500,
                                    template='plotly_dark'
                                    ):
    fig = px.area(x=recall, y=precision,
                 title=f'{title}',
                 labels=dict(x='Recall', y='Precision'),
                 width=width, height=height, template=template
                )
    fig.add_shape(type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=1, y1=0
                )
    return fig


def plot_table(data, header_fill_color='paleturquoise', header_align='left',
               cells_align='left', cell_color='lavender'):
    fig = go.Figure(data=[go.Table(header=dict(values=list(data.columns),
                                            fill_color=header_fill_color,
                                            align=header_align
                                            ),
                                  cells=dict(values=data.transpose().values.tolist(),
                                            fill_color=cell_color,
                                            align=cells_align
                                            )
                                )
                        ]
                  )
    return fig    

