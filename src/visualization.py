import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_beam_3d(length_mm, width_mm, height_mm, deflection_mm):
    """
    Creates a 3D Plotly figure representing a beam with deflection.
    The deflection is exaggerated for visualization purposes.
    """
    # Exaggerate deflection for visibility
    vis_scale = 1.0
    if deflection_mm > 0:
        target_vis_defl = height_mm * 1.5
        vis_scale = target_vis_defl / (deflection_mm + 1e-9)
        vis_scale = min(vis_scale, 1000) 
    
    scaled_defl = deflection_mm * vis_scale
    
    n_seg = 20
    x = np.linspace(0, length_mm, n_seg + 1)
    
    # Simple parabolic deflection curve
    z_offset = (4 * scaled_defl / (length_mm**2 + 1e-9)) * x * (length_mm - x)
    
    verts = []
    w2 = width_mm / 2.0
    for i in range(len(x)):
        xi = x[i]
        zi = z_offset[i]
        verts.append([xi, -w2, -zi])          
        verts.append([xi,  w2, -zi])          
        verts.append([xi, -w2, -zi - height_mm]) 
        verts.append([xi,  w2, -zi - height_mm]) 
    
    verts = np.array(verts)
    
    i_list, j_list, k_list = [], [], []
    for s in range(n_seg):
        o = s * 4
        n = (s + 1) * 4
        def add_quad(p1, p2, p3, p4):
            i_list.extend([p1, p2, p3]); j_list.extend([p2, p4, p4]); k_list.extend([p3, p3, p1])
            i_list.extend([p1, p3, p4]); j_list.extend([p3, p4, p2]); k_list.extend([p4, p2, p1])
        
        add_quad(o+0, o+1, n+0, n+1) 
        add_quad(o+2, o+3, n+2, n+3) 
        add_quad(o+0, o+2, n+0, n+2) 
        add_quad(o+1, o+3, n+1, n+3) 
    
    add_quad(0, 1, 2, 3) 
    add_quad(n_seg*4+0, n_seg*4+1, n_seg*4+2, n_seg*4+3) 

    colors = np.abs(verts[:, 0] - length_mm/2)
    colors = 1.0 - (colors / (length_mm/2)) 
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=i_list, j=j_list, k=k_list,
            intensity=colors,
            colorscale='Viridis',
            showscale=False,
            name='Beam'
        )
    ])
    
    fig.add_trace(go.Scatter3d(
        x=[0, length_mm], y=[0, 0], z=[0, 0],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Supports'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Length [mm]', 
                range=[-length_mm * 0.1, length_mm * 1.1],
                backgroundcolor="rgb(230, 230, 230)",
                showbackground=True,
                gridcolor="white",
                zerolinecolor="white",
            ),
            yaxis=dict(
                title='Width [mm]', 
                range=[-width_mm * 2.0, width_mm * 2.0],
                backgroundcolor="rgb(220, 220, 220)",
                showbackground=True,
                gridcolor="white",
                zerolinecolor="white",
            ),
            zaxis=dict(
                title='Height [mm]', 
                range=[-(scaled_defl + height_mm) * 1.5, height_mm * 1.5],
                backgroundcolor="rgb(220, 220, 220)",
                showbackground=True,
                gridcolor="white",
                zerolinecolor="white",
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                projection=dict(type='orthographic')
            ),
            bgcolor='rgba(0,0,0,0)',
            aspectratio=dict(x=3, y=1, z=1),
            aspectmode='manual'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_doe_distribution_plotly(df_smart, df_bad):
    """
    Creates a 2D comparison plot of the input distributions
    to show Clustered (Bad) vs LHS (Smart) space-filling.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bad DoE (Clustered)", "Smart DoE (LHS)"))
    
    # Bad DoE
    fig.add_trace(
        go.Scatter(x=df_bad["length_mm"], y=df_bad["youngs_modulus_gpa"], mode='markers',
                   marker=dict(color='red', size=10, line=dict(width=1, color='black')),
                   name='Bad DoE'),
        row=1, col=1
    )
    
    # Smart DoE
    fig.add_trace(
        go.Scatter(x=df_smart["length_mm"], y=df_smart["youngs_modulus_gpa"], mode='markers',
                   marker=dict(color='green', size=8, line=dict(width=1, color='black')),
                   name='Smart DoE'),
        row=1, col=2
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgb(220, 220, 220)',
        font=dict(color="white"),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Common axis styles
    axis_style = dict(gridcolor="white", zerolinecolor="white", tickfont=dict(color="white"), title_font=dict(color="white"))
    
    fig.update_xaxes(title_text="Length [mm]", **axis_style, row=1, col=1)
    fig.update_yaxes(title_text="Youngs Modulus [GPa]", **axis_style, row=1, col=1)
    fig.update_xaxes(title_text="Length [mm]", **axis_style, row=1, col=2)
    fig.update_yaxes(title_text="Youngs Modulus [GPa]", **axis_style, row=1, col=2)
    
    return fig

def plot_accuracy_comparison_plotly(test_df, y_preds_dict, target_label):
    """
    Creates a scatter plot comparison: True value vs Predicted value.
    y_preds_dict: dict { "Model Name": array_of_predictions_for_the_specific_target }
    """
    fig = go.Figure()
    
    # Calculate global min/max for the 45-degree line
    all_vals = test_df[target_label].tolist()
    for name, p in y_preds_dict.items():
        all_vals.extend(p.tolist())
    
    min_val, max_val = min(all_vals), max(all_vals)
    padding = abs(max_val - min_val) * 0.1
    range_min = min_val - padding
    range_max = max_val + padding

    # 45 degree line (Ideal)
    fig.add_trace(go.Scatter(
        x=[range_min, range_max], y=[range_min, range_max],
        mode='lines',
        line=dict(color='white', dash='dash', width=2),
        name='Ideal (100% Accuracy)'
    ))
    
    # Professional colors for the 4 variants
    colors = {
        "Bad DoE + XGBoost": "rgb(255, 100, 100)",      # Light Red
        "Bad DoE + NN": "rgb(255, 180, 100)",           # Orange
        "Smart DoE + XGBoost": "rgb(150, 255, 150)",    # Light Green
        "Smart DoE + NN": "rgb(100, 230, 255)"          # Cyan
    }

    for name, preds in y_preds_dict.items():
        # Add detailed hover information with input parameters
        hover_text = [
            f"<b>{name}</b><br>" +
            f"Wahr: {t:.2f}<br>KI: {p:.2f}<br>" +
            f"Abweichung: {abs(t-p):.2f} ({abs(t-p)/(t+1e-9)*100:.1f}%)<br>" +
            f"L: {l:.0f}mm | W: {w:.0f}mm | H: {h:.0f}mm"
            for t, p, l, w, h in zip(test_df[target_label], preds, test_df['length_mm'], test_df['width_mm'], test_df['height_mm'])
        ]

        fig.add_trace(go.Scatter(
            x=test_df[target_label],
            y=preds,
            mode='markers',
            text=hover_text,
            hoverinfo='text',
            marker=dict(size=10, color=colors.get(name, "white"), 
                        line=dict(width=1, color='black'),
                        opacity=0.7),
            name=name
        ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgb(220, 220, 220)',
        font=dict(color="white"),
        title=dict(text=f"Genauigkeit: True vs. Predicted ({target_label})", font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=100, b=40),
        height=600
    )
    
    axis_style = dict(gridcolor="white", zerolinecolor="white", tickfont=dict(color="white"), title_font=dict(color="white"))
    
    fig.update_xaxes(title_text=f"True {target_label}", **axis_style)
    fig.update_yaxes(title_text=f"Predicted {target_label}", **axis_style)
    
    return fig
