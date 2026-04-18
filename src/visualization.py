import numpy as np
import plotly.graph_objects as go

def plot_beam_3d(length_mm, width_mm, height_mm, deflection_mm):
    """
    Creates a 3D Plotly figure representing a beam with deflection.
    The deflection is exaggerated for visualization purposes.
    """
    # Exaggerate deflection for visibility (e.g. if deflection is 2mm on a 2000mm beam, it's invisible)
    # We want it to be noticeable, so we scale it relative to the height.
    # Max visible deflection should be around 1-2x the height of the beam.
    vis_scale = 1.0
    if deflection_mm > 0:
        target_vis_defl = height_mm * 1.5
        vis_scale = target_vis_defl / (deflection_mm + 1e-9)
        # Cap scale to avoid extreme distortion
        vis_scale = min(vis_scale, 1000) 
    
    scaled_defl = deflection_mm * vis_scale
    
    # Grid along length
    n_seg = 20
    x = np.linspace(0, length_mm, n_seg + 1)
    
    # Deflection curve (Simplified parabolic for simply supported beam: d(x) = k * x * (L-x))
    # Real formula is more complex, but for visualization x(L-x) is enough.
    # We normalize it so the peak is at deflection_mm.
    # Midpoint x = L/2 => L/2 * (L-L/2) = L^2/4.
    z_offset = (4 * scaled_defl / (length_mm**2 + 1e-9)) * x * (length_mm - x)
    
    # Vertices calculation
    # For each x, we have 4 points (rectangle cross section)
    # Corners: 
    # 0: (x, -w/2, -z_off)
    # 1: (x,  w/2, -z_off)
    # 2: (x, -w/2, -z_off - h)
    # 3: (x,  w/2, -z_off - h)
    
    verts = []
    w2 = width_mm / 2.0
    for i in range(len(x)):
        xi = x[i]
        zi = z_offset[i]
        verts.append([xi, -w2, -zi])          # Top Left
        verts.append([xi,  w2, -zi])          # Top Right
        verts.append([xi, -w2, -zi - height_mm]) # Bottom Left
        verts.append([xi,  w2, -zi - height_mm]) # Bottom Right
    
    verts = np.array(verts)
    
    # Mesh triangles
    i_idx = []
    j_idx = []
    k_idx = []
    
    for s in range(n_seg):
        off = s * 4
        # Top face
        i_idx.extend([off+0, off+1, off+4])
        j_idx.extend([off+1, off+5, off+5])
        k_idx.extend([off+4, off+4, off+1]) # Wait, simpler triangulation:
        
        # Proper triangulation for 4 faces (top, bottom, left, right)
        # Top (0,1,4,5)
        i_idx.extend([off+0, off+1, off+4]); j_idx.extend([off+1, off+5, off+5]); k_idx.extend([off+4, off+4, off+0])
        i_idx.append(off+1); j_idx.append(off+5); k_idx.append(off+4) # Actually use 2 triangles per face
        
    # Let's use a cleaner manual mapping:
    i_list, j_list, k_list = [], [], []
    for s in range(n_seg):
        o = s * 4
        n = (s + 1) * 4
        # Faces connecting o and n
        # Top: o0, o1, n0, n1
        i_list.extend([o+0, o+1, n+0]); j_list.extend([o+1, n+1, n+0]); k_list.extend([n+0, n+0, o+0]) # wait logic is hard
        # Simplified:
        def add_quad(p1, p2, p3, p4):
            i_list.extend([p1, p2, p3]); j_list.extend([p2, p4, p4]); k_list.extend([p3, p3, p1])
            i_list.extend([p1, p3, p4]); j_list.extend([p3, p4, p2]); k_list.extend([p4, p2, p1]) # overlap but safe
        
        add_quad(o+0, o+1, n+0, n+1) # Top
        add_quad(o+2, o+3, n+2, n+3) # Bottom
        add_quad(o+0, o+2, n+0, n+2) # Left
        add_quad(o+1, o+3, n+1, n+3) # Right
    
    # Caps
    add_quad(0, 1, 2, 3) # Start
    add_quad(n_seg*4+0, n_seg*4+1, n_seg*4+2, n_seg*4+3) # End

    # Colors based on distance from center (stress proxy)
    # Center is at x = L/2
    colors = np.abs(verts[:, 0] - length_mm/2)
    colors = 1.0 - (colors / (length_mm/2)) # 1.0 at center, 0 at ends
    
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
    
    # Add support markers
    fig.add_trace(go.Scatter3d(
        x=[0, length_mm], y=[0, 0], z=[0, 0],
        mode='markers',
        marker=dict(size=10, color='red', symbol='cone'),
        name='Supports'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Length [mm]', range=[-500, max(length_mm, 3000)+500]),
            yaxis=dict(title='Width [mm]', range=[-500, 500]),
            zaxis=dict(title='Height [mm]', range=[-1000, 500]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    
    return fig
