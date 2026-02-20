import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set Page Config
st.set_page_config(page_title="BChS Phase Explorer", layout="wide")

def solve_steady_state(c, a, p, steps=1000):
    off_diag = (1 - a) / (c - 1) if c > 1 else 0
    pi = np.full((c, c), off_diag)
    np.fill_diagonal(pi, a)
    
    m = np.zeros(c)
    m[:c//2], m[c//2:] = 0.2, -0.2
    s = np.full(c, 2/3)
    dt = 0.1
    
    for _ in range(steps):
        M, S = pi @ m, pi @ s
        dm = (1 - 2*p) * (1 - s/2) * M - (S/2) * m
        ds = S * (1 - 1.5 * s) + (0.5 * (1 - 2*p)) * m * M
        m += dm * dt
        s += ds * dt
        # Early exit if collapsed to zero
        if np.abs(m).max() < 1e-5: return 0.0
        
    return np.abs(m).mean()

# --- Sidebar ---
st.sidebar.header("Model Configuration")
c_val = st.sidebar.slider("Number of Groups (c)", 2, 10, 4, step=2)
mode = st.sidebar.radio("View Mode", ["Live Simulator", "Phase Diagram (Heatmap)"])

if mode == "Live Simulator":
    st.title("Live BChS Dynamics")
    p_val = st.sidebar.slider("Contrarian Prob (p)", 0.0, 0.5, 0.15)
    a_val = st.sidebar.slider("Self-mixing (a)", 0.5, 1.0, 0.9)
    
    # (Insert the simulation and plotting logic from the previous code here)
    # ... [Same as previous script for brevity] ...

else:
    st.title("Phase Boundary Analysis")
    st.markdown("This heatmap shows final polarization. **Green line** = Existence; **Blue line** = Stability.")
    
    if st.button("Generate Heatmap (May take 10-20 seconds)"):
        p_range = np.linspace(0, 0.35, 30)
        a_range = np.linspace(0.51, 0.99, 30)
        grid = np.zeros((len(p_range), len(a_range)))

        progress = st.progress(0)
        for i, p in enumerate(p_range):
            for j, a in enumerate(a_range):
                grid[i, j] = solve_steady_state(c_val, a, p)
            progress.progress((i + 1) / len(p_range))

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, origin='lower', extent=[0.51, 0.99, 0, 0.35], aspect='auto', cmap='magma')
        plt.colorbar(im, label="Final Average Magnetization")
        
        # Theoretical Green Line
        a_theory = np.linspace(0.51, 0.99, 100)
        lambda_theory = (a_theory * c_val - 1) / (c_val - 1)
        p_green = 0.5 * (1 - 1/(2 * lambda_theory))
        ax.plot(a_theory, p_green, color='lime', linestyle='--', label="Existence (Green)")
        
        ax.set_xlabel("Self-mixing (a)")
        ax.set_ylabel("Noise (p)")
        ax.legend()
        st.pyplot(fig)
