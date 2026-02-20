import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. UI Configuration
st.set_page_config(page_title="BChS PhD Tool", layout="wide")
st.title("BChS Modular State Simulator")

# 2. Sidebar Parameters
with st.sidebar:
    st.header("Network Settings")
    c = st.slider("Number of Groups (c)", 2, 20, 4, step=2)
    a = st.slider("Self-mixing (a)", 0.5, 1.0, 0.95)
    
    st.header("Dynamics Settings")
    p = st.slider("Contrarian Prob (p)", 0.0, 0.5, 0.10, step=0.01)
    
    st.header("Simulation Settings")
    steps = st.number_input("Timesteps", 100, 5000, 1000)
    dt = 0.1

# 3. The Math Engine
def run_simulation(c, a, p, steps, dt):
    # Mixing Matrix
    off_diag = (1 - a) / (c - 1) if c > 1 else 0
    pi = np.full((c, c), off_diag)
    np.fill_diagonal(pi, a)
    
    # Init: Break symmetry with a small kick
    m = np.zeros(c)
    m[:c//2], m[c//2:] = 0.05, -0.05
    s = np.full(c, 0.66)
    
    m_hist = np.zeros((steps, c))
    s_hist = np.zeros((steps, c))
    
    for t in range(steps):
        M, S = pi @ m, pi @ s
        
        # Differential equations
        dm = (1 - 2*p) * (1 - s/2) * M - (S/2) * m
        ds = S * (1 - 1.5 * s) + (0.5 * (1 - 2*p)) * m * M
        
        m += dm * dt
        s += ds * dt
        
        # Stability Clips
        s = np.clip(s, 0.01, 0.99)
        m = np.clip(m, -s, s)
        
        m_hist[t] = m
        s_hist[t] = s
        
    return m_hist, s_hist

# 4. Run and Display
# No "If button" here so it updates live as you slide!
m_data, s_data = run_simulation(c, a, p, steps, dt)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Magnetization ($m_g$)")
    fig_m, ax_m = plt.subplots(figsize=(6, 4))
    colors = ['#1f77b4' if i < c/2 else '#d62728' for i in range(c)]
    for i in range(c):
        ax_m.plot(m_data[:, i], color=colors[i], alpha=0.7)
    ax_m.set_xlabel("Time")
    ax_m.set_ylabel("Order Parameter")
    ax_m.set_ylim(-1, 1)
    ax_m.grid(True, alpha=0.3)
    st.pyplot(fig_m)

with col2:
    st.subheader("Active Density ($s_g$)")
    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    for i in range(c):
        ax_s.plot(s_data[:, i], color='green', alpha=0.3)
    ax_s.set_xlabel("Time")
    ax_s.set_ylabel("Activity")
    ax_s.set_ylim(0, 1)
    ax_s.grid(True, alpha=0.3)
    st.pyplot(fig_s)

# 5. Theoretical Context
lambda_c = (a*c - 1)/(c - 1)
p_crit = 0.5 * (1 - 1/(2*lambda_c))

st.info(f"For c={c}, a={a}, the **Modular Eigenvalue** $\lambda$ is **{lambda_c:.3f}**. "
        f"Theoretical bifurcation occurs at **p = {p_crit:.3f}**.")
