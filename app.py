import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(page_title="BChS SBM Dashboard", layout="wide")

# Helper for high-res PNG export
def convert_fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    return buf.getvalue()

# --- 1. Math Engine ---
def run_bchs(c, a, p, steps, init_type, pos_count=None):
    dt = 0.1
    off_diag = (1 - a) / (c - 1) if c > 1 else 0
    pi = np.full((c, c), off_diag)
    np.fill_diagonal(pi, a)
    
    m = np.zeros(c)
    if init_type == "Symmetric (All +1)": 
        m = np.full(c, 0.5)
    elif init_type == "Anti-Symmetric (Balanced)": 
        m[:c//2], m[c//2:] = 0.5, -0.5
    elif init_type == "Custom Count" and pos_count is not None:
        m[:pos_count] = 0.5
        m[pos_count:] = -0.5
    else: 
        m = np.random.uniform(-0.1, 0.1, c)
        
    s = np.full(c, 0.66)
    m_hist = np.zeros((steps, c))
    g_list, p_list, v_list = [], [], []
    
    for t in range(steps):
        M, S = pi @ m, pi @ s
        dm = (1 - 2*p) * (1 - s/2) * M - (S/2) * m
        ds = S * (1 - 1.5 * s) + (0.5 * (1 - 2*p)) * m * M
        m, s = m + dm * dt, s + ds * dt
        m = np.clip(m, -s, s)
        m_hist[t] = m
        g_list.append(np.abs(np.mean(m)))
        p_list.append(np.mean(np.abs(m)))
        v_list.append(np.var(m))
        
    return m_hist, {"global": g_list, "polar": p_list, "var": v_list}

# --- 2. Sidebar Controls ---
st.sidebar.title("🔬 Controls")
analysis_mode = st.sidebar.radio("Navigation", ["Dynamics", "Phase Space Explorer"])

c_slider = st.sidebar.slider("Number of Groups (c)", 2, 30, 6, step=2)

st.sidebar.subheader("Initial Conditions")
init_options = ["Anti-Symmetric (Balanced)", "Symmetric (All +1)", "Disordered (Noise)", "Custom Count"]
init_type = st.sidebar.selectbox("Initial State", init_options)

pos_count = None
if init_type == "Custom Count":
    pos_count = st.sidebar.slider("Number of Positive Groups", 0, c_slider, c_slider//2)

# Conditional logic for Dynamics
if analysis_mode == "Dynamics":
    st.sidebar.divider()
    a_val = st.sidebar.slider("Self-mixing (a)", 0.0, 1.0, 0.9)
    p_val = st.sidebar.slider("Contrarian Prob (p)", 0.0, 1.0, 0.1)
    steps = st.sidebar.number_input("Timesteps", 100, 5000, 1000)

# --- 3. Live Dynamics Mode ---
if analysis_mode == "Dynamics":
    st.title("Network Dynamics & Trajectories")
    m_hist, ops = run_bchs(c_slider, a_val, p_val, steps, init_type, pos_count)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Magnetization Evolution ($m_g$)")
        fig1, ax1 = plt.subplots()
        ax1.plot(m_hist, alpha=0.7)
        ax1.set_xlabel("Time Steps"); ax1.set_ylabel("Group Magnetizations ($m_g$)")
        st.pyplot(fig1)
        st.download_button("💾 Save PNG", convert_fig_to_png(fig1), "magnetization.png")

        st.subheader("Spatiotemporal Heatmap")
        fig2, ax2 = plt.subplots()
        im = ax2.imshow(m_hist.T, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label="Group Magnetizations ($m_g$)", ax=ax2)
        ax2.set_xlabel("Time Steps"); ax2.set_ylabel("Groups ($g$)")
        st.pyplot(fig2)
        st.download_button("💾 Save PNG", convert_fig_to_png(fig2), "heatmap.png")

    with col2:
        st.subheader("Order Parameters Trajectories")
        fig3, ax3 = plt.subplots()
        ax3.plot(ops["global"], label="Global Consensus $|⟨m_g⟩|$")
        ax3.plot(ops["polar"], label="Group Consensus $⟨|m_g|⟩$")
        ax3.plot(ops["var"], label="Polarization $Var(m_g)$", ls='--')
        ax3.set_ylim(-0.05, 1.05); ax3.legend(); ax3.grid(True, alpha=0.2)
        st.pyplot(fig3)
        st.download_button("💾 Save PNG", convert_fig_to_png(fig3), "order_params.png")

# --- 4. Phase Space Explorer Mode ---
else:
    st.title("Phase Boundary Analysis")
    p_min_s = st.sidebar.slider("P-range", 0.0, 1.0, (0.0, 0.5))
    a_min_s = st.sidebar.slider("A-range", 0.0, 1.0, (0.0, 1.0))
    res_s = st.sidebar.number_input("Resolution (NxN Grid)", 10, 60, 25)
    
    # We use a button to trigger the calculation
    calc_button = st.button("🚀 Calculate Phase Maps")
    
    if calc_button or 'phase_data' in st.session_state:
        # Check if we need to run or re-run the simulation
        # We only run if the button is pressed OR if data doesn't exist yet
        if calc_button or 'phase_data' not in st.session_state:
            ps = np.linspace(p_min_s[0], p_min_s[1], res_s)
            as_ = np.linspace(a_min_s[0], a_min_s[1], res_s)
            g_map, p_map, v_map = np.zeros((res_s, res_s)), np.zeros((res_s, res_s)), np.zeros((res_s, res_s))
            
            progress = st.progress(0)
            for i, pv in enumerate(ps):
                for j, av in enumerate(as_):
                    _, op = run_bchs(c_slider, av, pv, 800, init_type, pos_count)
                    # We store the steady state (last value)
                    g_map[i, j] = op["global"][-1]
                    p_map[i, j] = op["polar"][-1]
                    v_map[i, j] = op["var"][-1]
                progress.progress((i+1)/res_s)
            
            # STORE EVERYTHING: Data AND the parameters used to generate it
            st.session_state['phase_data'] = {
                'g_map': g_map, 'p_map': p_map, 'v_map': v_map,
                'ps': ps, 'as': as_,
                'p_lims': p_min_s, 'a_lims': a_min_s,
                'c_used': c_slider
            }

        # Retrieve the data from session state
        data = st.session_state['phase_data']
        fig_p, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # Calculate
