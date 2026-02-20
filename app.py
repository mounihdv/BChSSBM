import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(page_title="BChS Research Dashboard", layout="wide")

# --- 1. Math Engine ---
def run_bchs(c, a, p, steps, init_type, custom_ratio=0.5):
    dt = 0.1
    off_diag = (1 - a) / (c - 1) if c > 1 else 0
    pi = np.full((c, c), off_diag)
    np.fill_diagonal(pi, a)
    
    # Initial Conditions logic
    m = np.zeros(c)
    if init_type == "Symmetric (All +1)":
        m = np.full(c, 0.5)
    elif init_type == "Anti-Symmetric (Balanced)":
        m[:c//2], m[c//2:] = 0.5, -0.5
    elif init_type == "Custom Ratio":
        split = int(c * custom_ratio)
        m[:split], m[split:] = 0.5, -0.5
    else: # Disordered/Noise
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

# --- 2. Sidebar & Global Settings ---
st.sidebar.title("🔬 BChS Model Controls")

# Global Settings for both modes
c = st.sidebar.slider("Number of Groups (c)", 2, 20, 6, step=2)
a_current = st.sidebar.slider("Global Self-mixing (a)", 0.5, 1.0, 0.9)

st.sidebar.subheader("Global Initial Conditions")
init_type = st.sidebar.selectbox("Initial State", 
    ["Anti-Symmetric (Balanced)", "Symmetric (All +1)", "Disordered (Noise)", "Custom Ratio"])

custom_ratio = 0.5
if init_type == "Custom Ratio":
    custom_ratio = st.sidebar.slider("Ratio of +ve groups", 0.0, 1.0, 0.5)

st.sidebar.divider()
analysis_mode = st.sidebar.radio("Navigation", ["Live Dynamics", "Phase Space Explorer"])

# --- 3. Live Dynamics Mode ---
if analysis_mode == "Live Dynamics":
    st.title("Network Dynamics & Trajectories")
    p = st.sidebar.slider("Contrarian Prob (p)", 0.0, 0.5, 0.1)
    
    m_hist, ops = run_bchs(c, a_current, p, 1000, init_type, custom_ratio)

    # Data Export
    df_export = pd.DataFrame(m_hist, columns=[f"Group_{i}" for i in range(c)])
    df_export["Global_Mag"] = ops["global"]
    df_export["Polarization"] = ops["polar"]
    df_export["Variance"] = ops["var"]
    csv = df_export.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download Data", data=csv, file_name="bchs_results.csv", mime="text/csv")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Magnetization Evolution ($m_g$)")
        fig1, ax1 = plt.subplots()
        ax1.plot(m_hist, alpha=0.7)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Magnetization")
        st.pyplot(fig1)

        st.subheader("Spatiotemporal Heatmap")
        fig2, ax2 = plt.subplots()
        im = ax2.imshow(m_hist.T, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, label="Opinion Intensity", ax=ax2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Group Index")
        st.pyplot(fig2)

    with col2:
        st.subheader("Order Parameter Trajectories")
        fig3, ax3 = plt.subplots()
        ax3.plot(ops["global"], label="Global $|⟨m_g⟩|$ (Consensus)", lw=2)
        ax3.plot(ops["polar"], label="Polarization $⟨|m_g|⟩$", lw=2)
        ax3.plot(ops["var"], label="Variance $Var(m_g)$", lw=2, ls='--')
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend()
        ax3.grid(True, alpha=0.2)
        st.pyplot(fig3)

# --- 4. Phase Space Explorer Mode ---
else:
    st.title("Phase Boundary Analysis")
    st.info(f"Currently scanning phase space using **{init_type}** initialization.")
    
    p_min, p_max = st.sidebar.slider("P-range", 0.0, 1.0, (0.0, 0.4))
    a_min, a_max = st.sidebar.slider("A-range", 0.0, 1.0, (0.5, 1.0))
    res = st.sidebar.number_input("Resolution (NxN Grid)", 10, 50, 20)
    
    if st.button("🚀 Calculate Phase Maps"):
        ps = np.linspace(p_min, p_max, res)
        as_ = np.linspace(a_min, a_max, res)
        g_map, p_map, v_map = np.zeros((res, res)), np.zeros((res, res)), np.zeros((res, res))
        
        progress = st.progress(0)
        for i, pv in enumerate(ps):
            for j, av in enumerate(as_):
                # Now uses the GLOBAL init_type and custom_ratio selected in sidebar
                _, op = run_bchs(c, av, pv, 800, init_type, custom_ratio)
                g_map[i, j] = op["global"][-1]
                p_map[i, j] = op["polar"][-1]
                v_map[i, j] = op["var"][-1]
            progress.progress((i+1)/res)
            
        fig_p, axs = plt.subplots(1, 3, figsize=(18, 5))
        maps = [g_map, p_map, v_map]
        titles = ["Global Consensus Map", "Polarization Map", "Variance Map"]
        
        # Overlay Theoretical Line
        a_theory = np.linspace(a_min, a_max, 100)
        lam = (a_theory * c - 1) / (c - 1)
        p_green = 0.5 * (1 - 1/(2 * np.maximum(lam, 0.501)))

        for i in range(3):
            im = axs[i].imshow(maps[i], origin='lower', extent=[a_min, a_max, p_min, p_max], aspect='auto', cmap='inferno')
            axs[i].plot(a_theory, p_green, color='cyan', lw=2, ls='--', label='Bifurcation Theory')
            axs[i].set_title(titles[i])
            axs[i].set_xlabel("Mixing (a)")
            axs[i].set_ylabel("Noise (p)")
            plt.colorbar(im, ax=axs[i])
            if i == 0: axs[i].legend(loc='upper left', fontsize='small')
            
        st.pyplot(fig_p)
