import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="BChS Research Dashboard v5.1", layout="wide")


# ------------------------- Utilities -------------------------
def convert_fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    return buf.getvalue()


def _build_pi(c: int, a: float) -> np.ndarray:
    off_diag = (1 - a) / (c - 1) if c > 1 else 0.0
    pi = np.full((c, c), off_diag, dtype=float)
    np.fill_diagonal(pi, a)
    return pi


def _eigenmode_vector(c: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """
    Returns a vector v of length c with max(abs(v)) = 1.
    Modes:
      - "Uniform (consensus)" : v = (1,1,...,1)
      - "Contrast (balanced)" : v = (+1...+1,-1...-1) (requires even c)
      - "Random contrast (orthogonal to uniform)" : random vector orthogonal to 1
    """
    if mode == "Uniform (consensus)":
        v = np.ones(c, dtype=float)

    elif mode == "Contrast (balanced)":
        v = np.ones(c, dtype=float)
        v[c // 2 :] = -1.0

    elif mode == "Random contrast (orthogonal to uniform)":
        v = rng.standard_normal(c)
        # Remove the uniform component: v <- v - mean(v)*1
        v = v - np.mean(v)
        # If it accidentally becomes tiny, fallback
        if np.max(np.abs(v)) < 1e-12:
            v = np.ones(c, dtype=float)
            v[c // 2 :] = -1.0

    else:
        # Safe default
        v = np.ones(c, dtype=float)

    v = v / (np.max(np.abs(v)) + 1e-12)
    return v


# ------------------------- 1. Math Engine -------------------------
def run_bchs(
    c: int,
    a: float,
    p: float,
    steps: int,
    init_type: str,
    pos_count: int | None = None,
    noise_sigma: float = 0.0,
    noise_seed: int = 0,
    eigen_eps: float = 0.0,
    eigen_mode: str = "Uniform (consensus)",
    eigen_seed: int = 0,
    keep_mean_zero_after_noise: bool = False,
):
    dt = 0.1
    pi = _build_pi(c, a)

    # ---- Initial conditions for m ----
    rng_noise = np.random.default_rng(noise_seed)
    rng_eig = np.random.default_rng(eigen_seed)

    m = np.zeros(c, dtype=float)

    if init_type == "Symmetric (All +1)":
        m[:] = 0.5

    elif init_type == "Anti-Symmetric (Balanced)":
        m[: c // 2] = 0.5
        m[c // 2 :] = -0.5

    elif init_type == "Custom Count" and pos_count is not None:
        m[:pos_count] = 0.5
        m[pos_count:] = -0.5

    elif init_type == "Disordered (Noise)":
        m = rng_noise.uniform(-0.1, 0.1, c)

    elif init_type == "Balanced + Gaussian Noise (sigma)":
        # Start from perfectly balanced then add controlled noise
        m[: c // 2] = 0.5
        m[c // 2 :] = -0.5
        m = m + noise_sigma * rng_noise.standard_normal(c)
        if keep_mean_zero_after_noise:
            m = m - np.mean(m)

    elif init_type == "Balanced + Eigenmode Seed (eps)":
        # Start from balanced then add eps * eigenmode_vector
        m[: c // 2] = 0.5
        m[c // 2 :] = -0.5
        v = _eigenmode_vector(c, eigen_mode, rng_eig)
        m = m + eigen_eps * v

    else:
        # Fallback (old behavior)
        m = rng_noise.uniform(-0.1, 0.1, c)

    # ---- Initial condition for s ----
    s = np.full(c, 0.66, dtype=float)

    # Histories / observables
    m_hist = np.zeros((steps, c), dtype=float)
    g_list, p_list, v_list = [], [], []

    # ---- Integrate ODE ----
    for t in range(steps):
        M = pi @ m
        S = pi @ s

        dm = (1 - 2 * p) * (1 - s / 2) * M - (S / 2) * m
        ds = S * (1 - 1.5 * s) + (0.5 * (1 - 2 * p)) * m * M

        m = m + dm * dt
        s = s + ds * dt

        # Physical constraints: |m_g| <= s_g and 0<=s_g<=1 (optional clamp for stability)
        s = np.clip(s, 0.0, 1.0)
        m = np.clip(m, -s, s)

        m_hist[t] = m
        g_list.append(float(np.abs(np.mean(m))))
        p_list.append(float(np.mean(np.abs(m))))
        v_list.append(float(np.var(m)))

    return m_hist, {"global": g_list, "polar": p_list, "var": v_list}


# ------------------------- 2. Sidebar & Mode Selection -------------------------
st.sidebar.title("🔬 BChS Control Panel")
analysis_mode = st.sidebar.radio("Navigation", ["Dynamics", "Phase Space Explorer"])

c_slider = st.sidebar.slider("Number of Groups (c)", 2, 30, 6, step=2)

st.sidebar.subheader("Initial Conditions")
init_options = [
    "Anti-Symmetric (Balanced)",
    "Symmetric (All +1)",
    "Disordered (Noise)",
    "Custom Count",
    "Balanced + Gaussian Noise (sigma)",
    "Balanced + Eigenmode Seed (eps)",
]
init_type = st.sidebar.selectbox("Initial State", init_options)

pos_count = None
if init_type == "Custom Count":
    pos_count = st.sidebar.slider("Number of Positive Groups", 0, c_slider, c_slider // 2)

# Controls for new IC options
noise_sigma = 0.0
noise_seed = 0
keep_mean_zero_after_noise = False

eigen_eps = 0.0
eigen_seed = 0
eigen_mode = "Uniform (consensus)"

if init_type == "Balanced + Gaussian Noise (sigma)":
    st.sidebar.markdown("**Noise controls**")
    noise_sigma = st.sidebar.slider("Gaussian σ", 0.0, 0.2, 0.01, step=0.005)
    noise_seed = st.sidebar.number_input("Noise seed", min_value=0, max_value=10_000_000, value=0, step=1)
    keep_mean_zero_after_noise = st.sidebar.checkbox("Recenter to keep ⟨m⟩=0 after noise", value=False)

if init_type == "Balanced + Eigenmode Seed (eps)":
    st.sidebar.markdown("**Eigenmode seed controls**")
    eigen_mode = st.sidebar.selectbox(
        "Eigenmode type",
        ["Uniform (consensus)", "Contrast (balanced)", "Random contrast (orthogonal to uniform)"],
    )
    eigen_eps = st.sidebar.slider("Seed amplitude ε", 0.0, 0.2, 0.01, step=0.005)
    eigen_seed = st.sidebar.number_input("Eigen seed", min_value=0, max_value=10_000_000, value=0, step=1)


# ------------------------- 3. Dynamics Mode -------------------------
if analysis_mode == "Dynamics":
    st.sidebar.divider()
    a_val = st.sidebar.slider("Self-mixing (a)", 0.0, 1.0, 0.9)
    p_val = st.sidebar.slider("Contrarian Prob (p)", 0.0, 1.0, 0.1)
    steps = st.sidebar.number_input("Timesteps", 100, 5000, 1000)

    st.title("Network Dynamics & Trajectories")
    m_hist, ops = run_bchs(
        c_slider,
        a_val,
        p_val,
        steps,
        init_type,
        pos_count=pos_count,
        noise_sigma=noise_sigma,
        noise_seed=noise_seed,
        eigen_eps=eigen_eps,
        eigen_mode=eigen_mode,
        eigen_seed=eigen_seed,
        keep_mean_zero_after_noise=keep_mean_zero_after_noise,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Magnetization Evolution ($m_g$)")
        fig1, ax1 = plt.subplots()
        ax1.plot(m_hist, alpha=0.7)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Group Magnetizations ($m_g$)")
        st.pyplot(fig1)

        st.subheader("Heatmap")
        fig2, ax2 = plt.subplots()
        im = ax2.imshow(
            m_hist.T,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ticks = range(0, c_slider, 5 if c_slider > 20 else 1)
        ax2.set_yticks(list(ticks))
        ax2.set_yticklabels(list(ticks))
        plt.colorbar(im, label="$m_g$", ax=ax2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Group Index")
        st.pyplot(fig2)

    with col2:
        st.subheader("Order Parameter Trajectories")
        fig3, ax3 = plt.subplots()
        ax3.plot(ops["global"], label="Global Consensus $|\\langle m_g\\rangle|$")
        ax3.plot(ops["polar"], label="Group Polarization $\\langle|m_g|\\rangle$")
        ax3.plot(ops["var"], label="Variance $Var(m_g)$", ls="--")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend()
        ax3.grid(True, alpha=0.2)
        st.pyplot(fig3)
        st.download_button("💾 Save Order Params PNG", convert_fig_to_png(fig3), "order_params.png")


# ------------------------- 4. Phase Space Explorer Mode -------------------------
else:
    st.title("Phase Boundary Analysis")

    p_range = st.sidebar.slider("P-range", 0.0, 1.0, (0.0, 0.5))
    a_range = st.sidebar.slider("A-range", 0.0, 1.0, (0.0, 1.0))
    res = st.sidebar.number_input("Resolution (NxN Grid)", 10, 60, 25)

    st.sidebar.markdown("---")
    st.sidebar.caption("Phase scan uses the selected initial condition (including σ/ε controls).")

    col_btn1, col_btn2 = st.columns([1, 4])
    if col_btn1.button("🚀 Calculate"):
        ps = np.linspace(p_range[0], p_range[1], res)
        as_ = np.linspace(a_range[0], a_range[1], res)

        g_map = np.zeros((res, res), dtype=float)
        p_map = np.zeros((res, res), dtype=float)
        v_map = np.zeros((res, res), dtype=float)

        progress = st.progress(0.0)
        for i, pv in enumerate(ps):
            for j, av in enumerate(as_):
                _, op = run_bchs(
                    c_slider,
                    av,
                    pv,
                    800,
                    init_type,
                    pos_count=pos_count,
                    noise_sigma=noise_sigma,
                    noise_seed=noise_seed,
                    eigen_eps=eigen_eps,
                    eigen_mode=eigen_mode,
                    eigen_seed=eigen_seed,
                    keep_mean_zero_after_noise=keep_mean_zero_after_noise,
                )
                g_map[i, j] = op["global"][-1]
                p_map[i, j] = op["polar"][-1]
                v_map[i, j] = op["var"][-1]
            progress.progress((i + 1) / res)

        st.session_state["phase_data"] = {
            "g_map": g_map,
            "p_map": p_map,
            "v_map": v_map,
            "p_lims": p_range,
            "a_lims": a_range,
            "c_used": c_slider,
        }

    if "phase_data" in st.session_state:
        data = st.session_state["phase_data"]
        st.success(f"Results for $c={data['c_used']}$")

        fig_p, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Theory curves computed over the shown a-range
        a0 = max(data["a_lims"][0], 1e-6)
        a1 = max(data["a_lims"][1], a0 + 1e-6)
        a_theory = np.linspace(a0, a1, 400)

        c_used = data["c_used"]
        lam = (a_theory * c_used - 1) / (c_used - 1)

        # ---- Green line (existence) ----
        p_green = np.full_like(a_theory, np.nan, dtype=float)
        mask_lam = lam > 0
        p_tmp = 0.5 * (1 - 1 / (2 * lam[mask_lam]))
        # keep only within chosen p-range for clean plotting
        mask_pg = (p_tmp >= data["p_lims"][0]) & (p_tmp <= data["p_lims"][1])
        p_green[mask_lam] = np.nan
        p_green[np.where(mask_lam)[0][mask_pg]] = p_tmp[mask_pg]

        # ---- Blue line (refined stability from your corrected quadratic) ----
        def get_p_stable_refined(a_v: float, c_v: int) -> float:
            # lam_v must be > 0
            lam_v = (a_v * c_v - 1) / (c_v - 1)
            if not np.isfinite(lam_v) or lam_v <= 0:
                return np.nan

            # Quadratic: (1+lam)q^2 + (2*lam^2 + lam - 2)q - lam^2 = 0
            A = (1 + lam_v)
            B = (2 * lam_v**2 + lam_v - 2)
            C = -(lam_v**2)
            roots = np.roots([A, B, C])
            real_roots = roots[np.isreal(roots)].real
            if real_roots.size == 0:
                return np.nan
            q_c = np.max(real_roots)

            p_val = 0.5 * (1 - q_c / lam_v)
            if not np.isfinite(p_val):
                return np.nan
            if p_val < data["p_lims"][0] or p_val > data["p_lims"][1]:
                return np.nan
            return p_val

        p_blue = np.array([get_p_stable_refined(v, c_used) for v in a_theory], dtype=float)

        maps = [data["g_map"], data["p_map"], data["v_map"]]
        titles = ["Global Consensus", "Group Polarization", "Variance Map"]

        for i in range(3):
            im = axs[i].imshow(
                maps[i],
                origin="lower",
                extent=[data["a_lims"][0], data["a_lims"][1], data["p_lims"][0], data["p_lims"][1]],
                aspect="auto",
                cmap="inferno",
            )
            axs[i].plot(a_theory, p_green, color="lime", lw=2, ls="--", label="Existence")
            axs[i].plot(a_theory, p_blue, color="dodgerblue", lw=2, label="Stability")

            axs[i].set_xlim(data["a_lims"][0], data["a_lims"][1])
            axs[i].set_ylim(data["p_lims"][0], data["p_lims"][1])
            axs[i].set_title(titles[i])
            plt.colorbar(im, ax=axs[i])

            if i == 0:
                axs[i].legend(loc="upper left", fontsize="small")

        st.pyplot(fig_p)
        st.download_button("💾 Save Maps PNG", convert_fig_to_png(fig_p), "phase_maps.png")
