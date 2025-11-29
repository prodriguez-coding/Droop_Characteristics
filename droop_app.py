"""
Streamlit application to explore active power–frequency droop control
for two synchronous generators operating in parallel.

The app reproduces the two-scenario exercise we have been working with:
 - Scenario 1: system operates at nominal frequency, partial load.
 - Scenario 2: system operates at a frequency deviated from nominal
   when the load is different and the power sharing is changed.

From the input data the app computes:
 - Pref1, Pref2: reference powers at nominal frequency
 - KPF1, KPF2: droop gains in kW/(rad/s)
and it plots:
 - Droop characteristic of Generator 1 (red)
 - Droop characteristic of Generator 2 (blue)
 - Combined droop curve (total power vs frequency, grey dashed)

Author: Pedro Rodriguez (concept) + ChatGPT (implementation helper)
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# Core droop functions
# ============================================================

def freq_from_power(P: np.ndarray, P_ref: float, KPF: float, f_nom: float) -> np.ndarray:
    """
    Compute frequency from active power using a linear droop characteristic:

        f(P) = f_nom - (P - P_ref) / (KPF * 2*pi)

    Parameters
    ----------
    P : np.ndarray
        Active power values (kW).
    P_ref : float
        Reference power at nominal frequency (kW).
    KPF : float
        Droop gain in kW/(rad/s).
    f_nom : float
        Nominal system frequency (Hz).

    Returns
    -------
    np.ndarray
        Frequency values corresponding to P (Hz).
    """
    two_pi = 2.0 * np.pi
    return f_nom - (P - P_ref) / (KPF * two_pi)


def compute_droop_parameters(
    f_nom: float,
    P_load_nom: float,
    P_nom1: float,
    P_nom2: float,
    P_L1_pct: float,
    frac_G1_s1_pct: float,
    P_L2_pct: float,
    frac_G1_s2_pct: float,
    freq_err_pct: float,
):
    """
    Compute droop-related quantities from the two-scenario specification.

    Scenario 1 (at f_nom):
        - Load P_L1 = P_L1_pct * P_load_nom / 100
        - G1 supplies frac_G1_s1_pct % of P_L1
        - G2 supplies the remaining part
        -> Pref1 = P1_s1, Pref2 = P2_s1

    Scenario 2 (at f_op != f_nom):
        - Load P_L2 = P_L2_pct * P_load_nom / 100
        - G1 supplies frac_G1_s2_pct % of P_L2
        - G2 supplies the remaining part
        - f_op = f_nom * (1 + freq_err_pct/100)

    Using scenario 2 we solve:
        P_i(f_op) = P_i_s2 = Pref_i - KPF_i * 2*pi * (f_op - f_nom)

    for KPF_i.

    Returns
    -------
    results : dict
        Dictionary with scalar results (Pref1, Pref2, KPF1, KPF2, etc.).
    curves : dict
        Dictionary with arrays for plotting droop and combined curves.
    """

    two_pi = 2.0 * np.pi

    # ----- Scenario 1: nominal frequency -----
    P_L1 = (P_L1_pct / 100.0) * P_load_nom
    frac_G1_s1 = frac_G1_s1_pct / 100.0

    P1_s1 = frac_G1_s1 * P_L1
    P2_s1 = P_L1 - P1_s1

    Pref1 = P1_s1
    Pref2 = P2_s1

    # ----- Scenario 2: frequency deviation from nominal -----
    P_L2 = (P_L2_pct / 100.0) * P_load_nom
    frac_G1_s2 = frac_G1_s2_pct / 100.0

    P1_s2 = frac_G1_s2 * P_L2
    P2_s2 = P_L2 - P1_s2

    f_op = f_nom * (1.0 + freq_err_pct / 100.0)

    if abs(f_op - f_nom) < 1e-9:
        raise ValueError("Operating frequency in Scenario 2 must be different from f_nom.")

    # Droop gains KPF1 and KPF2
    KPF1 = (Pref1 - P1_s2) / (two_pi * (f_op - f_nom))
    KPF2 = (Pref2 - P2_s2) / (two_pi * (f_op - f_nom))

    # ----- Build droop curves for plotting -----
    # Individual generator curves
    P1_range = np.linspace(0.0, P_nom1, 300)
    P2_range = np.linspace(0.0, P_nom2, 300)

    f1_curve = freq_from_power(P1_range, Pref1, KPF1, f_nom)
    f2_curve = freq_from_power(P2_range, Pref2, KPF2, f_nom)

    # Key points of each droop characteristic
    f1_P0 = float(freq_from_power(np.array([0.0]),    Pref1, KPF1, f_nom)[0])
    f1_Pnom = float(freq_from_power(np.array([P_nom1]), Pref1, KPF1, f_nom)[0])

    f2_P0 = float(freq_from_power(np.array([0.0]),    Pref2, KPF2, f_nom)[0])
    f2_Pnom = float(freq_from_power(np.array([P_nom2]), Pref2, KPF2, f_nom)[0])

    # ----- Combined droop curve -----
    # Sweep frequency from slightly above no-load values down to below full load
    f_max = max(f1_P0, f2_P0, f_nom) + 2.0
    f_min = min(f1_Pnom, f2_Pnom, f_op) - 2.0
    f_grid = np.linspace(f_max, f_min, 400)

    # Raw power from each generator at these frequencies
    P1_raw = Pref1 - KPF1 * two_pi * (f_grid - f_nom)
    P2_raw = Pref2 - KPF2 * two_pi * (f_grid - f_nom)

    # Clip to rated ranges [0, P_nom]
    P1_clip = np.clip(P1_raw, 0.0, P_nom1)
    P2_clip = np.clip(P2_raw, 0.0, P_nom2)

    P_sum = P1_clip + P2_clip

    results = dict(
        f_nom=f_nom,
        f_op=f_op,
        P_load_nom=P_load_nom,
        P_nom1=P_nom1,
        P_nom2=P_nom2,
        P_L1=P_L1,
        P_L2=P_L2,
        P1_s1=P1_s1,
        P2_s1=P2_s1,
        P1_s2=P1_s2,
        P2_s2=P2_s2,
        Pref1=Pref1,
        Pref2=Pref2,
        KPF1=KPF1,
        KPF2=KPF2,
        f1_P0=f1_P0,
        f1_Pnom=f1_Pnom,
        f2_P0=f2_P0,
        f2_Pnom=f2_Pnom,
        P_L1_pct=P_L1_pct,
        P_L2_pct=P_L2_pct,
        frac_G1_s1_pct=frac_G1_s1_pct,
        frac_G1_s2_pct=frac_G1_s2_pct,
        freq_err_pct=freq_err_pct,
    )

    curves = dict(
        P1_range=P1_range,
        f1_curve=f1_curve,
        P2_range=P2_range,
        f2_curve=f2_curve,
        f_grid=f_grid,
        P_sum=P_sum,
    )

    return results, curves


# ============================================================
# Streamlit user interface
# ============================================================

st.set_page_config(
    page_title="ADVANCED CONTROL OF POWER SYSTEMS - Droop Control",
    layout="wide",
)

st.title("ADVANCED CONTROL OF POWER SYSTEMS")
st.subheader("Droop Control of Two Synchronous Generators")
st.markdown(
    """
Developed by *Prof. Pedro Rodriguez* this tool illustrates **active power–frequency droop control** for two synchronous
generators operating in parallel and sharing the system load.

You can:
- Define the **rated system load** and **rated power** of each generator.
- Specify two operating scenarios:
  - **Scenario 1**: the system operates at nominal frequency.
  - **Scenario 2**: the system operates at a frequency deviated from nominal,
    given in percent (e.g. -2% → 0.98 f_nom).
- Assuming the droop control equation for each generator is guven by:
    P_i(f) = P_ref_i - KPF_i * 2π * (f - f_nom)
- The app then computes:
  - The reference powers **P_ref1**, **P_ref2** and the droop gains **KPF1**, **KPF2** in kW/(rad/s).
  - The droop characteristics of both generators.
  - The **combined droop curve** (total power vs frequency).

Students shoukd modify the parameters and observe how the curves and operating points move.
"""
)

# ------------- Sidebar: parameters ---------------------------

st.sidebar.header("Configuration")

use_example = st.sidebar.checkbox(
    "Use example parameters (Pedro’s exercise)",
    value=True,
    help="If enabled, pre-loads the data from the reference example.",
)

if use_example:
    # Example data: this matches the exercise we have been working on.
    f_nom = 50.0
    P_load_nom = 100.0

    P_nom1 = 80.0
    P_nom2 = 60.0

    P_L1_pct = 50.0       # Scenario 1 load: 50% of rated
    frac_G1_s1_pct = 71.4 # ~35.7 kW out of 50 kW for G1

    P_L2_pct = 100.0      # Scenario 2 load: 100% of rated
    frac_G1_s2_pct = 50.0 # 50/50 sharing at scenario 2

    freq_err_pct = -2.0   # -2% → 0.98 * f_nom
else:
    st.sidebar.subheader("System")
    f_nom = st.sidebar.number_input("Nominal frequency f_nom [Hz]", value=50.0)
    P_load_nom = st.sidebar.number_input("Rated system load [kW]", value=100.0, min_value=1.0)

    st.sidebar.subheader("Generator ratings")
    P_nom1 = st.sidebar.number_input("Rated power Generator 1 [kW]", value=80.0, min_value=0.1)
    P_nom2 = st.sidebar.number_input("Rated power Generator 2 [kW]", value=60.0, min_value=0.1)

    st.sidebar.subheader("Scenario 1 (at nominal frequency)")
    P_L1_pct = st.sidebar.number_input(
        "Scenario 1 load level [% of rated load]", value=50.0, min_value=0.0, max_value=200.0
    )
    frac_G1_s1_pct = st.sidebar.number_input(
        "% of Scenario 1 load supplied by G1", value=70.0, min_value=0.0, max_value=100.0
    )

    st.sidebar.subheader("Scenario 2 (frequency deviation)")
    P_L2_pct = st.sidebar.number_input(
        "Scenario 2 load level [% of rated load]", value=100.0, min_value=0.0, max_value=300.0
    )
    frac_G1_s2_pct = st.sidebar.number_input(
        "% of Scenario 2 load supplied by G1", value=50.0, min_value=0.0, max_value=100.0
    )
    freq_err_pct = st.sidebar.number_input(
        "Frequency error in Scenario 2 [% of f_nom]",
        value=-2.0,
        help="Example: -2% → f_op = 0.98 f_nom.",
    )

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run droop analysis", type="primary")


# ============================================================
# Main computation and plotting
# ============================================================

if run_button:
    try:
        results, curves = compute_droop_parameters(
            f_nom=f_nom,
            P_load_nom=P_load_nom,
            P_nom1=P_nom1,
            P_nom2=P_nom2,
            P_L1_pct=P_L1_pct,
            frac_G1_s1_pct=frac_G1_s1_pct,
            P_L2_pct=P_L2_pct,
            frac_G1_s2_pct=frac_G1_s2_pct,
            freq_err_pct=freq_err_pct,
        )
    except Exception as exc:
        st.error(f"Error during computation: {exc}")
    else:
        # ----------------- Show numerical results -----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("System and scenarios")
            st.write(f"**Nominal frequency**: {results['f_nom']:.3f} Hz")
            st.write(
                f"**Operating frequency (Scenario 2)**: {results['f_op']:.3f} Hz "
                f"({results['freq_err_pct']:.2f} % of f_nom)"
            )
            st.write(f"**Rated system load**: {results['P_load_nom']:.2f} kW")
            st.write(f"**Rated power G1**: {results['P_nom1']:.2f} kW")
            st.write(f"**Rated power G2**: {results['P_nom2']:.2f} kW")

            st.markdown("**Scenario 1 (at nominal frequency)**")
            st.write(
                f"Load = {results['P_L1']:.2f} kW "
                f"({results['P_L1_pct']:.1f} % of rated load)"
            )
            st.write(
                f"G1: {results['P1_s1']:.2f} kW "
                f"({results['frac_G1_s1_pct']:.1f} % of P_L1)"
            )
            st.write(
                f"G2: {results['P2_s1']:.2f} kW "
                f"({100.0 - results['frac_G1_s1_pct']:.1f} % of P_L1)"
            )

            st.markdown("**Scenario 2 (with frequency deviation)**")
            st.write(
                f"Load = {results['P_L2']:.2f} kW "
                f"({results['P_L2_pct']:.1f} % of rated load)"
            )
            st.write(
                f"G1: {results['P1_s2']:.2f} kW "
                f"({results['frac_G1_s2_pct']:.1f} % of P_L2)"
            )
            st.write(
                f"G2: {results['P2_s2']:.2f} kW "
                f"({100.0 - results['frac_G1_s2_pct']:.1f} % of P_L2)"
            )

        with col2:
            st.subheader("Droop parameters")
            st.write(f"**P_ref1** (G1 reference power at f_nom): {results['Pref1']:.3f} kW")
            st.write(f"**P_ref2** (G2 reference power at f_nom): {results['Pref2']:.3f} kW")
            st.write(f"**KPF1**: {results['KPF1']:.4f} kW/(rad/s)")
            st.write(f"**KPF2**: {results['KPF2']:.4f} kW/(rad/s)")
            st.write(
                f"No-load frequency G1 (P = 0): {results['f1_P0']:.3f} Hz\n\n"
                f"No-load frequency G2 (P = 0): {results['f2_P0']:.3f} Hz"
            )
            st.write(
                f"Frequency at rated power G1 (P = P_nom1): {results['f1_Pnom']:.3f} Hz\n\n"
                f"Frequency at rated power G2 (P = P_nom2): {results['f2_Pnom']:.3f} Hz"
            )

        st.markdown("---")
        st.subheader("Droop characteristics and operating points")

        # ----------------- Build the matplotlib figure -----------------
        fig, ax = plt.subplots(figsize=(9, 5))

        # Convenience aliases
        P_nom1 = results["P_nom1"]
        P_nom2 = results["P_nom2"]
        P_load_nom = results["P_load_nom"]
        f_nom = results["f_nom"]
        f_op = results["f_op"]

        P_L1 = results["P_L1"]
        P_L2 = results["P_L2"]
        P1_s1 = results["P1_s1"]
        P2_s1 = results["P2_s1"]
        P1_s2 = results["P1_s2"]

        f1_P0 = results["f1_P0"]
        f1_Pnom = results["f1_Pnom"]
        f2_P0 = results["f2_P0"]
        f2_Pnom = results["f2_Pnom"]

        # Baseline slightly below the minimum frequency
        baseline = min(f1_Pnom, f2_Pnom, f_op) - 2.0

        # Individual droop curves
        ax.plot(curves["P1_range"], curves["f1_curve"], color="red", linewidth=2, label="Generator 1 droop")
        ax.plot(curves["P2_range"], curves["f2_curve"], color="blue", linewidth=2, label="Generator 2 droop")

        # Combined droop curve (total power vs frequency)
        ax.plot(curves["P_sum"], curves["f_grid"], linestyle="--", color="gray", linewidth=2,
                label="Combined droop (G1 + G2)")

        # Vertical saturation segments at rated powers
        ax.vlines(P_nom1, baseline, f1_Pnom, colors="red", linewidth=2)
        ax.vlines(P_nom2, baseline, f2_Pnom, colors="blue", linewidth=2)

        # Horizontal reference lines for nominal and operating frequency
        ax.axhline(f_nom, color="black", linestyle="--", linewidth=1)
        ax.axhline(f_op,  color="black", linestyle="--", linewidth=1)

        # Vertical dashed lines at key powers
        for x in [P2_s1, P1_s1, P1_s2, P_load_nom]:
            ax.vlines(x, baseline, f_nom, colors="black", linestyles="--", linewidth=1)

        # Labels for f_nom and f_op on the right of the figure
        x_right = max(P_nom1 + P_nom2, P_load_nom) * 1.15
        ax.text(x_right, f_nom + 0.05,
                rf"$f_{{\mathrm{{nom}}}} = {f_nom:.2f}\ \mathrm{{Hz}}$",
                fontsize=9, ha="right", va="bottom")
        ax.text(x_right, f_op + 0.05,
                rf"$f_{{\mathrm{{op}}}} = {f_op:.2f}\ \mathrm{{Hz}}$",
                fontsize=9, ha="right", va="bottom")

        # Markers at no-load points
        ax.scatter([0.0], [f1_P0], color="red")
        ax.scatter([0.0], [f2_P0], color="blue")

        # Markers at Scenario 1 points (G1 and G2 at f_nom)
        ax.scatter([P2_s1], [f_nom], color="blue")
        ax.scatter([P1_s1], [f_nom], color="red")

        # Marker at Scenario 2 point for G1
        ax.scatter([P1_s2], [f_op], color="black")

        # Markers at rated power frequencies for each generator
        ax.scatter([P_nom2], [f2_Pnom], color="blue")
        ax.scatter([P_nom1], [f1_Pnom], color="red")

        # Markers on the combined curve at Scenario 1 and 2 total loads
        ax.scatter([P_L1], [f_nom], color="gray")
        ax.scatter([P_L2], [f_op], color="gray")

        # ---------- Text annotations (generic but informative) ----------

        # No-load labels
        ax.text(1.0, f1_P0 + 0.0, f"(0 kW,\n {f1_P0:.2f} Hz)",
                color="red", fontsize=8, ha="left", va="bottom")
        ax.text(1.0, f2_P0 + 0.5, f"(0 kW,\n {f2_P0:.2f} Hz)",
                color="blue", fontsize=8, ha="left", va="top")

        # Scenario 1 sharing at nominal frequency
        ax.text(P2_s1 + 1.0, f_nom + 0.15,
                f"({P2_s1:.1f} kW,\n {f_nom:.1f} Hz)",
                color="blue", fontsize=8, ha="left", va="bottom")
        ax.text(P1_s1 + 1.0, f_nom + 0.15,
                f"({P1_s1:.1f} kW,\n {f_nom:.1f} Hz)",
                color="red", fontsize=8, ha="left", va="bottom")

        # Scenario 2 operating point for G1
        ax.text(P1_s2 + 1.0, f_op + 0.15,
                f"({P1_s2:.1f} kW,\n {f_op:.2f} Hz)",
                color="black", fontsize=8, ha="left", va="bottom")

        # Rated power frequency points
        ax.text(P_nom2 + 2.0, f2_Pnom - 0.2,
                f"({P_nom2:.1f} kW,\n {f2_Pnom:.2f} Hz)",
                color="blue", fontsize=8, ha="left", va="bottom")
        ax.text(P_nom1 + 2.0, f1_Pnom + 0.0,
                f"({P_nom1:.1f} kW,\n {f1_Pnom:.2f} Hz)",
                color="red", fontsize=8, ha="left", va="bottom")

        # Rated load vertical label
        ax.text(P_load_nom + 1.5, baseline + 0.1,
                f"Rated load {P_load_nom:.1f} kW",
                rotation=90, fontsize=8, ha="left", va="bottom", color="black")

        # Axis formatting
        ax.set_xlim(0, max(P_nom1 + P_nom2, P_load_nom) * 1.2)
        ax.set_ylim(baseline, f_nom + 3.0)
        ax.set_xlabel("Active Power P (kW)")
        ax.set_ylabel("Frequency f (Hz)")
        ax.grid(True, linestyle=":", linewidth=0.7)
        ax.legend(loc="upper right")

        fig.tight_layout()
        st.pyplot(fig)

        st.info(
            "Tip for teaching: ask students to change the load percentages and the "
            "frequency error in Scenario 2 and discuss how the droop gains and the "
            "location of the operating points change."
        )
else:
    st.info("Configure the parameters in the sidebar and click **Run droop analysis** to start.")
