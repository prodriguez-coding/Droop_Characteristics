import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper: safe float input with optional default
# ============================================================

def ask_float(prompt, default=None):
    """
    Ask the user for a float. If default is not None and the user
    just presses ENTER, the default value is returned.
    """
    if default is not None:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    while True:
        s = input(full_prompt)
        if not s.strip():
            if default is not None:
                return float(default)
            else:
                print("Please enter a number.")
                continue
        try:
            return float(s)
        except ValueError:
            print("Invalid number, try again.")


# ============================================================
# Droop helper
# ============================================================

def freq_from_power(P, P_ref, KPF, f_nom):
    """
    Droop characteristic: frequency as a function of active power.

    f(P) = f_nom - (P - P_ref) / (KPF * 2*pi)
    """
    two_pi = 2 * np.pi
    return f_nom - (P - P_ref) / (KPF * two_pi)


# ============================================================
# Main interactive routine
# ============================================================

def main():
    print("=== DROOP CHARACTERISTIC SETUP (INTERACTIVE) ===")
    print("All powers in kW, frequencies in Hz, shares in percent.\n")

    # --------------------------------------------------------
    # Mode selection: example vs fully interactive
    # --------------------------------------------------------
    use_example = input("Use built-in example (Pedro's case)? [Y/n]: ").strip().lower()
    example_mode = (use_example in ("", "y", "yes", "s", "si"))

    if example_mode:
        # --------------------------------------------
        # Pedro's example (hard-coded)
        # --------------------------------------------
        print("\nUsing example mode (Pedro's case)...\n")

        f_nom      = 50.0
        P_load_nom = 100.0

        P_nom1 = 80.0
        P_nom2 = 60.0

        # Scenario 1: 50% of rated load, G1 ≈ 71.4% of that
        P_L1_pct        = 50.0
        frac_G1_s1_pct  = 71.4

        # Scenario 2: 100% of rated load, G1 50%, error -2%
        P_L2_pct        = 100.0
        frac_G1_s2_pct  = 50.0
        freq_err_pct    = -2.0

    else:
        # --------------------------------------------
        # Fully interactive mode
        # --------------------------------------------
        # --- System-level data ---
        f_nom = ask_float("Nominal system frequency f_nom [Hz]", 50.0)
        P_load_nom = ask_float("Rated system load P_load_nom [kW]", 100.0)

        # --- Generator ratings ---
        P_nom1 = ask_float("Rated power of Generator 1, P_nom1 [kW]", 80.0)
        P_nom2 = ask_float("Rated power of Generator 2, P_nom2 [kW]", 60.0)

        # --- Scenario 1 ---
        print("\n--- Scenario 1 (system at nominal frequency f_nom) ---")
        print("Scenario 1 load is defined as a PERCENT of the rated load.")
        P_L1_pct = ask_float("Scenario 1: load level as % of rated load", 50.0)

        print("\nPower sharing in Scenario 1 (at f_nom):")
        print("Specify the PERCENT of P_L1 supplied by Generator 1.")
        frac_G1_s1_pct = ask_float("Scenario 1: % of P_L1 supplied by G1", 71.4)

        # --- Scenario 2 ---
        print("\n--- Scenario 2 (frequency deviation from f_nom) ---")
        print("Scenario 2 load is also defined as a PERCENT of the rated load.")
        P_L2_pct = ask_float("Scenario 2: load level as % of rated load", 100.0)

        print("\nPower sharing in Scenario 2:")
        print("Specify the PERCENT of P_L2 supplied by Generator 1.")
        frac_G1_s2_pct = ask_float("Scenario 2: % of P_L2 supplied by G1", 50.0)

        print("\nFrequency deviation in Scenario 2:")
        print("Specify the FREQUENCY ERROR in percent with respect to f_nom.")
        print("Example: -2 means f_op = 0.98 * f_nom.")
        freq_err_pct = ask_float("Scenario 2: frequency error [% of f_nom]", -2.0)

    # ========================================================
    # Compute scenario powers and droop gains
    # ========================================================

    # Convert percentages to absolute powers
    P_L1 = (P_L1_pct / 100.0) * P_load_nom
    P_L2 = (P_L2_pct / 100.0) * P_load_nom

    frac_G1_s1 = frac_G1_s1_pct / 100.0
    frac_G1_s2 = frac_G1_s2_pct / 100.0

    P1_s1 = frac_G1_s1 * P_L1
    P2_s1 = P_L1 - P1_s1

    P1_s2 = frac_G1_s2 * P_L2
    P2_s2 = P_L2 - P1_s2

    # Scenario 1 is at nominal frequency
    f_op = f_nom * (1.0 + freq_err_pct / 100.0)

    # Reference powers
    P_ref1 = P1_s1
    P_ref2 = P2_s1

    two_pi = 2 * np.pi

    if abs(f_op - f_nom) < 1e-6:
        raise ValueError("f_op must be different from f_nom to compute droop gains.")

    # From: P_i(f) = P_ref_i - KPF_i * 2*pi * (f - f_nom)
    KPF1 = (P_ref1 - P1_s2) / (two_pi * (f_op - f_nom))
    KPF2 = (P_ref2 - P2_s2) / (two_pi * (f_op - f_nom))

    # ========================================================
    # Print results
    # ========================================================
    print("\n=== RESULTS ===")
    print(f"f_nom              = {f_nom:.3f} Hz")
    print(f"f_op               = {f_op:.3f} Hz  (error {freq_err_pct:.2f} %)")
    print(f"Rated load         = {P_load_nom:.3f} kW")
    print(f"P_nom1 (G1)        = {P_nom1:.3f} kW")
    print(f"P_nom2 (G2)        = {P_nom2:.3f} kW\n")

    print(f"Scenario 1 load    = {P_L1:.3f} kW ({P_L1_pct:.1f} % of rated)")
    print(f"  G1: P1_s1        = {P1_s1:.3f} kW ({frac_G1_s1_pct:.1f} % of P_L1)")
    print(f"  G2: P2_s1        = {P2_s1:.3f} kW ({100 - frac_G1_s1_pct:.1f} % of P_L1)")
    print(f"  => P_ref1        = {P_ref1:.3f} kW")
    print(f"  => P_ref2        = {P_ref2:.3f} kW\n")

    print(f"Scenario 2 load    = {P_L2:.3f} kW ({P_L2_pct:.1f} % of rated)")
    print(f"  G1: P1_s2        = {P1_s2:.3f} kW ({frac_G1_s2_pct:.1f} % of P_L2)")
    print(f"  G2: P2_s2        = {P2_s2:.3f} kW ({100 - frac_G1_s2_pct:.1f} % of P_L2)\n")

    print(f"KPF1               = {KPF1:.4f} kW/(rad/s)")
    print(f"KPF2               = {KPF2:.4f} kW/(rad/s)")
    print("================================================\n")

    # ========================================================
    # Build droop curves and combined curve
    # ========================================================

    # Power ranges for individual droops
    P1_range = np.linspace(0, P_nom1, 300)
    P2_range = np.linspace(0, P_nom2, 300)

    f1_curve = freq_from_power(P1_range, P_ref1, KPF1, f_nom)
    f2_curve = freq_from_power(P2_range, P_ref2, KPF2, f_nom)

    # No-load and full-load frequencies
    f1_P0   = freq_from_power(0.0,    P_ref1, KPF1, f_nom)
    f1_Pnom = freq_from_power(P_nom1, P_ref1, KPF1, f_nom)
    f2_P0   = freq_from_power(0.0,    P_ref2, KPF2, f_nom)
    f2_Pnom = freq_from_power(P_nom2, P_ref2, KPF2, f_nom)

    # Combined droop: P_total as function of f
    # We sweep frequency from slightly above max(no-loads, f_nom)
    # down to slightly below min(full-loads, f_op)
    f_max = max(f1_P0, f2_P0, f_nom) + 0.2
    f_min_comb = min(f1_Pnom, f2_Pnom, f_op) - 0.2
    f_grid = np.linspace(f_max, f_min_comb, 400)

    # For each frequency, compute power of each generator, then clip
    P1_raw = P_ref1 - KPF1 * two_pi * (f_grid - f_nom)
    P2_raw = P_ref2 - KPF2 * two_pi * (f_grid - f_nom)

    P1_clip = np.clip(P1_raw, 0.0, P_nom1)
    P2_clip = np.clip(P2_raw, 0.0, P_nom2)
    P_sum   = P1_clip + P2_clip

    # Baseline slightly below the lowest frequency of interest
    f_min_global = min(f1_Pnom, f2_Pnom, f_op, f_min_comb) - 1.0
    baseline = f_min_global

    fig, ax = plt.subplots(figsize=(9, 5))

    # Droop curves
    ax.plot(P1_range, f1_curve, color='red',  linewidth=2, label="Generator 1 droop")
    ax.plot(P2_range, f2_curve, color='blue', linewidth=2, label="Generator 2 droop")

    # Combined droop (P_total vs f)
    ax.plot(P_sum, f_grid, linestyle='--', color='gray', linewidth=2,
            label="Combined droop (G1+G2)")

    # Vertical saturation segments
    ax.vlines(P_nom1, baseline, f1_Pnom, colors='red',  linewidth=2)
    ax.vlines(P_nom2, baseline, f2_Pnom, colors='blue', linewidth=2)

    # Reference lines at f_nom and f_op
    ax.axhline(f_nom, color='black', linestyle='--', linewidth=1)
    ax.axhline(f_op,  color='black', linestyle='--', linewidth=1)

    # Vertical dashed lines for scenario 1 and 2 + rated load
    for x in [P2_s1, P1_s1, P1_s2, P_load_nom]:
        ax.vlines(x, baseline, f_nom, colors='black', linestyles='--', linewidth=1)

    # Labels for f_nom and f_op (right side)
    x_right = max(P_nom1 + P_nom2, P_load_nom) * 1.15
    ax.text(x_right, f_nom + 0.02,
            r"$f_{\mathrm{nom}} = %.2f\ \mathrm{Hz}$" % f_nom,
            fontsize=9, ha='right', va='bottom')
    ax.text(x_right, f_op + 0.02,
            r"$f_{\mathrm{op}} = %.2f\ \mathrm{Hz}$" % f_op,
            fontsize=9, ha='right', va='bottom')

    # Markers at key points
    ax.scatter([0.0],   [f1_P0],   color='red')
    ax.scatter([0.0],   [f2_P0],   color='blue')
    ax.scatter([P2_s1], [f_nom],   color='blue')
    ax.scatter([P1_s1], [f_nom],   color='red')
    ax.scatter([P1_s2], [f_op],    color='black')
    ax.scatter([P_nom2],[f2_Pnom], color='blue')
    ax.scatter([P_nom1],[f1_Pnom], color='red')

    # Also mark combined points of interest (optional)
    # At f_nom, total P = P1_s1 + P2_s1 = P_L1
    ax.scatter([P_L1], [f_nom], color='gray')
    # At f_op, total P = P_L2
    ax.scatter([P_L2], [f_op], color='gray')

    # Text labels (can be tuned further if needed)
    ax.text(1.0, f1_P0 + 0.05, f"(0 kW, {f1_P0:.2f} Hz)",
            color='red',  fontsize=8)
    ax.text(1.0, f2_P0 + 0.05, f"(0 kW, {f2_P0:.2f} Hz)",
            color='blue', fontsize=8)

    ax.text(P2_s1 + 3.0, f_nom + 0.05,
            f"({P2_s1:.1f} kW, {f_nom:.1f} Hz)", color='blue', fontsize=8)
    ax.text(P1_s1 + 1.0, f_nom + 0.05,
            f"({P1_s1:.1f} kW, {f_nom:.1f} Hz)", color='red',  fontsize=8)

    ax.text(P1_s2 + 2.0, f_op + 0.10,
            f"({P1_s2:.1f} kW, {f_op:.2f} Hz)", color='black', fontsize=8)

    ax.text(P_nom2 + 1.5, f2_Pnom + 0.10,
            f"({P_nom2:.1f} kW, {f2_Pnom:.2f} Hz)", color='blue', fontsize=8)
    ax.text(P_nom1 + 1.0, f1_Pnom + 0.15,
            f"({P_nom1:.1f} kW, {f1_Pnom:.2f} Hz)", color='red',  fontsize=8)

    # Rated load label
    ax.text(P_load_nom + 1.5, baseline + 0.1,
            f"Rated load {P_load_nom:.1f} kW",
            rotation=90, fontsize=8, ha='left', va='bottom', color='black')

    # Axes formatting
    ax.set_xlim(0, max(P_nom1 + P_nom2, P_load_nom) * 1.2)
    ax.set_ylim(baseline, f_nom + 3.0)
    ax.set_xlabel("Active Power P (kW)")
    ax.set_ylabel("Frequency f (Hz)")
    ax.grid(True, linestyle=':', linewidth=0.7)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("droop_characteristics_interactive_with_combined.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()