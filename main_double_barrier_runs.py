#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 00:40:04 2025

@author: freddiedooley
"""

"""
main_double_barrier_runs.py

Stages 2 and 3: Double-barrier geometry and time-domain dynamics.

This script:
- Defines a double-barrier potential and the associated regions used
  to measure reflection, middle-well probability, and transmission.
- Runs a time-domain simulation for a resonant incident Gaussian
  wavepacket (k0 chosen from the energy scan).
- Runs a time-domain simulation for a non-resonant wavepacket.
- Produces:
    Figure 2: Potential profile with regions (geometry).
    Figure 3: |psi(x,t)|^2 snapshots for the resonant case.
    Figure 4: R(t), T(t), P_mid(t), and Norm(t) for the resonant case.
    Figure 5: Comparison of resonant vs non-resonant probability flow.
"""

# Parameters (kept consistent with main_energy_scan.py)

# Physical parameters
mass = 1.0
hbar = 1.0

U0 = 4.0        # Barrier height
d = 1.0         # Barrier width
L_well = 4.0    # Central well width

# Spatial grid
x_min = -80.0
x_max = 80.0
num_x = 2401

# Time grid for time-domain runs
t_min = 0.0
t_max = 40.0
num_t = 1001

# Initial packet parameters
packet_sigma = 4.0
packet_center = -20.0  # Start well to the left

# Chosen incident wavenumbers
k0_resonant = 2.6   # from energy scan: second resonance
k0_nonres = 1.6    # far from resonances for contrast


# Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qt_utils import (
    make_spatial_grid,
    make_time_grid,
    double_barrier_regions,
    build_laplacian,
    build_hamiltonian,
    gaussian_wavepacket,
    evolve_wavefunction,
    compute_density,
    compute_norm,
    compute_R_T_Pmid,
)


# Helper function

def run_time_domain_simulation(
    x,
    dx,
    t_array,
    V,
    left_region_mask,
    middle_region_mask,
    right_region_mask,
    k0,
    packet_center,
    packet_sigma,
    mass=1.0,
    hbar=1.0,
):
    """
    Run a time-domain simulation and compute R(t), T(t), P_mid(t), and norm(t).
    """
    num_x = x.size
    num_t = t_array.size

    laplacian = build_laplacian(num_x, dx)
    H = build_hamiltonian(laplacian, V, mass=mass, hbar=hbar)

    psi0 = gaussian_wavepacket(x, x0=packet_center, sigma=packet_sigma, k0=k0)

    psi_t = evolve_wavefunction(
        H,
        psi0,
        t_array=t_array,
        hbar=hbar,
        rtol=1e-7,
        atol=1e-9,
        method="RK45",
    )

    R_t = np.zeros(num_t)
    T_t = np.zeros(num_t)
    P_mid_t = np.zeros(num_t)
    norm_t = np.zeros(num_t)

    for i in range(num_t):
        psi = psi_t[i, :]
        R_i, T_i, P_mid_i = compute_R_T_Pmid(
            psi, dx, left_region_mask, middle_region_mask, right_region_mask
        )
        R_t[i] = R_i
        T_t[i] = T_i
        P_mid_t[i] = P_mid_i
        norm_t[i] = compute_norm(psi, dx)

    return psi_t, R_t, T_t, P_mid_t, norm_t


# Main routine

def main():
    # Grids
    x, dx = make_spatial_grid(x_min, x_max, num_x)
    t_array, dt = make_time_grid(t_min, t_max, num_t)

    # Double-barrier potential and regions (Figure 2)
    V, left_region_mask, middle_region_mask, right_region_mask = double_barrier_regions(
        x, U0, d, L_well
    )

    # Figure 2: geometry (zoomed region)

    fig2, ax2 = plt.subplots(figsize=(7, 4))

    # Plot only near the barrier region for clarity
    x_zoom_min = -10.0
    x_zoom_max = 10.0

    zoom_mask = (x >= x_zoom_min) & (x <= x_zoom_max)

    ax2.plot(x[zoom_mask], V[zoom_mask], label="V(x)")

    left_indices = np.where(left_region_mask & zoom_mask)[0]
    right_indices = np.where(right_region_mask & zoom_mask)[0]
    mid_indices = np.where(middle_region_mask & zoom_mask)[0]

    # Left boundary
    if left_indices.size > 0:
        x_left_boundary = x[left_indices[-1]]
        ax2.axvline(x_left_boundary, linestyle="--", color="C1",
                    label="Left region boundary")

    # Right boundary
    if right_indices.size > 0:
        x_right_boundary = x[right_indices[0]]
        ax2.axvline(x_right_boundary, linestyle="--", color="C2",
                    label="Right region boundary")

    # Middle-well shading
    if mid_indices.size > 0:
        ax2.axvspan(
            x[mid_indices[0]],
            x[mid_indices[-1]],
            alpha=0.15,
            color="C3",
            label="Middle (well) region",
        )

    ax2.set_xlim(x_zoom_min, x_zoom_max)
    ax2.set_xlabel("x")
    ax2.set_ylabel("V(x)")

    ax2.text(
        0.02,
        0.92,
        f"Full simulation domain: [{x_min}, {x_max}]",
        transform=ax2.transAxes,
        fontsize=9,
        color="gray",
    )

    # Barrier and well edges (from construction in qt_utils)
    total_width = 2.0 * d + L_well
    left_barrier_start = -total_width / 2.0
    left_barrier_end = left_barrier_start + d
    right_barrier_end = total_width / 2.0
    right_barrier_start = right_barrier_end - d

    # Label barrier height U0 near the top of a barrier
    ax2.text(
        left_barrier_start + 0.2,
        U0 * 0.92,
        r"$U_0$",
        fontsize=10,
    )

    # Label well width L_well with a double arrow between inner faces
    y_arrow = 0.25 * U0  # place arrow below barrier top for readability
    ax2.annotate(
        "",
        xy=(right_barrier_start, y_arrow),
        xytext=(left_barrier_end, y_arrow),
        arrowprops=dict(arrowstyle="<->", linewidth=1.0),
    )
    ax2.text(
        0.5 * (left_barrier_end + right_barrier_start),
        y_arrow + 0.05 * U0,
        r"$L_{\mathrm{well}}$",
        ha="center",
        va="bottom",
        fontsize=10)
    ax2.legend(loc="upper right")
    fig2.tight_layout()

    # Resonant run (Figures 3 and 4)

    print(f"Running resonant case with k0 ≈ {k0_resonant:.3f}")

    psi_t_res, R_t_res, T_t_res, P_mid_t_res, norm_t_res = run_time_domain_simulation(
        x=x,
        dx=dx,
        t_array=t_array,
        V=V,
        left_region_mask=left_region_mask,
        middle_region_mask=middle_region_mask,
        right_region_mask=right_region_mask,
        k0=k0_resonant,
        packet_center=packet_center,
        packet_sigma=packet_sigma,
        mass=mass,
        hbar=hbar,
    )

    print(
        f"Resonant case norm: min = {norm_t_res.min():.6f}, "
        f"max = {norm_t_res.max():.6f}"
    )

    # Find the time where P_mid(t) is maximal (quasi-bound peak)
    idx_peak_res = np.argmax(P_mid_t_res)
    t_peak_res = t_array[idx_peak_res]
    P_peak_res = P_mid_t_res[idx_peak_res]

    # Figure 3: |psi(x,t)|^2 snapshots

    times_for_snapshots = [0.0, 9.0, 14.0, 23.0]

    mid_indices = np.where(middle_region_mask)[0]
    x_well_min = x[mid_indices[0]]
    x_well_max = x[mid_indices[-1]]

    fig3, axes3 = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    axes3 = axes3.ravel()

    for ax, t_target in zip(axes3, times_for_snapshots):
        idx = np.argmin(np.abs(t_array - t_target))
        density = compute_density(psi_t_res[idx, :])
        t_val = t_array[idx]

        ax.plot(x, density, color="C0")

        # Shade well region
        ax.axvspan(x_well_min, x_well_max, alpha=0.08, color="C3")

        # Optional light outlines of the well edges
        ax.axvline(x_well_min, color="gray", linewidth=0.4)
        ax.axvline(x_well_max, color="gray", linewidth=0.4)
        ax.text(
            0.98, 0.95,
            f"t = {t_val:.1f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, color="black"
            )
        ax.set_xlim(-50, 50)

    fig3.supxlabel("x")
    fig3.supylabel(r"$|\psi(x,t)|^2$")
    fig3.tight_layout(rect=[0, 0, 1, 0.93])

    # Figure 4: R(t), T(t), P_mid(t), Norm(t)

    fig4, ax4 = plt.subplots(figsize=(7, 5))

    ax4.plot(t_array, R_t_res, label="R(t) (left region)")
    ax4.plot(t_array, T_t_res, label="T(t) (right region)")
    ax4.plot(t_array, P_mid_t_res, label=r"$P_{\mathrm{mid}}(t)$ (well)")

    ax4.plot(
        t_array,
        norm_t_res,
        linestyle="--",
        linewidth=1.0,
        label="Norm(t)",
    )

    ax4.axvline(
        t_peak_res,
        linestyle="--",
        linewidth=0.8,
        color="gray",
        label=r"Peak $P_{\mathrm{mid}}$",
    )

    ax4.set_xlabel("t")
    ax4.set_ylabel("Probability")

    ax4.set_xlim(0, 45)
    ax4.set_ylim(0.0, 1.05)

    ax4.legend()
    fig4.tight_layout()

    # Exponential decay fit to P_mid(t) (resonant case)

    def exp_decay(t, A, tau, B):
        return A * np.exp(-t / tau) + B

    # Use times after the peak for the fit
    mask_decay = t_array > t_peak_res
    t_decay = t_array[mask_decay]
    P_decay = P_mid_t_res[mask_decay]

    tau_fit_res = np.nan  # default if fit fails

    try:
        p0 = [P_peak_res, 10.0, 0.0]  # initial guesses
        popt_exp, _ = curve_fit(exp_decay, t_decay, P_decay, p0=p0, maxfev=10000)
        A_fit_res, tau_fit_res, B_fit_res = popt_exp
        print(f"Exponential fit to P_mid(t) (resonant): tau ≈ {tau_fit_res:.3f}")
    except RuntimeError:
        print("Exponential fit to P_mid(t) (resonant) failed; tau not estimated.")

    # Non-resonant run (Figure 5)

    print(f"Running non-resonant case with k0 = {k0_nonres:.3f}")

    psi_t_non, R_t_non, T_t_non, P_mid_t_non, norm_t_non = run_time_domain_simulation(
        x=x,
        dx=dx,
        t_array=t_array,
        V=V,
        left_region_mask=left_region_mask,
        middle_region_mask=middle_region_mask,
        right_region_mask=right_region_mask,
        k0=k0_nonres,
        packet_center=packet_center,
        packet_sigma=packet_sigma,
        mass=mass,
        hbar=hbar,
    )

    print(
        f"Non-resonant case norm: min = {norm_t_non.min():.6f}, "
        f"max = {norm_t_non.max():.6f}"
    )

    # Peak P_mid for non-resonant case
    idx_peak_non = np.argmax(P_mid_t_non)
    t_peak_non = t_array[idx_peak_non]
    P_peak_non = P_mid_t_non[idx_peak_non]

    # Figure 5: comparison

    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

    # Top panel: P_mid(t)
    ax5a.plot(t_array, P_mid_t_res, color="C0", label=r"$P_{\mathrm{mid}}(t)$ resonant")
    ax5a.plot(
        t_array,
        P_mid_t_non,
        color="C1",
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
        label=r"$P_{\mathrm{mid}}(t)$ non-resonant",
    )


    ax5a.set_ylabel(r"$P_{\mathrm{mid}}(t)$")

    ax5a.axvline(t_peak_res, color="gray", linestyle="--", alpha=0.5)

    ax5a.set_ylim(0, 1.15 * max(P_mid_t_res.max(), P_mid_t_non.max()))
    ax5a.legend(loc="upper right")

    # Bottom panel: T(t)
    ax5b.plot(t_array, T_t_res, color="C0", label="T(t) resonant")
    ax5b.plot(
        t_array,
        T_t_non,
        color="C1",
        linestyle="--",
        linewidth=2.0,
        alpha=0.95,
        label="T(t) non-resonant",
    )

    
    ax5b.set_xlabel("t")
    ax5b.set_ylabel("T(t)")
    ax5b.set_ylim(0, 1.15 * max(T_t_res.max(), T_t_non.max()))
    ax5b.legend(loc="lower right")

    ax5b.set_xlim(t_array[0], t_array[-1])

    fig5.tight_layout(rect=[0, 0, 1, 0.97])

    # Summary table: resonant vs non-resonant

    R_inf_res = R_t_res[-1]
    T_inf_res = T_t_res[-1]
    P_inf_res = P_mid_t_res[-1]
    delta_inf_res = abs(R_inf_res + T_inf_res + P_inf_res - norm_t_res[-1])

    R_inf_non = R_t_non[-1]
    T_inf_non = T_t_non[-1]
    P_inf_non = P_mid_t_non[-1]
    delta_inf_non = abs(R_inf_non + T_inf_non + P_inf_non - norm_t_non[-1])

    print("\nSummary of time-domain runs (final values and peaks):")
    header = (
        f"{'Case':>10} {'R_final':>10} {'T_final':>10} {'P_mid_final':>12} "
        f"{'P_mid_max':>10} {'t_peak':>10} {'Δ_final':>10} {'tau_Pmid':>10}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'Resonant':>10} "
        f"{R_inf_res:10.3f} {T_inf_res:10.3f} {P_inf_res:12.3f} "
        f"{P_peak_res:10.3f} {t_peak_res:10.3f} {delta_inf_res:10.2e} "
        f"{tau_fit_res:10.3f}"
    )
    print(
        f"{'Non-res':>10} "
        f"{R_inf_non:10.3f} {T_inf_non:10.3f} {P_inf_non:12.3f} "
        f"{P_peak_non:10.3f} {t_peak_non:10.3f} {delta_inf_non:10.2e} "
        f"{'-':>10}"
    )

    plt.show()


if __name__ == "__main__":
    main()
