#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 00:40:05 2025

@author: freddiedooley
"""

"""
main_energy_scan.py

Stage 4 and Stage 5: Energy (k0) scan for double-barrier tunnelling.

This script:
- Builds a fixed double-barrier potential and Hamiltonian.
- Scans over a range of incident wavenumbers k0.
- For each k0, chooses a time window t_array based on
  t_end(k0) = A + B / k0, then:
    * launches a Gaussian wavepacket from the left,
    * evolves it in time,
    * computes reflection R, transmission T, middle-well probability P_mid,
      total norm, and a quality metric Delta = |R + T + P_mid - Norm|.
- Produces:
    Figure 6: Transmission T(k0) with quality flags.
    Figure 7: Middle-well probability P_mid(k0).
    Figure 8: Zoomed-in transmission T(E) around one resonance
    selected by the largest P_mid peak.
- Prints summary tables for:
    * scan-wide quality metrics,
    * Lorentzian fit parameters around one resonance.
"""

# Parameters (kept consistent with main_double_barrier_runs.py)

# Physical parameters
mass = 1.0
hbar = 1.0

U0 = 4.0        # Barrier height
d = 1.0         # Barrier width
L_well = 4.0    # Well width

# Spatial grid
x_min = -80.0
x_max = 80.0
num_x = 1201    # start with 2401 REDUCE IF RUN TAKES TOO LONG - try 801. Note that if chnaged, resonant peak position may shift slightly

# Packet parameters
packet_sigma = 4.0
packet_center = -20.0

# Time sampling per k0 (Option B)
num_t = 801
A = 10.0   # base time
B = 80.0   # 1/k0 scaling factor

# k0 scan range (chosen to cover first two resonances and over-barrier region)
num_k0 = 80
k_min = 0.5
k_max = 4.0


# Imports

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from qt_utils import (
    make_spatial_grid,
    double_barrier_regions,
    build_laplacian,
    build_hamiltonian,
    gaussian_wavepacket,
    evolve_wavefunction,
    compute_norm,
    compute_R_T_Pmid,
)


# Time-window helper

def choose_t_array_for_k0(k0, num_t, A=10.0, B=80.0):
    """
    Choose a time grid for a given k0 using:

        t_end(k0) = A + B / k0

    where A and B are constants chosen so that t_end is long enough for
    interaction and decay but shorter than boundary-reflection times.
    """
    t_end = A + B / k0
    t_array = np.linspace(0.0, t_end, num_t)
    return t_array


# Single-k0 helper

def run_single_k0_scan(
    k0,
    x,
    dx,
    H,
    left_region_mask,
    middle_region_mask,
    right_region_mask,
    packet_center,
    packet_sigma,
    num_t,
    A,
    B,
    mass=1.0,
    hbar=1.0,
):
    """
    Run a full time evolution for a given k0 with its own time grid
    defined by t_end(k0) = A + B / k0.
    """
    t_array = choose_t_array_for_k0(k0, num_t=num_t, A=A, B=B)

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

    num_t_local = t_array.size
    R_t = np.zeros(num_t_local)
    T_t = np.zeros(num_t_local)
    P_mid_t = np.zeros(num_t_local)
    norm_t = np.zeros(num_t_local)

    for i in range(num_t_local):
        psi = psi_t[i, :]
        R_i, T_i, P_mid_i = compute_R_T_Pmid(
            psi,
            dx,
            left_region_mask,
            middle_region_mask,
            right_region_mask,
        )
        R_t[i] = R_i
        T_t[i] = T_i
        P_mid_t[i] = P_mid_i
        norm_t[i] = compute_norm(psi, dx)

    R_final = R_t[-1]
    T_final = T_t[-1]
    P_mid_final = P_mid_t[-1]
    norm_final = norm_t[-1]
    delta = abs(R_final + T_final + P_mid_final - norm_final)

    return R_final, T_final, P_mid_final, norm_final, delta


# Main routine

def main():
    # Spatial grid
    x, dx = make_spatial_grid(x_min, x_max, num_x)

    # Potential, regions, and Hamiltonian
    V, left_region_mask, middle_region_mask, right_region_mask = double_barrier_regions(
        x, U0, d, L_well
    )
    laplacian = build_laplacian(num_x, dx)
    H = build_hamiltonian(laplacian, V, mass=mass, hbar=hbar)

    # k0 values
    k0_values = np.linspace(k_min, k_max, num_k0)

    R_final_vals = np.zeros(num_k0)
    T_final_vals = np.zeros(num_k0)
    P_mid_final_vals = np.zeros(num_k0)
    norm_final_vals = np.zeros(num_k0)
    delta_vals = np.zeros(num_k0)

    print("Starting energy (k0) scan using t_end(k0) = A + B/k0...")
    for i, k0 in enumerate(k0_values):
        print(f"  [{i+1}/{num_k0}] k0 = {k0:.4f}")

        (
            R_final,
            T_final,
            P_mid_final,
            norm_final,
            delta,
        ) = run_single_k0_scan(
            k0=k0,
            x=x,
            dx=dx,
            H=H,
            left_region_mask=left_region_mask,
            middle_region_mask=middle_region_mask,
            right_region_mask=right_region_mask,
            packet_center=packet_center,
            packet_sigma=packet_sigma,
            num_t=num_t,
            A=A,
            B=B,
            mass=mass,
            hbar=hbar,
        )

        R_final_vals[i] = R_final
        T_final_vals[i] = T_final
        P_mid_final_vals[i] = P_mid_final
        norm_final_vals[i] = norm_final
        delta_vals[i] = delta

    print("Energy scan complete.")
    print(
        f"Final norm range over scan: "
        f"min = {norm_final_vals.min():.6f}, max = {norm_final_vals.max():.6f}"
    )
    print(
        f"Quality metric Delta = |R+T+P_mid - Norm|: "
        f"min = {delta_vals.min():.3e}, max = {delta_vals.max():.3e}"
    )

    # Scan-wide quality summary table

    quality_threshold = 1e-2
    good_mask = delta_vals < quality_threshold
    bad_mask = ~good_mask
    mean_delta = delta_vals.mean()
    num_bad = np.count_nonzero(bad_mask)

    print("\nEnergy-scan quality metrics:")
    header = f"{'Metric':<25} {'Value':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'norm_min':<25} {norm_final_vals.min():12.6f}")
    print(f"{'norm_max':<25} {norm_final_vals.max():12.6f}")
    print(f"{'Δ_min':<25} {delta_vals.min():12.3e}")
    print(f"{'Δ_max':<25} {delta_vals.max():12.3e}")
    print(f"{'Δ_mean':<25} {mean_delta:12.3e}")
    print(f"{'#(Δ >= 1e-2)':<25} {num_bad:12d}")

    # Figure 6: Transmission T(k0) with resonance markers
    # Vertical dashed lines mark P_mid local maxima (resonances)

    fig6, ax6 = plt.subplots(figsize=(7, 5))

    # Main transmission curve
    ax6.plot(
        k0_values[good_mask],
        T_final_vals[good_mask],
        linestyle="-",
        linewidth=1.0,
        marker="o",
        markersize=3,
        markerfacecolor="none",
        markeredgewidth=0.8,
        alpha=0.9,
        label=rf"Good points ($\Delta <$ {quality_threshold:.0e})",
    )   


    if np.any(bad_mask):
        ax6.plot(
            k0_values[bad_mask],
            T_final_vals[bad_mask],
            "x",
            label=rf"Suspect points ($\Delta \geq$ {quality_threshold:.0e})",
        )

    # Find resonance positions from local maxima of P_mid(k0)

    k0_max_res_region = 3.0
    mask_res_region = (k0_values <= k0_max_res_region) & good_mask

    indices = np.where(mask_res_region)[0]
    peak_indices = []

    # Simple local-max detection: P[i] greater than its neighbours
    for j in range(1, len(indices) - 1):
        i = indices[j]
        i_prev = indices[j - 1]
        i_next = indices[j + 1]
        if (
            P_mid_final_vals[i] > P_mid_final_vals[i_prev]
            and P_mid_final_vals[i] > P_mid_final_vals[i_next]
        ):
            peak_indices.append(i)

    # If we found more than four local maxima, keep the four highest peaks
    if len(peak_indices) > 4:
        peak_indices = sorted(
            peak_indices,
            key=lambda i: P_mid_final_vals[i]
        )[-4:]

    # Convert to sorted k0 positions
    k0_res_peaks = np.sort(k0_values[peak_indices]) if peak_indices else []

    # Add vertical resonance markers
    first_marker = True
    for k_res in k0_res_peaks:
        if first_marker:
            ax6.axvline(
                k_res,
                linestyle="--",
                linewidth=0.8,
                color="gray",
                alpha=0.4,
                label=r"P$_{\mathrm{mid}}$ peaks (fig.7)",
            )
            first_marker = False
        else:
            ax6.axvline(
                k_res,
                linestyle="--",
                linewidth=0.8,
                color="gray",
                alpha=0.4,
            )

    if peak_indices:
        i_best = max(peak_indices, key=lambda i: P_mid_final_vals[i])
        k_best = k0_values[i_best]
        T_best = T_final_vals[i_best]

        ax6.plot(k_best, T_best, marker="o", markersize=5, markerfacecolor="none")
        ax6.annotate(
            rf"resonance $k_0 \approx {k_best:.2f}$",
            xy=(k_best, T_best),
            xytext=(k_best - 0.20, T_best + 0.08),
            fontsize=9,
        )

    ax6.set_xlabel(r"Incident wavenumber $k_0$")
    ax6.set_ylabel("Transmission $T(k_0)$")
    ax6.set_xlim(k0_values[0], k0_values[-1])
    ax6.legend(loc="best")

    fig6.tight_layout()





    # Figure 7: P_mid(k0)

    fig7, ax7 = plt.subplots(figsize=(7, 5))
    ax7.plot(k0_values, P_mid_final_vals, "o-", markersize=4, alpha=0.8)
    ax7.set_xlabel(r"Incident wavenumber $k_0$")
    ax7.set_ylabel(r"Middle-well probability $P_{\mathrm{mid}}$")
    ax7.set_xlim(k0_values[0], k0_values[-1])
    ax7.text(
        0.45,
        0.88,
        "Quasi-bound states",
        transform=ax7.transAxes,
        fontsize=10,
        color="gray",
    )
    fig7.tight_layout()

    # Figure 8: zoom on one resonance

    if np.any(good_mask):
        good_indices = np.where(good_mask)[0]

        # Select resonance by largest P_mid among good points
        P_mid_good = P_mid_final_vals[good_mask]
        idx_best_local = np.argmax(P_mid_good)
        idx_best = good_indices[idx_best_local]

        k0_center = k0_values[idx_best]
        T_center = T_final_vals[idx_best]
        P_center = P_mid_final_vals[idx_best]

        print(
            f"\nSelected resonance for zoom (coarse scan): "
            f"k0 ≈ {k0_center:.4f}, T ≈ {T_center:.3f}, P_mid ≈ {P_center:.3f}"
        )

        # Zoom window width based on coarse scan spacing
        if num_k0 > 1:
            dk_coarse = k0_values[1] - k0_values[0]
        else:
            dk_coarse = 0.1

        k_min_zoom = k0_center - 3.0 * dk_coarse
        k_max_zoom = k0_center + 3.0 * dk_coarse
        num_k0_zoom = 80

        k0_zoom_values = np.linspace(k_min_zoom, k_max_zoom, num_k0_zoom)

        T_zoom_vals = np.zeros(num_k0_zoom)
        norm_zoom_vals = np.zeros(num_k0_zoom)
        delta_zoom_vals = np.zeros(num_k0_zoom)

        print("Starting zoomed scan around selected resonance...")
        for j, k0_zoom in enumerate(k0_zoom_values):
            print(f"  [zoom {j+1}/{num_k0_zoom}] k0 = {k0_zoom:.4f}")

            (
                R_zoom,
                T_zoom,
                P_mid_zoom,
                norm_zoom,
                delta_zoom,
            ) = run_single_k0_scan(
                k0=k0_zoom,
                x=x,
                dx=dx,
                H=H,
                left_region_mask=left_region_mask,
                middle_region_mask=middle_region_mask,
                right_region_mask=right_region_mask,
                packet_center=packet_center,
                packet_sigma=packet_sigma,
                num_t=num_t,
                A=A,
                B=B,
                mass=mass,
                hbar=hbar,
            )

            T_zoom_vals[j] = T_zoom
            norm_zoom_vals[j] = norm_zoom
            delta_zoom_vals[j] = delta_zoom

        print("Zoomed scan complete.")
        print(
            f"Zoomed scan norm range: "
            f"min = {norm_zoom_vals.min():.6f}, max = {norm_zoom_vals.max():.6f}"
        )
        print(
            f"Zoomed scan Delta range: "
            f"min = {delta_zoom_vals.min():.3e}, max = {delta_zoom_vals.max():.3e}"
        )

        # Convert k0 to energy
        E_zoom_values = 0.5 * k0_zoom_values**2

        # Lorentzian + constant background fit
       
        def lorentzian_bg(E, A, E0, Gamma, C):
            return C + A / ((E - E0)**2 + (Gamma / 2)**2)

        # Use numerical peak as initial guess for E0
        idx_peak_zoom = np.argmax(T_zoom_vals)
        E_peak = E_zoom_values[idx_peak_zoom]
        T_peak = T_zoom_vals[idx_peak_zoom]

        # Initial parameter guesses
        C_guess = T_zoom_vals.min()
        Gamma_guess = (E_zoom_values[-1] - E_zoom_values[0]) / 10
        A_guess = (T_peak - C_guess) * (Gamma_guess**2)

        p0 = [A_guess, E_peak, Gamma_guess, C_guess]

        try:
            popt, pcov = curve_fit(
                lorentzian_bg,
                E_zoom_values,
                T_zoom_vals,
                p0=p0,
                maxfev=10000,
            )
            A_fit, E0_fit, Gamma_fit, C_fit = popt
            T_fit_vals = lorentzian_bg(E_zoom_values, *popt)

            # 1-sigma uncertainties from covariance matrix
            perr = np.sqrt(np.diag(pcov))
            A_err, E0_err, Gamma_err, C_err = perr

            tau_fit = 1.0 / Gamma_fit
            tau_err = Gamma_err / (Gamma_fit**2)

            print("\nLorentzian fit succeeded.")
            print("Lorentzian fit parameters (with 1σ uncertainties):")
            header_fit = f"{'Parameter':<15} {'Value':>14} {'Uncertainty':>14}"
            print(header_fit)
            print("-" * len(header_fit))
            print(f"{'E0':<15} {E0_fit:14.6f} {E0_err:14.2e}")
            print(f"{'Gamma':<15} {Gamma_fit:14.6f} {Gamma_err:14.2e}")
            print(f"{'tau=1/Gamma':<15} {tau_fit:14.6f} {tau_err:14.2e}")

            # Plot data + fit
            fig8, ax8 = plt.subplots(figsize=(7, 5))
            ax8.plot(E_zoom_values, T_zoom_vals, "o", markersize=4, alpha=0.8, color="C0", label="Numerical T(E)")
            ax8.plot(E_zoom_values, T_fit_vals, "-", color="C1", label="Lorentzian fit")

            ax8.axvline(E0_fit, color="gray", linestyle="--", alpha=0.7,
                        label=f"$E_0 \\approx {E0_fit:.4f}$")

            ax8.set_xlabel("Energy E")
            ax8.set_ylabel("Transmission T(E)")
            ax8.legend()
            fig8.tight_layout()

        except RuntimeError:
            print("Lorentzian fit failed; plotting numerical data only.")

            fig8, ax8 = plt.subplots(figsize=(7, 5))
            ax8.plot(E_zoom_values, T_zoom_vals, "-", color="C0")
            ax8.axvline(E_peak, color="gray", linestyle="--",
                        label=f"Peak at E ≈ {E_peak:.4f}")
            ax8.plot(E_peak, T_peak, "o", color="C0")

            ax8.set_xlabel("Energy E")
            ax8.set_ylabel("Transmission T(E)")
            ax8.legend()
            fig8.tight_layout()

    else:
        print("No 'good' points found; skipping zoomed resonance analysis.")

    plt.show()


if __name__ == "__main__":
    main()
