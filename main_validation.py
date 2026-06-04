#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 00:40:03 2025

@author: freddiedooley
"""

"""
main_validation.py

Stage 1: Free-particle validation for the quantum tunnelling project - Test case.

This script:
- Builds the spatial and time grids specified in the project brief.
- Constructs the free-particle Hamiltonian.
- Initialises a Gaussian wavepacket with the given parameters.
- Evolves the wavepacket in time using the numerical solver.
- Compares the numerical |psi(x,t)|^2 with the analytical expression
  for selected times.
- Checks conservation of the total probability norm and prints a small
  table of error metrics for selected times.

The main output is Figure 1: numerical vs analytical free-packet evolution "Test Case".
"""

# Parameters

# Physical parameters for the free-particle test
mass = 1.0
hbar = 1.0
k0 = 2.0
a = -10.0
sigma = 2.0

# Numerical parameters for the validation case
x_min = -25.0
x_max = 25.0
num_x = 501

t_min = 0.0
t_max = 10.0
num_t = 1001


# Imports

import numpy as np
import matplotlib.pyplot as plt

from qt_utils import (
    make_spatial_grid,
    make_time_grid,
    V_free,
    build_laplacian,
    build_hamiltonian,
    gaussian_wavepacket,
    evolve_wavefunction,
    compute_norm,
    compute_density,
)


# Analytical free-packet density

def analytical_free_packet_density(x, t, a, sigma, k0, mass=1.0, hbar=1.0):
    """
    Analytical probability density |psi(x,t)|^2 for a free Gaussian wavepacket.

    |psi(x,t)|^2 = 1 / [ sigma * sqrt(2*pi) * sqrt(1 + Delta(t)^2) ]
                   * exp( - (x - a - v t)^2 /
                            [ 2 * sigma^2 * (1 + Delta(t)^2) ] )

    where
        v = hbar * k0 / mass
        Delta(t) = t * hbar / (2 * mass * sigma^2)
    """
    v = hbar * k0 / mass
    Delta_t = t * hbar / (2.0 * mass * sigma**2)

    prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi) * np.sqrt(1.0 + Delta_t**2))
    denom = 2.0 * sigma**2 * (1.0 + Delta_t**2)
    exponent = - (x - a - v * t)**2 / denom

    density = prefactor * np.exp(exponent)
    return density


# Main validation routine

def main():
    # Spatial and time grids
    x, dx = make_spatial_grid(x_min, x_max, num_x)
    t_array, dt = make_time_grid(t_min, t_max, num_t)

    # Free-particle Hamiltonian
    V = V_free(x)
    laplacian = build_laplacian(num_x, dx)
    H = build_hamiltonian(laplacian, V, mass=mass, hbar=hbar)

    # Initial state: Gaussian wavepacket
    psi0 = gaussian_wavepacket(x, x0=a, sigma=sigma, k0=k0)

    # Time evolution
    psi_t = evolve_wavefunction(
        H,
        psi0,
        t_array=t_array,
        hbar=hbar,
        rtol=1e-7,
        atol=1e-9,
        method="RK45",
    )

    # Norm conservation over the whole run
    norms = np.array([compute_norm(psi_t[i, :], dx) for i in range(num_t)])
    print(f"Norm over time: min = {norms.min():.6f}, max = {norms.max():.6f}")

    # Compare numerical and analytical densities at selected times
    times_to_compare = [0.0, 5.0, 10.0]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Store metrics for a small summary table
    metrics_rows = []

    v_group = hbar * k0 / mass  # expected group velocity

    for t_target in times_to_compare:
        idx = np.argmin(np.abs(t_array - t_target))
        t_val = t_array[idx]

        psi_num = psi_t[idx, :]
        density_num = compute_density(psi_num)
        density_analytic = analytical_free_packet_density(
            x, t_val, a, sigma, k0, mass=mass, hbar=hbar
        )

        # Plot numerical and analytical curves
        ax.plot(
            x,
            density_num,
            linestyle="-",
            label=f"Numerical, t = {t_val:.2f}",
        )
        ax.plot(
            x,
            density_analytic,
            linestyle="--",
            label=f"Analytical, t = {t_val:.2f}",
        )

        # Error metrics for this time
        abs_error = np.abs(density_num - density_analytic)
        max_error = abs_error.max()
        l2_error = np.sqrt(np.sum(abs_error**2) * dx)

        # Packet centre check
        mean_x_num = np.sum(x * density_num) * dx
        x_expected = a + v_group * t_val
        centre_error = np.abs(mean_x_num - x_expected)

        norm_here = norms[idx]

        metrics_rows.append(
            (t_val, max_error, l2_error, norm_here, centre_error)
        )

    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(x,t)|^2$")
    ax.legend()
    fig.tight_layout()

    # Print a small metrics table
    print("\nFree-particle validation metrics (numerical vs analytical):")
    header = f"{'t':>6} {'max|Δρ|':>12} {'L2(Δρ)':>12} {'norm':>10} {'|⟨x⟩-x_exp|':>14}"
    print(header)
    print("-" * len(header))
    for t_val, max_err, l2_err, norm_val, centre_err in metrics_rows:
        print(
            f"{t_val:6.2f} {max_err:12.3e} {l2_err:12.3e} "
            f"{norm_val:10.6f} {centre_err:14.3e}"
        )

    # Norm vs time plot
    fig_norm, ax_norm = plt.subplots(figsize=(6, 4))
    ax_norm.plot(t_array, norms)
    ax_norm.set_xlabel("t")
    ax_norm.set_ylabel("Norm")
    fig_norm.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
