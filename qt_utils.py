#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 00:40:01 2025

@author: freddiedooley
"""

"""
qt_utils.py

Utility functions for 1D quantum tunnelling simulations.

This file contains:
- Grid creation (spatial and temporal).
- Potential definitions (free particle and double barrier) and region masks.
- Operators (finite-difference Laplacian and Hamiltonian).
- Initial state construction (Gaussian wavepacket).
- Time evolution of the Schrödinger equation using solve_ivp.
- Observable calculations (norm, region probabilities, reflection/transmission/middle-well).
"""

# Imports

import numpy as np
from scipy.sparse import spdiags, diags
from scipy.integrate import solve_ivp


# Grid utilities

def make_spatial_grid(x_min, x_max, num_points):
    """
    Create a 1D spatial grid.

    Parameters
    x_min : float
        Minimum x value.
    x_max : float
        Maximum x value.
    num_points : int
        Number of spatial grid points.

    Returns
    x : ndarray
        Array of spatial grid points of length num_points.
    dx : float
        Grid spacing.
    """
    x = np.linspace(x_min, x_max, num_points)
    dx = x[1] - x[0]
    return x, dx


def make_time_grid(t_min, t_max, num_points):
    """
    Create a 1D time grid.

    Parameters
    t_min : float
        Initial time.
    t_max : float
        Final time.
    num_points : int
        Number of time points.

    Returns
    t : ndarray
        Array of time points of length num_points.
    dt : float
        Approximate time step size.
    """
    t = np.linspace(t_min, t_max, num_points)
    dt = t[1] - t[0]
    return t, dt


# Potentials and regions

def V_free(x):
    """
    Free particle potential V(x) = 0 everywhere.

    Parameters
    x : ndarray
        Spatial grid.

    Returns
    V : ndarray
        Potential array of the same shape as x (all zeros).
    """
    return np.zeros_like(x)


def V_double_barrier(x, U0, d, L_well):
    """
    Double rectangular barrier potential with a central well.

    The structure is centred at x = 0, with:
    - Left barrier of width d.
    - Central well of width L_well.
    - Right barrier of width d.

    Parameters
    x : ndarray
        Spatial grid.
    U0 : float
        Barrier height.
    d : float
        Barrier width.
    L_well : float
        Width of the central well.

    Returns
    V : ndarray
        Potential array on the grid.
    """
    V = np.zeros_like(x, dtype=float)

    total_width = 2.0 * d + L_well
    left_barrier_start = -total_width / 2.0
    left_barrier_end = left_barrier_start + d
    right_barrier_end = total_width / 2.0
    right_barrier_start = right_barrier_end - d

    left_mask = (x >= left_barrier_start) & (x <= left_barrier_end)
    right_mask = (x >= right_barrier_start) & (x <= right_barrier_end)

    V[left_mask] = U0
    V[right_mask] = U0

    return V


def double_barrier_regions(x, U0, d, L_well):
    """
    Construct double-barrier potential and region masks for reflection,
    middle-well probability, and transmission.

    The regions are defined as:
    - Left region: x < left_barrier_start
    - Middle region (well): between the inner faces of the barriers
      (between left_barrier_end and right_barrier_start)
    - Right region: x > right_barrier_end

    Parameters
    x : ndarray
        Spatial grid.
    U0 : float
        Barrier height.
    d : float
        Barrier width.
    L_well : float
        Width of the central well.

    Returns
    V : ndarray
        Potential array V(x) for the double barrier.
    left_region_mask : ndarray of bool
        Mask for the reflection region.
    middle_region_mask : ndarray of bool
        Mask for the central well region.
    right_region_mask : ndarray of bool
        Mask for the transmission region.
    """
    V = V_double_barrier(x, U0, d, L_well)

    total_width = 2.0 * d + L_well
    left_barrier_start = -total_width / 2.0
    left_barrier_end = left_barrier_start + d
    right_barrier_end = total_width / 2.0
    right_barrier_start = right_barrier_end - d

    left_region_mask = x < left_barrier_start
    right_region_mask = x > right_barrier_end
    middle_region_mask = (x >= left_barrier_end) & (x <= right_barrier_start)

    return V, left_region_mask, middle_region_mask, right_region_mask


# Operators and initial states

def build_laplacian(num_points, dx):
    """
    Build the 1D finite-difference Laplacian as a sparse tridiagonal matrix.

    Uses the central difference approximation:
    d^2 psi / dx^2 ~ (psi_{n-1} - 2 psi_n + psi_{n+1}) / dx^2.

    Parameters
    num_points : int
        Number of spatial grid points.
    dx : float
        Grid spacing.

    Returns
    L : scipy.sparse.spmatrix
        Laplacian operator.
    """
    main_diag = -2.0 * np.ones(num_points)
    off_diag = np.ones(num_points - 1)

    # Prepare diagonals in a (3, num_points) array for spdiags
    data = np.zeros((3, num_points))
    # Subdiagonal (offset -1): entries 1..N-1
    data[0, 1:] = off_diag
    # Main diagonal (offset 0): entries 0..N-1
    data[1, :] = main_diag
    # Superdiagonal (offset +1): entries 0..N-2
    data[2, :-1] = off_diag

    offsets = np.array([-1, 0, 1])

    L = spdiags(data, offsets, num_points, num_points, format="csr") / (dx ** 2)
    return L


def build_hamiltonian(laplacian, V, mass=1.0, hbar=1.0):
    """
    Build the Hamiltonian H = - (hbar^2 / 2m) * Laplacian + V(x).

    Parameters
    laplacian : scipy.sparse.spmatrix
        Laplacian operator on the grid.
    V : ndarray
        Potential array V(x) on the same grid.
    mass : float, optional
        Particle mass (default is 1.0).
    hbar : float, optional
        Reduced Planck constant (default is 1.0).

    Returns
    H : scipy.sparse.spmatrix
        Hamiltonian operator.
    """
    kinetic_prefactor = - (hbar ** 2) / (2.0 * mass)
    kinetic = kinetic_prefactor * laplacian

    # Potential as a diagonal sparse matrix
    potential = diags(V, offsets=0, format="csr")

    H = kinetic + potential
    return H


def gaussian_wavepacket(x, x0, sigma, k0):
    """
    Construct a normalised Gaussian wavepacket.

    psi(x, 0) = (1 / (2*pi)^(1/4) / sqrt(sigma)) * exp(-(x-x0)^2/(4 sigma^2)) * exp(i k0 x)

    Parameters
    x : ndarray
        Spatial grid.
    x0 : float
        Centre position of the wavepacket.
    sigma : float
        Width of the wavepacket.
    k0 : float
        Central wavenumber.

    Returns
    psi0 : ndarray of complex
        Initial wavefunction psi(x, 0) on the grid.
    """
    norm_factor = 1.0 / ((2.0 * np.pi) ** 0.25 * np.sqrt(sigma))
    gaussian_envelope = np.exp(- (x - x0) ** 2 / (4.0 * sigma ** 2))
    plane_wave_factor = np.exp(1j * k0 * x)

    psi0 = norm_factor * gaussian_envelope * plane_wave_factor
    return psi0


# Time evolution

def _schrodinger_rhs(t, psi_flat, H, hbar):
    """
    Right-hand side of the time-dependent Schrödinger equation.

    dpsi/dt = -i / hbar * H psi

    The solver works with a real-valued vector psi_flat that concatenates
    the real and imaginary parts of psi.

    Parameters
    t : float
        Time (not used explicitly for a time-independent Hamiltonian).
    psi_flat : ndarray
        Flattened state vector (real parts followed by imaginary parts).
    H : scipy.sparse.spmatrix
        Hamiltonian operator.
    hbar : float
        Reduced Planck constant.

    Returns
    dpsi_flat_dt : ndarray
        Time derivative of psi_flat, in the same flattened format.
    """
    num = psi_flat.size // 2

    # Reconstruct complex wavefunction
    psi_real = psi_flat[:num]
    psi_imag = psi_flat[num:]
    psi = psi_real + 1j * psi_imag

    # Apply Hamiltonian
    H_psi = H.dot(psi)

    # Time derivative from Schrödinger equation
    dpsi_dt = -1j * H_psi / hbar

    # Return as concatenated real and imaginary parts
    dpsi_real = np.real(dpsi_dt)
    dpsi_imag = np.imag(dpsi_dt)
    dpsi_flat_dt = np.concatenate((dpsi_real, dpsi_imag))

    return dpsi_flat_dt


def evolve_wavefunction(H, psi0, t_array, hbar=1.0, rtol=1e-7, atol=1e-9, method="RK45"):
    """
    Evolve a wavefunction in time under a time-independent Hamiltonian H.

    Parameters
    H : scipy.sparse.spmatrix
        Hamiltonian operator.
    psi0 : ndarray of complex
        Initial wavefunction psi(x, 0) on the spatial grid.
    t_array : ndarray
        Array of time points at which to store the solution.
    hbar : float, optional
        Reduced Planck constant (default is 1.0).
    rtol : float, optional
        Relative tolerance for solve_ivp (default is 1e-7).
    atol : float, optional
        Absolute tolerance for solve_ivp (default is 1e-9).
    method : str, optional
        Integration method for solve_ivp (default is "RK45").

    Returns
    psi_t : ndarray of complex
        Array of shape (len(t_array), N) containing psi(x, t) at the
        requested times, where N is the number of spatial grid points.
    """
    # Flatten initial state into real and imaginary parts
    psi0_real = np.real(psi0)
    psi0_imag = np.imag(psi0)
    psi0_flat = np.concatenate((psi0_real, psi0_imag))

    t_span = (t_array[0], t_array[-1])

    sol = solve_ivp(
        fun=lambda t, y: _schrodinger_rhs(t, y, H, hbar),
        t_span=t_span,
        y0=psi0_flat,
        t_eval=t_array,
        rtol=rtol,
        atol=atol,
        method=method,
    )

    if not sol.success:
        raise RuntimeError(f"Time evolution failed: {sol.message}")

    # Reconstruct complex solution
    num = psi0.size
    psi_flat_t = sol.y  # shape (2N, Nt)
    psi_real_t = psi_flat_t[:num, :]
    psi_imag_t = psi_flat_t[num:, :]
    psi_t = psi_real_t + 1j * psi_imag_t

    # Transpose to shape (Nt, N)
    psi_t = psi_t.T

    return psi_t


# Observables

def compute_density(psi):
    """
    Compute the probability density |psi|^2.

    Parameters
    psi : ndarray of complex
        Wavefunction on the grid.

    Returns
    density : ndarray of float
        Probability density on the grid.
    """
    return np.abs(psi) ** 2


def compute_norm(psi, dx):
    """
    Compute the total probability norm of psi(x).

    Parameters
    psi : ndarray of complex
        Wavefunction on the grid.
    dx : float
        Spatial step size.

    Returns
    norm : float
        Integral of |psi|^2 over all x.
    """
    density = compute_density(psi)
    norm = np.sum(density) * dx
    return norm


def compute_region_probability(psi, dx, region_mask):
    """
    Compute the probability in a specified region.

    Parameters
    psi : ndarray of complex
        Wavefunction on the grid.
    dx : float
        Spatial step size.
    region_mask : ndarray of bool
        Boolean mask selecting the region of interest.

    Returns
    probability : float
        Integral of |psi|^2 over the region.
    """
    density = compute_density(psi)
    probability = np.sum(density[region_mask]) * dx
    return probability


def compute_R_T_Pmid(psi, dx, left_region_mask, middle_region_mask, right_region_mask):
    """
    Compute reflection (R), transmission (T), and middle-well probability (P_mid).

    Parameters
    psi : ndarray of complex
        Wavefunction on the grid at a given time.
    dx : float
        Spatial step size.
    left_region_mask : ndarray of bool
        Mask for the reflection (left) region.
    middle_region_mask : ndarray of bool
        Mask for the central well region.
    right_region_mask : ndarray of bool
        Mask for the transmission (right) region.

    Returns
    R : float
        Reflection probability (left region).
    T : float
        Transmission probability (right region).
    P_mid : float
        Probability in the central well region.
    """
    R = compute_region_probability(psi, dx, left_region_mask)
    T = compute_region_probability(psi, dx, right_region_mask)
    P_mid = compute_region_probability(psi, dx, middle_region_mask)
    return R, T, P_mid
