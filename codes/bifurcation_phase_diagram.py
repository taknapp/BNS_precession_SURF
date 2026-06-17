"""
Bifurcation phase diagrams for NS/exotic compact object binary spin precession.

Plots the number of roots of (d delta_chi / dt)^2 as a function of:
  1. k vs r       (left panel)
  2. chi_cons vs r (middle panel)
  3. k vs chi_cons (right panel)

Based on the generalized precession equations from Fumagalli et al. (ShortNotes),
extending the BH case (k=1) to arbitrary quadrupole coefficient k.

Parameters
----------
chi_cons : float
    Conserved spin quantity (generalizes chi_eff to non-BH case).
kappa : float
    Asymptotic angular momentum kappa = (J^2 - L^2) / (2L).
q : float
    Mass ratio m2/m1 <= 1.
chi1, chi2 : float
    Dimensionless Kerr spin parameters.
M : float
    Total mass (set to 1 in geometric units).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── fiducial parameters ────────────────────────────────────────────────────────
chi_cons = -0.145
kappa    = -0.020
q        = 1.0
chi1     = 0.5
chi2     = 0.6
M        = 1.0

# ── grid resolution ────────────────────────────────────────────────────────────
N_GRID = 60          # points per axis (increase for smoother diagrams)
N_DCHI = 2000        # points along delta_chi axis for root finding


def count_roots(k1, k2, r, chi_cons, kappa, q, chi1, chi2, M, n=N_DCHI):
    """
    Count the number of sign changes of (d delta_chi / dt)^2 across the
    physical interval delta_chi in [-(chi1+q*chi2)/(1+q), +(chi1+q*chi2)/(1+q)].

    Returns
    -------
    int : number of roots (-1 if the configuration is unphysical)
    """
    dchi_max = (chi1 + q * chi2) / (1 + q)
    dchi_arr = np.linspace(-dchi_max * 0.999, dchi_max * 0.999, n)
    rM = np.sqrt(r / M)

    # ── nested radical R (appears in cos theta_1, cos theta_2) ────────────────
    R_inside = (
        4 * r / M
        + (1 - k1 * k2) * dchi_arr**2
        + (2 + k1 + k2) * chi_cons**2
        - 2 * rM * ((k1 - k2) * dchi_arr + (2 + k1 + k2) * chi_cons)
    )
    if np.all(R_inside < 0):
        return -1
    R = np.where(R_inside >= 0, np.sqrt(np.abs(R_inside)), np.nan)

    # ── cos theta_1, cos theta_2  (Eqs. 36-37 of ShortNotes) ─────────────────
    ct1 = ((1 + q) * (2 * rM + (1 + k2) * dchi_arr) - (1 + q) * R) / ((2 + k1 + k2) * chi1)
    ct2 = ((1 + q) * (2 * rM - (1 + k1) * dchi_arr) - (1 + q) * R) / (q * (2 + k1 + k2) * chi2)

    # ── A^2 (signed, from Eq. 35) ─────────────────────────────────────────────
    inside_A = (
        dchi_arr**2 * M * (1 - k1 * k2)
        - 2 * np.sqrt(M * r) * (dchi_arr * (k1 - k2) + chi_cons * (k1 + k2 + 2))
        + M * chi_cons**2 * (k1 + k2 + 2)
        + 4 * r
    )
    A2_signed = (3 * M**3 * q * chi1 * chi2)**2 * inside_A / (4 * (1 + q)**4 * r**7)

    # ── cos Delta Phi numerator N and denominator D  (Eq. 38) ─────────────────
    # kappa plays the role of k in the cos Delta Phi expression
    N = (
        2 * kappa * q * (q + 1)**2 * np.sqrt(r)
        - M**2 * (
            2 * q * chi1 * ct1 * (q * chi2 * np.sqrt(M) * ct2 + np.sqrt(r))
            + 2 * q**3 * chi2 * np.sqrt(r) * ct2
            + np.sqrt(M) * (q**4 * chi2**2 + chi1**2)
        )
    )
    D = 2 * M**(5 / 2) * q**2 * chi1 * chi2

    # ── regularized product (1-cos^2 theta1)(1-cos^2 theta2)(1-cos^2 DeltaPhi)
    sin2_prod = (1 - ct1**2) * (1 - ct2**2)
    reg = sin2_prod * D**2 - N**2          # D^2 * sin2 * (1 - cos^2 DP)

    rhs = A2_signed * reg / D**2

    # ── count sign changes, ignoring NaNs ─────────────────────────────────────
    valid = ~np.isnan(rhs)
    if np.sum(valid) < 10:
        return -1
    sign_arr = np.where(valid, np.sign(rhs), 0)
    changes = np.where(np.diff(sign_arr) != 0)[0]
    changes = [i for i in changes if valid[i] and valid[i + 1]]
    return len(changes)


# ── parameter grids ────────────────────────────────────────────────────────────
k_vals      = np.linspace(1, 15, N_GRID)
r_vals      = np.logspace(0.5, 3, N_GRID)      # r/M from ~3 to 1000
chicons_vals = np.linspace(-0.5, 0.5, N_GRID)

# ── compute phase diagrams ─────────────────────────────────────────────────────
print("Computing k vs r phase diagram...")
nroots_kr = np.zeros((len(k_vals), len(r_vals)))
for i, k in enumerate(k_vals):
    for j, r in enumerate(r_vals):
        nroots_kr[i, j] = count_roots(k, k, r, chi_cons, kappa, q, chi1, chi2, M)

print("Computing chi_cons vs r phase diagram...")
nroots_cr = np.zeros((len(chicons_vals), len(r_vals)))
for i, cc in enumerate(chicons_vals):
    for j, r in enumerate(r_vals):
        nroots_cr[i, j] = count_roots(10, 10, r, cc, kappa, q, chi1, chi2, M)

print("Computing k vs chi_cons phase diagram...")
nroots_kc = np.zeros((len(k_vals), len(chicons_vals)))
for i, k in enumerate(k_vals):
    for j, cc in enumerate(chicons_vals):
        nroots_kc[i, j] = count_roots(k, k, 20, cc, kappa, q, chi1, chi2, M)

# ── colormap: gray=unphysical, black=0 roots, green=2 roots, orange=4 roots ───
colors = ['#aaaaaa', '#222222', '#2ecc71', '#e67e22']
cmap   = mcolors.ListedColormap(colors)
bounds = [-1.5, -0.5, 0.5, 2.5, 4.5]
norm   = mcolors.BoundaryNorm(bounds, cmap.N)

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# panel 1: k vs r
im1 = axes[0].pcolormesh(r_vals, k_vals, nroots_kr, cmap=cmap, norm=norm)
axes[0].set_xscale('log')
axes[0].set_xlabel('r / M', fontsize=12)
axes[0].set_ylabel('k  (k1 = k2)', fontsize=12)
axes[0].set_title(f'χ_cons = {chi_cons},  κ = {kappa}', fontsize=11)
axes[0].axhline(1,  color='white', lw=1.5, ls='--', alpha=0.8, label='BH (k=1)')
axes[0].axhline(10, color='cyan',  lw=1.5, ls='--', alpha=0.8, label='k=10')
axes[0].legend(fontsize=8)
plt.colorbar(im1, ax=axes[0], label='n roots', ticks=[-1, 0, 2, 4]).set_ticklabels(
    ['unphysical', '0 roots', '2 roots', '4 roots'])

# panel 2: chi_cons vs r
im2 = axes[1].pcolormesh(r_vals, chicons_vals, nroots_cr, cmap=cmap, norm=norm)
axes[1].set_xscale('log')
axes[1].set_xlabel('r / M', fontsize=12)
axes[1].set_ylabel('χ_cons', fontsize=12)
axes[1].set_title('k1 = k2 = 10,  κ = {}'.format(kappa), fontsize=11)
axes[1].axhline(chi_cons, color='cyan',  lw=1.5, ls='--', alpha=0.8, label=f'χ_cons = {chi_cons}')
axes[1].axhline(0,        color='white', lw=1.5, ls='--', alpha=0.8, label='χ_cons = 0')
axes[1].legend(fontsize=8)
plt.colorbar(im2, ax=axes[1], label='n roots', ticks=[-1, 0, 2, 4]).set_ticklabels(
    ['unphysical', '0 roots', '2 roots', '4 roots'])

# panel 3: k vs chi_cons
im3 = axes[2].pcolormesh(chicons_vals, k_vals, nroots_kc, cmap=cmap, norm=norm)
axes[2].set_xlabel('χ_cons', fontsize=12)
axes[2].set_ylabel('k  (k1 = k2)', fontsize=12)
axes[2].set_title('r = 20 M,  κ = {}'.format(kappa), fontsize=11)
axes[2].axvline(chi_cons, color='cyan',  lw=1.5, ls='--', alpha=0.8, label=f'χ_cons = {chi_cons}')
axes[2].axhline(1,        color='white', lw=1.5, ls='--', alpha=0.8, label='BH (k=1)')
axes[2].axhline(10,       color='cyan',  lw=1.5, ls='--', alpha=0.8, label='k=10')
axes[2].legend(fontsize=8)
plt.colorbar(im3, ax=axes[2], label='n roots', ticks=[-1, 0, 2, 4]).set_ticklabels(
    ['unphysical', '0 roots', '2 roots', '4 roots'])

plt.suptitle(
    f'Bifurcation phase diagrams  (q={q}, χ₁={chi1}, χ₂={chi2})\n'
    f'Green = 2 roots (normal precession),  Orange = 4 roots (bifurcated)',
    fontsize=12
)
plt.tight_layout()
plt.savefig('bifurcation_phase_diagrams.png', dpi=150, bbox_inches='tight')
print("Saved bifurcation_phase_diagrams.png")
plt.show()
