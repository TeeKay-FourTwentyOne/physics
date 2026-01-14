#
# CylindricalBoundary.mpl
#
# Surface Current Calculations at Cylindrical Boundaries
#
# Computes surface currents K^mu required at a cylindrical boundary rho = R
# to sustain the electromagnetic field.
#
# At a boundary, the jump in the tangential magnetic field requires a surface current:
#     K^mu = (1/4*Pi) n_alpha [F^alpha_mu]  (jump condition)
#
# where n_alpha is the outward normal and [F^alpha_mu] is the discontinuity.
#
# For a field that exists only inside rho < R (with vacuum outside), the required
# surface current is:
#     K^mu = -(1/4*Pi) n_alpha F^alpha_mu|_{rho=R}
#
# Reference: Misra, M. & Radhakrishna, L. (1962).
#

CylindricalBoundary := module()
    option package;

    export Create, InverseMetricAtBoundary, FUpperAtBoundary,
           NormalCovariant, SurfaceCurrentK, SurfaceCurrentPhysical,
           TotalCurrentZ, TotalCurrentPhi, MagneticFieldAtBoundary;

    local rho, t;

    rho := 'rho';
    t := 't';

    #
    # Create: Initialize boundary calculator
    #
    # Parameters:
    #   solution: A CaseI, CaseII, or CaseIII solution record
    #   solution_module: The solution module
    #   radius: Boundary radius R
    #
    # Returns:
    #   Record containing solution, module, and radius
    #
    Create := proc(solution, solution_module, radius)
        return Record(
            'solution' = solution,
            'sol_module' = solution_module,
            'radius' = radius
        );
    end proc;

    #
    # InverseMetricAtBoundary: Compute inverse metric g^mu_nu at the boundary
    #
    InverseMetricAtBoundary := proc(bd, t_val)
        local R, lam_val, mu_val, exp_mu_minus_lam, exp_mu, exp_minus_mu;

        R := bd:-radius;
        lam_val := bd:-sol_module:-Lambda(bd:-solution, R, t_val);
        mu_val := bd:-sol_module:-Mu(bd:-solution, R, t_val);

        exp_mu_minus_lam := evalf(exp(mu_val - lam_val));
        exp_mu := evalf(exp(mu_val));
        exp_minus_mu := evalf(exp(-mu_val));

        return Matrix(4, 4, [
            [-exp_mu_minus_lam, 0, 0, 0],
            [0, -exp_mu / R^2, 0, 0],
            [0, 0, -exp_minus_mu, 0],
            [0, 0, 0, exp_mu_minus_lam]
        ]);
    end proc;

    #
    # FUpperAtBoundary: Compute F^mu_nu at the boundary
    #
    FUpperAtBoundary := proc(bd, t_val)
        local R, F_lower, g_inv, F_upper, mu_idx, nu_idx;

        R := bd:-radius;
        F_lower := bd:-sol_module:-FieldTensorAt(bd:-solution, R, t_val);
        g_inv := InverseMetricAtBoundary(bd, t_val);

        F_upper := Matrix(4, 4, 0);

        for mu_idx from 1 to 4 do
            for nu_idx from 1 to 4 do
                F_upper[mu_idx, nu_idx] := g_inv[mu_idx, mu_idx] * g_inv[nu_idx, nu_idx] * F_lower[mu_idx, nu_idx];
            end do;
        end do;

        return evalf(F_upper);
    end proc;

    #
    # NormalCovariant: Compute the outward-pointing unit normal n_alpha (covariant)
    #
    # For a surface at constant rho, the normal is in the rho direction.
    # The covariant normal is n_alpha = (n_rho, 0, 0, 0) where:
    #     n_rho = sqrt|g_rho_rho| = sqrt(e^(lambda-mu)) for proper normalization
    #
    NormalCovariant := proc(bd, t_val)
        local R, lam_val, mu_val, n_rho;

        R := bd:-radius;
        lam_val := bd:-sol_module:-Lambda(bd:-solution, R, t_val);
        mu_val := bd:-sol_module:-Mu(bd:-solution, R, t_val);

        # g_rho_rho = -e^(lambda-mu), so |g_rho_rho| = e^(lambda-mu)
        n_rho := evalf(sqrt(exp(lam_val - mu_val)));

        return Vector([n_rho, 0, 0, 0]);
    end proc;

    #
    # SurfaceCurrentK: Compute surface current density K^mu at the boundary
    #
    # For a field existing only inside rho < R with vacuum outside:
    #     K^mu = -(1/4*Pi) n_alpha F^alpha_mu|_{rho=R}
    #
    # Returns: K^mu = (K^rho, K^Phi, K^z, K^t)
    #     Note: K^rho should be ~0 (no current normal to surface)
    #
    SurfaceCurrentK := proc(bd, t_val)
        local F_upper, n, K, mu_idx;

        F_upper := FUpperAtBoundary(bd, t_val);
        n := NormalCovariant(bd, t_val);

        # K^mu = -(1/4*Pi) n_alpha F^alpha_mu = -(1/4*Pi) n_rho F^rho_mu
        K := Vector(4, 0);

        for mu_idx from 1 to 4 do
            K[mu_idx] := evalf(-(1 / (4 * Pi)) * n[1] * F_upper[1, mu_idx]);
        end do;

        return K;
    end proc;

    #
    # SurfaceCurrentPhysical: Return surface current components with physical interpretation
    #
    # Returns table with:
    #     K_z: Axial current density (current per unit azimuthal length)
    #     K_Phi: Azimuthal current density (current per unit axial length)
    #     K_t: Charge density on surface
    #
    SurfaceCurrentPhysical := proc(bd, t_val)
        local K, result;

        K := SurfaceCurrentK(bd, t_val);

        result := table();
        result["K_rho"] := K[1];  # Should be ~0 (tangent to surface)
        result["K_Phi"] := K[2];  # Azimuthal current
        result["K_z"] := K[3];    # Axial current
        result["K_t"] := K[4];    # Surface charge density

        return eval(result);
    end proc;

    #
    # TotalCurrentZ: Compute total axial current I_z through the cylinder
    #
    # I_z = integral K^z * R * dPhi = 2*Pi * R * K^z
    #
    TotalCurrentZ := proc(bd, t_val)
        local K, K_z, R;

        K := SurfaceCurrentK(bd, t_val);
        K_z := K[3];
        R := bd:-radius;

        return evalf(2 * Pi * R * K_z);
    end proc;

    #
    # TotalCurrentPhi: Compute total azimuthal current over a length z_length
    #
    # I_Phi = integral K^Phi * dz = K^Phi * z_length
    #
    TotalCurrentPhi := proc(bd, z_length, t_val)
        local K, K_Phi;

        K := SurfaceCurrentK(bd, t_val);
        K_Phi := K[2];

        return evalf(K_Phi * z_length);
    end proc;

    #
    # MagneticFieldAtBoundary: Extract physical magnetic field components
    #
    # For the cylindrically symmetric case:
    #     B_z ~ F_12 (from d_phi/d_rho)
    #     B_Phi ~ F_13 (from d_psi/d_rho)
    #
    MagneticFieldAtBoundary := proc(bd, t_val)
        local R, F, result;

        R := bd:-radius;
        F := bd:-sol_module:-FieldTensorAt(bd:-solution, R, t_val);

        result := table();
        result["F_12"] := F[1, 2];  # Related to B_z
        result["F_13"] := F[1, 3];  # Related to B_Phi
        result["F_24"] := F[2, 4];  # Related to E_Phi
        result["F_34"] := F[3, 4];  # Related to E_z

        return eval(result);
    end proc;

end module:

# Save the module
save CylindricalBoundary, cat(currentdir(), "/CylindricalBoundary.m"):
