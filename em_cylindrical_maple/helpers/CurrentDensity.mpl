#
# CurrentDensity.mpl
#
# Current Density Calculations
#
# Computes the 4-current density J^mu from Maxwell's equations in curved spacetime:
#     nabla_mu F^mu_nu = (1/sqrt(-g)) d_mu (sqrt(-g) F^mu_nu) = 4*Pi * J^nu
#
# For vacuum solutions (in vacuo), J^mu = 0 everywhere.
# This module verifies that property and provides tools for computing currents
# when sources are present.
#
# Reference: Misra, M. & Radhakrishna, L. (1962).
#

CurrentDensity := module()
    option package;

    export Create, SqrtNegG, InverseMetric, FUpper, DivergenceF,
           Current4VectorAt, VerifyVacuum, ChargeDensity, CurrentDensityZ;

    local rho, t;

    rho := 'rho';
    t := 't';

    #
    # Create: Initialize current density calculator with a solution
    #
    # Parameters:
    #   solution: A CaseI, CaseII, or CaseIII solution record
    #   solution_module: The solution module (CaseISolution, CaseIISolution, or CaseIIISolution)
    #
    # Returns:
    #   Record containing the solution and module references
    #
    Create := proc(solution, solution_module)
        return Record(
            'solution' = solution,
            'sol_module' = solution_module
        );
    end proc;

    #
    # SqrtNegG: Compute sqrt(-g) = rho * e^(lambda - mu) at a point
    #
    SqrtNegG := proc(cd, rho_val, t_val)
        local lam_val, mu_val;

        lam_val := cd:-sol_module:-Lambda(cd:-solution, rho_val, t_val);
        mu_val := cd:-sol_module:-Mu(cd:-solution, rho_val, t_val);

        return evalf(rho_val * exp(lam_val - mu_val));
    end proc;

    #
    # InverseMetric: Compute inverse metric g^mu_nu at a point
    #
    InverseMetric := proc(cd, rho_val, t_val)
        local lam_val, mu_val, exp_mu_minus_lam, exp_mu, exp_minus_mu;

        lam_val := cd:-sol_module:-Lambda(cd:-solution, rho_val, t_val);
        mu_val := cd:-sol_module:-Mu(cd:-solution, rho_val, t_val);

        exp_mu_minus_lam := evalf(exp(mu_val - lam_val));
        exp_mu := evalf(exp(mu_val));
        exp_minus_mu := evalf(exp(-mu_val));

        return Matrix(4, 4, [
            [-exp_mu_minus_lam, 0, 0, 0],
            [0, -exp_mu / rho_val^2, 0, 0],
            [0, 0, -exp_minus_mu, 0],
            [0, 0, 0, exp_mu_minus_lam]
        ]);
    end proc;

    #
    # FUpper: Compute F^mu_nu (contravariant field tensor) by raising indices
    #
    FUpper := proc(cd, rho_val, t_val)
        local F_lower, g_inv, F_upper, mu_idx, nu_idx;

        F_lower := cd:-sol_module:-FieldTensorAt(cd:-solution, rho_val, t_val);
        g_inv := InverseMetric(cd, rho_val, t_val);

        F_upper := Matrix(4, 4, 0);

        # F^mu_nu = g^mu_mu * g^nu_nu * F_mu_nu (no sum, diagonal metric)
        for mu_idx from 1 to 4 do
            for nu_idx from 1 to 4 do
                F_upper[mu_idx, nu_idx] := g_inv[mu_idx, mu_idx] * g_inv[nu_idx, nu_idx] * F_lower[mu_idx, nu_idx];
            end do;
        end do;

        return evalf(F_upper);
    end proc;

    #
    # DivergenceF: Compute the covariant divergence nabla_mu F^mu_nu = 4*Pi * J^nu
    #
    # Uses: nabla_mu F^mu_nu = (1/sqrt(-g)) d_mu (sqrt(-g) F^mu_nu)
    #
    # Returns: 4-vector (4*Pi*J^rho, 4*Pi*J^Phi, 4*Pi*J^z, 4*Pi*J^t)
    #
    DivergenceF := proc(cd, rho_val, t_val)
        local eps, sqrt_g, div_F, nu_idx,
              sqrt_g_F_rho_nu_plus, sqrt_g_F_rho_nu_minus, d_rho,
              sqrt_g_F_t_nu_plus, sqrt_g_F_t_nu_minus, d_t;

        eps := 1e-6;
        sqrt_g := SqrtNegG(cd, rho_val, t_val);

        div_F := Vector(4, 0);

        for nu_idx from 1 to 4 do
            # Compute d/d_rho (sqrt(-g) * F^rho_nu)
            sqrt_g_F_rho_nu_plus := SqrtNegG(cd, rho_val + eps, t_val) *
                                    FUpper(cd, rho_val + eps, t_val)[1, nu_idx];
            sqrt_g_F_rho_nu_minus := SqrtNegG(cd, rho_val - eps, t_val) *
                                     FUpper(cd, rho_val - eps, t_val)[1, nu_idx];
            d_rho := (sqrt_g_F_rho_nu_plus - sqrt_g_F_rho_nu_minus) / (2 * eps);

            # Compute d/dt (sqrt(-g) * F^t_nu)
            sqrt_g_F_t_nu_plus := SqrtNegG(cd, rho_val, t_val + eps) *
                                  FUpper(cd, rho_val, t_val + eps)[4, nu_idx];
            sqrt_g_F_t_nu_minus := SqrtNegG(cd, rho_val, t_val - eps) *
                                   FUpper(cd, rho_val, t_val - eps)[4, nu_idx];
            d_t := (sqrt_g_F_t_nu_plus - sqrt_g_F_t_nu_minus) / (2 * eps);

            # Note: d/d_Phi and d/d_z terms are zero due to cylindrical symmetry
            div_F[nu_idx] := (d_rho + d_t) / sqrt_g;
        end do;

        return evalf(div_F);
    end proc;

    #
    # Current4VectorAt: Compute the 4-current density J^mu at a point
    #
    # Returns: J^mu = (J^rho, J^Phi, J^z, J^t)
    #
    Current4VectorAt := proc(cd, rho_val, t_val)
        local div_F;

        div_F := DivergenceF(cd, rho_val, t_val);

        return evalf(div_F / (4 * Pi));
    end proc;

    #
    # VerifyVacuum: Verify that J^mu ~= 0 for vacuum solutions
    #
    # Returns table with residuals for each component of J^mu
    #
    VerifyVacuum := proc(cd, rho_val, t_val)
        local J, result;

        J := Current4VectorAt(cd, rho_val, t_val);

        result := table();
        result["J_rho"] := J[1];
        result["J_Phi"] := J[2];
        result["J_z"] := J[3];
        result["J_t"] := J[4];
        result["J_magnitude"] := evalf(sqrt(J[1]^2 + J[2]^2 + J[3]^2 + J[4]^2));

        return eval(result);
    end proc;

    #
    # ChargeDensity: Compute the charge density rho_charge = J^t
    #
    ChargeDensity := proc(cd, rho_val, t_val)
        local J;

        J := Current4VectorAt(cd, rho_val, t_val);
        return J[4];
    end proc;

    #
    # CurrentDensityZ: Compute the axial current density J^z
    #
    CurrentDensityZ := proc(cd, rho_val, t_val)
        local J;

        J := Current4VectorAt(cd, rho_val, t_val);
        return J[3];
    end proc;

end module:

# Save the module
save CurrentDensity, cat(currentdir(), "/CurrentDensity.m"):
