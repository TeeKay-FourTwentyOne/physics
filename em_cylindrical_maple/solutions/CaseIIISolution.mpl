#
# CaseIIISolution.mpl
#
# Case (iii): phi != 0, psi != 0 Solutions
#
# Both components of the electromagnetic 4-potential are non-zero.
# These solutions represent the most general electromagnetic fields
# with both electric and magnetic components.
#
# Key equations from Misra & Radhakrishna (1962):
# - Specific solutions (71)-(73), (78)
#
# Variants:
# - 'variant_71': psi = a*t + b, phi depends on rho
# - 'variant_72': psi depends on rho, phi depends on t
# - 'variant_73': psi depends on rho (tanh form), phi = a*t + b
# - 'variant_78': phi quadratic in rho (related to Bonnor's theorem)
#
# Reference: Misra, M. & Radhakrishna, L. (1962). Equations (71)-(73), (78).
#

CaseIIISolution := module()
    option package;

    export Create, Psi, Phi, Mu, Lambda,
           MetricAt, InverseMetricAt, FieldTensorAt, EnergyTensorAt,
           GetSymbolicExpressions, VerifyFieldEquations;

    local rho, t, BuildVariant71, BuildVariant72, BuildVariant73, BuildVariant78;

    rho := 'rho';
    t := 't';

    #
    # Create: Construct a Case III solution
    #
    # Parameters:
    #   variant: Solution variant ('variant_71', 'variant_72', 'variant_73', 'variant_78')
    #   Optional keyword parameters:
    #     m, n, a: Parameters for specific solutions
    #     p, q, b_param: Integration constants
    #
    # Returns:
    #   Record containing solution data and functions
    #
    Create := proc(variant := "variant_71", {
        m := 1.0, n := 0.0, a := 1.0,
        p := 0.0, q := 0.0, b_param := 0.0
    })
        local psi_sym, phi_sym, mu_sym, lambda_sym, result;

        # Build symbolic expressions based on variant
        if variant = "variant_71" then
            (psi_sym, phi_sym, mu_sym, lambda_sym) := BuildVariant71(m, n, a, p, q, b_param);
        elif variant = "variant_72" then
            (psi_sym, phi_sym, mu_sym, lambda_sym) := BuildVariant72(m, n, a, p, q, b_param);
        elif variant = "variant_73" then
            (psi_sym, phi_sym, mu_sym, lambda_sym) := BuildVariant73(m, n, a, p, q, b_param);
        elif variant = "variant_78" then
            (psi_sym, phi_sym, mu_sym, lambda_sym) := BuildVariant78(m, n, a, p, q, b_param);
        else
            error "Unknown variant: %1", variant;
        end if;

        result := Record(
            'variant' = variant,
            'm' = m, 'n' = n, 'a' = a,
            'p' = p, 'q' = q, 'b' = b_param,
            'psi_symbolic' = psi_sym,
            'phi_symbolic' = phi_sym,
            'mu_symbolic' = mu_sym,
            'lambda_symbolic' = lambda_sym
        );

        return result;
    end proc;

    #
    # BuildVariant71: Equation (71) - psi linear in t, phi depends on rho
    #
    # psi = a*t + b
    # phi = p - (m/(2*a)) * tanh(-(m/2)*ln(rho) + n)
    # mu = ln[(4*a^2/m^2) * rho^2 * cosh^2(-(m/2)*ln(rho) + n)]
    # lambda = (m^2/2)*ln(rho) + ln[rho^2 * cosh^4(-(m/2)*ln(rho) + n)] + q
    #
    BuildVariant71 := proc(m_val, n_val, a_val, p_val, q_val, b_val)
        local arg, psi_sym, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * ln(rho) + n_val;

        psi_sym := a_val * t + b_val;
        phi_sym := p_val - (m_val / (2*a_val)) * tanh(arg);
        mu_sym := ln((4*a_val^2 / m_val^2) * rho^2 * cosh(arg)^2);
        lambda_sym := (m_val^2 / 2) * ln(rho) + ln(rho^2 * cosh(arg)^4) + q_val;

        return psi_sym, phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # BuildVariant72: Equation (72) - psi quadratic in rho, phi depends on t
    #
    # psi = (1/2)*a*rho^2 + b
    # phi = p - (m/(2*a)) * tanh(-(m/2)*t + n)
    # mu = ln[(4*a^2/m^2) * rho^2 * cosh^2(-(m/2)*t + n)]
    # lambda = (m^2*rho^2)/4 + ln[rho^2 * cosh^4(-(m/2)*t + n)] + q
    #
    BuildVariant72 := proc(m_val, n_val, a_val, p_val, q_val, b_val)
        local arg, psi_sym, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * t + n_val;

        psi_sym := (1/2) * a_val * rho^2 + b_val;
        phi_sym := p_val - (m_val / (2*a_val)) * tanh(arg);
        mu_sym := ln((4*a_val^2 / m_val^2) * rho^2 * cosh(arg)^2);
        lambda_sym := (m_val^2 * rho^2) / 4 + ln(rho^2 * cosh(arg)^4) + q_val;

        return psi_sym, phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # BuildVariant73: Equation (73) - psi depends on rho (tanh), phi linear in t
    #
    # psi = p - (m/(2*a)) * tanh(-(m/2)*ln(rho) + n)
    # phi = a*t + b
    # mu = ln[(m^2/(4*a^2)) * sech^2(-(m/2)*ln(rho) + n)]
    # lambda = (m^2/2)*ln(rho) + q
    #
    BuildVariant73 := proc(m_val, n_val, a_val, p_val, q_val, b_val)
        local arg, psi_sym, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * ln(rho) + n_val;

        psi_sym := p_val - (m_val / (2*a_val)) * tanh(arg);
        phi_sym := a_val * t + b_val;
        mu_sym := ln((m_val^2 / (4*a_val^2)) / cosh(arg)^2);
        lambda_sym := (m_val^2 / 2) * ln(rho) + q_val;

        return psi_sym, phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # BuildVariant78: Equation (78) - phi quadratic in rho (Bonnor's theorem)
    #
    # phi = (1/2)*a*rho^2 + b
    # psi = 0 (derived from case (ii B) via Bonnor's theorem)
    # mu = ln[(m^2/(2*a^2)) * sech^2(-(m/2)*t + n)]
    # lambda = (m^2*rho^2)/4 + q
    #
    BuildVariant78 := proc(m_val, n_val, a_val, p_val, q_val, b_val)
        local arg, psi_sym, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * t + n_val;

        phi_sym := (1/2) * a_val * rho^2 + b_val;
        psi_sym := 0;  # Derived from case (ii B)
        mu_sym := ln((m_val^2 / (2*a_val^2)) / cosh(arg)^2);
        lambda_sym := (m_val^2 * rho^2) / 4 + q_val;

        return psi_sym, phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # Psi: Evaluate psi at (rho, t)
    #
    Psi := proc(sol, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, sol:-psi_symbolic));
    end proc;

    #
    # Phi: Evaluate phi at (rho, t)
    #
    Phi := proc(sol, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, sol:-phi_symbolic));
    end proc;

    #
    # Mu: Evaluate mu at (rho, t)
    #
    Mu := proc(sol, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, sol:-mu_symbolic));
    end proc;

    #
    # Lambda: Evaluate lambda at (rho, t)
    #
    Lambda := proc(sol, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, sol:-lambda_symbolic));
    end proc;

    #
    # MetricAt: Compute the metric tensor at a point
    #
    MetricAt := proc(sol, rho_val, t_val)
        local lam_val, mu_val, exp_lam_minus_mu, exp_minus_mu, exp_mu;

        lam_val := Lambda(sol, rho_val, t_val);
        mu_val := Mu(sol, rho_val, t_val);

        exp_lam_minus_mu := evalf(exp(lam_val - mu_val));
        exp_minus_mu := evalf(exp(-mu_val));
        exp_mu := evalf(exp(mu_val));

        return Matrix(4, 4, [
            [-exp_lam_minus_mu, 0, 0, 0],
            [0, -rho_val^2 * exp_minus_mu, 0, 0],
            [0, 0, -exp_mu, 0],
            [0, 0, 0, exp_lam_minus_mu]
        ]);
    end proc;

    #
    # InverseMetricAt: Compute the inverse metric tensor at a point
    #
    InverseMetricAt := proc(sol, rho_val, t_val)
        local lam_val, mu_val, exp_lam_minus_mu, exp_minus_mu, exp_mu;

        lam_val := Lambda(sol, rho_val, t_val);
        mu_val := Mu(sol, rho_val, t_val);

        exp_lam_minus_mu := evalf(exp(lam_val - mu_val));
        exp_minus_mu := evalf(exp(-mu_val));
        exp_mu := evalf(exp(mu_val));

        return Matrix(4, 4, [
            [-1/exp_lam_minus_mu, 0, 0, 0],
            [0, -exp_mu / rho_val^2, 0, 0],
            [0, 0, -1/exp_mu, 0],
            [0, 0, 0, 1/exp_lam_minus_mu]
        ]);
    end proc;

    #
    # FieldTensorAt: Compute the electromagnetic field tensor F_mu_nu at a point
    #
    # From equation (9), all four non-vanishing components:
    #     F_12 = -phi_1/sqrt(8*Pi)
    #     F_13 = -psi_1/sqrt(8*Pi)
    #     F_24 = phi_4/sqrt(8*Pi)
    #     F_34 = psi_4/sqrt(8*Pi)
    #
    FieldTensorAt := proc(sol, rho_val, t_val)
        local eps, phi_1, phi_4, psi_1, psi_4, sqrt_8pi, F;

        eps := 1e-6;

        # Numerical derivatives
        phi_1 := (Phi(sol, rho_val + eps, t_val) - Phi(sol, rho_val - eps, t_val)) / (2*eps);
        phi_4 := (Phi(sol, rho_val, t_val + eps) - Phi(sol, rho_val, t_val - eps)) / (2*eps);
        psi_1 := (Psi(sol, rho_val + eps, t_val) - Psi(sol, rho_val - eps, t_val)) / (2*eps);
        psi_4 := (Psi(sol, rho_val, t_val + eps) - Psi(sol, rho_val, t_val - eps)) / (2*eps);

        sqrt_8pi := evalf(sqrt(8 * Pi));

        F := Matrix(4, 4, 0);

        # F_12 = -phi_1/sqrt(8*Pi) [indices: F[1,2]]
        F[1, 2] := -phi_1 / sqrt_8pi;
        F[2, 1] := -F[1, 2];

        # F_13 = -psi_1/sqrt(8*Pi) [indices: F[1,3]]
        F[1, 3] := -psi_1 / sqrt_8pi;
        F[3, 1] := -F[1, 3];

        # F_24 = phi_4/sqrt(8*Pi) [indices: F[2,4]]
        F[2, 4] := phi_4 / sqrt_8pi;
        F[4, 2] := -F[2, 4];

        # F_34 = psi_4/sqrt(8*Pi) [indices: F[3,4]]
        F[3, 4] := psi_4 / sqrt_8pi;
        F[4, 3] := -F[3, 4];

        return F;
    end proc;

    #
    # EnergyTensorAt: Compute electromagnetic energy tensor E^beta_alpha at a point
    #
    # From equations (10), for both phi and psi non-zero
    #
    EnergyTensorAt := proc(sol, rho_val, t_val)
        local eps, mu_val, lam_val, phi_1, phi_4, psi_1, psi_4,
              exp_minus_lam, exp_2mu, r, E44, E22, E;

        eps := 1e-6;
        r := rho_val;

        mu_val := Mu(sol, r, t_val);
        lam_val := Lambda(sol, r, t_val);

        # Numerical derivatives
        phi_1 := (Phi(sol, r + eps, t_val) - Phi(sol, r - eps, t_val)) / (2*eps);
        phi_4 := (Phi(sol, r, t_val + eps) - Phi(sol, r, t_val - eps)) / (2*eps);
        psi_1 := (Psi(sol, r + eps, t_val) - Psi(sol, r - eps, t_val)) / (2*eps);
        psi_4 := (Psi(sol, r, t_val + eps) - Psi(sol, r, t_val - eps)) / (2*eps);

        exp_minus_lam := exp(-lam_val);
        exp_2mu := exp(2*mu_val);

        # From equation (10):
        # E^4_4 = -E^1_1 = (1/16*Pi)*e^(-lambda)*[(1/rho^2)*e^(2*mu)*(phi_1^2 + phi_4^2) + psi_1^2 + psi_4^2]
        # E^2_2 = -E^3_3 = (1/16*Pi)*e^(-lambda)*[-(1/rho^2)*e^(2*mu)*(phi_1^2 - phi_4^2) + psi_1^2 - psi_4^2]

        E44 := evalf((1/(16*Pi)) * exp_minus_lam * (
            (1/r^2) * exp_2mu * (phi_1^2 + phi_4^2) + psi_1^2 + psi_4^2
        ));
        E22 := evalf((1/(16*Pi)) * exp_minus_lam * (
            -(1/r^2) * exp_2mu * (phi_1^2 - phi_4^2) + psi_1^2 - psi_4^2
        ));

        E := Matrix(4, 4, 0);
        E[1, 1] := -E44;  # E^1_1
        E[2, 2] := E22;   # E^2_2
        E[3, 3] := -E22;  # E^3_3
        E[4, 4] := E44;   # E^4_4

        return E;
    end proc;

    #
    # GetSymbolicExpressions: Return a table of symbolic expressions
    #
    GetSymbolicExpressions := proc(sol)
        local exprs;

        exprs := table();
        exprs["psi"] := sol:-psi_symbolic;
        exprs["phi"] := sol:-phi_symbolic;
        exprs["mu"] := sol:-mu_symbolic;
        exprs["lambda"] := sol:-lambda_symbolic;

        return eval(exprs);
    end proc;

    #
    # VerifyFieldEquations: Check how well the solution satisfies the field equations (11)-(17)
    #
    VerifyFieldEquations := proc(sol, rho_val, t_val)
        local eps, r, tv, mu_val, mu_1, mu_4, phi_1, phi_4, psi_1, psi_4,
              mu_11, mu_44, phi_11, phi_44, psi_11, psi_44, exp_mu,
              lhs_12, rhs_12, residual_12, residual_15,
              lhs_16, rhs_16, residual_16,
              lhs_17, rhs_17, residual_17, residuals;

        eps := 1e-5;
        r := rho_val;
        tv := t_val;

        mu_val := Mu(sol, r, tv);

        # First derivatives
        mu_1 := (Mu(sol, r + eps, tv) - Mu(sol, r - eps, tv)) / (2*eps);
        mu_4 := (Mu(sol, r, tv + eps) - Mu(sol, r, tv - eps)) / (2*eps);
        phi_1 := (Phi(sol, r + eps, tv) - Phi(sol, r - eps, tv)) / (2*eps);
        phi_4 := (Phi(sol, r, tv + eps) - Phi(sol, r, tv - eps)) / (2*eps);
        psi_1 := (Psi(sol, r + eps, tv) - Psi(sol, r - eps, tv)) / (2*eps);
        psi_4 := (Psi(sol, r, tv + eps) - Psi(sol, r, tv - eps)) / (2*eps);

        # Second derivatives
        mu_11 := (Mu(sol, r + eps, tv) - 2*Mu(sol, r, tv) + Mu(sol, r - eps, tv)) / eps^2;
        mu_44 := (Mu(sol, r, tv + eps) - 2*Mu(sol, r, tv) + Mu(sol, r, tv - eps)) / eps^2;
        phi_11 := (Phi(sol, r + eps, tv) - 2*Phi(sol, r, tv) + Phi(sol, r - eps, tv)) / eps^2;
        phi_44 := (Phi(sol, r, tv + eps) - 2*Phi(sol, r, tv) + Phi(sol, r, tv - eps)) / eps^2;
        psi_11 := (Psi(sol, r + eps, tv) - 2*Psi(sol, r, tv) + Psi(sol, r - eps, tv)) / eps^2;
        psi_44 := (Psi(sol, r, tv + eps) - 2*Psi(sol, r, tv) + Psi(sol, r, tv - eps)) / eps^2;

        exp_mu := exp(mu_val);

        # Check equation (12): mu_11 - mu_44 + mu_1/rho = (e^mu/rho^2)*(phi_1^2 - phi_4^2) - e^(-mu)*(psi_1^2 - psi_4^2)
        lhs_12 := mu_11 - mu_44 + mu_1/r;
        rhs_12 := (exp_mu/r^2)*(phi_1^2 - phi_4^2) - exp(-mu_val)*(psi_1^2 - psi_4^2);
        residual_12 := lhs_12 - rhs_12;

        # Check equation (15): phi_1*psi_1 - phi_4*psi_4 = 0
        residual_15 := phi_1*psi_1 - phi_4*psi_4;

        # Check equation (16): phi_11 - phi_44 - phi_1/rho = -mu_1*phi_1 + mu_4*phi_4
        lhs_16 := phi_11 - phi_44 - phi_1/r;
        rhs_16 := -mu_1*phi_1 + mu_4*phi_4;
        residual_16 := lhs_16 - rhs_16;

        # Check equation (17): psi_11 - psi_44 + psi_1/rho = mu_1*psi_1 - mu_4*psi_4
        lhs_17 := psi_11 - psi_44 + psi_1/r;
        rhs_17 := mu_1*psi_1 - mu_4*psi_4;
        residual_17 := lhs_17 - rhs_17;

        residuals := table();
        residuals["eq_12_residual"] := evalf(residual_12);
        residuals["eq_15_residual"] := evalf(residual_15);
        residuals["eq_16_residual"] := evalf(residual_16);
        residuals["eq_17_residual"] := evalf(residual_17);

        return eval(residuals);
    end proc;

end module:

# Save the module
save CaseIIISolution, cat(currentdir(), "/CaseIIISolution.m"):
