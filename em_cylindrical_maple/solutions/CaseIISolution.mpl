#
# CaseIISolution.mpl
#
# Case (ii): psi = 0 Solutions
#
# When psi = 0, only the phi component of the electromagnetic 4-potential is non-zero.
#
# Key equations from Misra & Radhakrishna (1962):
# - Field equations (46)-(49)
# - General relation (eq 50): g_22 = -alpha - beta*phi + (1/2)*phi^2
# - Specific solutions (56)-(59)
#
# Variants:
# - 'rho_only': phi depends only on rho (eq 56)
# - 't_only': phi depends only on t (eq 59)
# - 'rho_quadratic': phi = (1/2)*a*rho^2 + b (eq 57)
# - 't_linear': phi = a*t + b (eq 58)
#
# Reference: Misra, M. & Radhakrishna, L. (1962). Equations (46)-(49), (56)-(59).
#

CaseIISolution := module()
    option package;

    export Create, Psi, Phi, Mu, Lambda,
           MetricAt, InverseMetricAt, FieldTensorAt, EnergyTensorAt,
           GetSymbolicExpressions, VerifyFieldEquations;

    local rho, t, BuildRhoOnlyVariant, BuildTOnlyVariant,
          BuildRhoQuadraticVariant, BuildTLinearVariant;

    rho := 'rho';
    t := 't';

    #
    # Create: Construct a Case II solution
    #
    # Parameters:
    #   alpha_val: Parameter alpha in the general relation
    #   beta_val: Parameter beta in the general relation
    #   variant: Solution variant ('rho_only', 't_only', 'rho_quadratic', 't_linear')
    #   Optional keyword parameters:
    #     m, n, a: Parameters for specific solutions
    #     p, q, b_param: Integration constants
    #
    # Returns:
    #   Record containing solution data and functions
    #
    Create := proc(alpha_val := 1.0, beta_val := 0.0, variant := "rho_only", {
        m := 1.0, n := 0.0, a := 1.0,
        p := 0.0, q := 0.0, b_param := 0.0
    })
        local psi_sym, phi_sym, mu_sym, lambda_sym, result;

        # psi is always 0 for Case II
        psi_sym := 0;

        # Build symbolic expressions based on variant
        if variant = "rho_only" then
            (phi_sym, mu_sym, lambda_sym) := BuildRhoOnlyVariant(m, n, a, p, q);
        elif variant = "t_only" then
            (phi_sym, mu_sym, lambda_sym) := BuildTOnlyVariant(m, n, a, p, q);
        elif variant = "rho_quadratic" then
            (phi_sym, mu_sym, lambda_sym) := BuildRhoQuadraticVariant(m, n, a, p, b_param);
        elif variant = "t_linear" then
            (phi_sym, mu_sym, lambda_sym) := BuildTLinearVariant(m, n, a, p, b_param);
        else
            error "Unknown variant: %1", variant;
        end if;

        result := Record(
            'alpha' = alpha_val,
            'beta' = beta_val,
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
    # BuildRhoOnlyVariant: Solution where phi depends only on rho (equation 56)
    #
    # phi = p - (m/a) * tanh(-(m/2)*ln(rho) + n)
    # mu = ln[(2*a^2/m^2) * rho^2 * cosh^2(-(m/2)*ln(rho) + n)]
    # lambda = (m^2/2)*ln(rho) + ln[rho^2 * cosh^4(-(m/2)*ln(rho) + n)] + q
    #
    BuildRhoOnlyVariant := proc(m_val, n_val, a_val, p_val, q_val)
        local arg, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * ln(rho) + n_val;

        phi_sym := p_val - (m_val/a_val) * tanh(arg);
        mu_sym := ln((2*a_val^2 / m_val^2) * rho^2 * cosh(arg)^2);
        lambda_sym := (m_val^2 / 2) * ln(rho) + ln(rho^2 * cosh(arg)^4) + q_val;

        return phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # BuildTOnlyVariant: Solution where phi depends only on t (equation 59)
    #
    # phi = p - (m/a) * tanh(-(m/2)*t + n)
    # mu = ln[(2*a^2/m^2) * rho^2 * cosh^2(-(m/2)*t + n)]
    # lambda = (m^2*rho^2)/4 + ln[rho^2 * cosh^4(-(m/2)*t + n)] + q
    #
    BuildTOnlyVariant := proc(m_val, n_val, a_val, p_val, q_val)
        local arg, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * t + n_val;

        phi_sym := p_val - (m_val/a_val) * tanh(arg);
        mu_sym := ln((2*a_val^2 / m_val^2) * rho^2 * cosh(arg)^2);
        lambda_sym := (m_val^2 * rho^2) / 4 + ln(rho^2 * cosh(arg)^4) + q_val;

        return phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # BuildRhoQuadraticVariant: Solution with phi quadratic in rho (equation 57)
    #
    # phi = (1/2)*a*rho^2 + b
    # mu = ln[(m^2/(2*a^2)) * sech^2(-(m/2)*t + n)]
    # lambda = (m^2*rho^2)/4 + p
    #
    BuildRhoQuadraticVariant := proc(m_val, n_val, a_val, p_val, b_val)
        local arg, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * t + n_val;

        phi_sym := (1/2) * a_val * rho^2 + b_val;
        mu_sym := ln((m_val^2 / (2*a_val^2)) / cosh(arg)^2);
        lambda_sym := (m_val^2 * rho^2) / 4 + p_val;

        return phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # BuildTLinearVariant: Solution with phi linear in t (equation 58)
    #
    # phi = a*t + b
    # mu = ln[(m^2/(2*a^2)) * sech^2(-(m/2)*ln(rho) + n)]
    # lambda = (m^2/2)*ln(rho) + p
    #
    BuildTLinearVariant := proc(m_val, n_val, a_val, p_val, b_val)
        local arg, phi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * ln(rho) + n_val;

        phi_sym := a_val * t + b_val;
        mu_sym := ln((m_val^2 / (2*a_val^2)) / cosh(arg)^2);
        lambda_sym := (m_val^2 / 2) * ln(rho) + p_val;

        return phi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # Psi: psi is always 0 for Case II
    #
    Psi := proc(sol, rho_val, t_val)
        return 0.0;
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
    # For Case (ii) with psi = 0:
    #     F_12 = -phi_1/sqrt(8*Pi)
    #     F_13 = 0
    #     F_24 = phi_4/sqrt(8*Pi)
    #     F_34 = 0
    #
    FieldTensorAt := proc(sol, rho_val, t_val)
        local eps, phi_1, phi_4, sqrt_8pi, F;

        eps := 1e-6;

        # Numerical derivatives
        phi_1 := (Phi(sol, rho_val + eps, t_val) - Phi(sol, rho_val - eps, t_val)) / (2*eps);
        phi_4 := (Phi(sol, rho_val, t_val + eps) - Phi(sol, rho_val, t_val - eps)) / (2*eps);

        sqrt_8pi := evalf(sqrt(8 * Pi));

        F := Matrix(4, 4, 0);

        # F_12 = -phi_1/sqrt(8*Pi) [indices: F[1,2]]
        F[1, 2] := -phi_1 / sqrt_8pi;
        F[2, 1] := -F[1, 2];

        # F_24 = phi_4/sqrt(8*Pi) [indices: F[2,4]]
        F[2, 4] := phi_4 / sqrt_8pi;
        F[4, 2] := -F[2, 4];

        return F;
    end proc;

    #
    # EnergyTensorAt: Compute electromagnetic energy tensor E^beta_alpha at a point
    #
    EnergyTensorAt := proc(sol, rho_val, t_val)
        local eps, mu_val, lam_val, phi_1, phi_4, exp_mu, factor, E44, E22, E;

        eps := 1e-6;

        mu_val := Mu(sol, rho_val, t_val);
        lam_val := Lambda(sol, rho_val, t_val);

        # Numerical derivatives
        phi_1 := (Phi(sol, rho_val + eps, t_val) - Phi(sol, rho_val - eps, t_val)) / (2*eps);
        phi_4 := (Phi(sol, rho_val, t_val + eps) - Phi(sol, rho_val, t_val - eps)) / (2*eps);

        # From equation (10), for psi = 0:
        exp_mu := exp(mu_val);
        factor := evalf(exp_mu / (16 * Pi * rho_val^2));

        E44 := factor * (phi_1^2 + phi_4^2);
        E22 := factor * (phi_1^2 - phi_4^2);

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
    # VerifyFieldEquations: Check how well the solution satisfies the field equations
    #
    VerifyFieldEquations := proc(sol, rho_val, t_val)
        local eps, r, tv, mu_val, mu_1, mu_4, phi_1, phi_4,
              mu_11, mu_44, phi_11, phi_44,
              lhs_46, rhs_46, residual_46,
              lhs_47, rhs_47, residual_47, residuals;

        eps := 1e-5;
        r := rho_val;
        tv := t_val;

        mu_val := Mu(sol, r, tv);

        # First derivatives
        mu_1 := (Mu(sol, r + eps, tv) - Mu(sol, r - eps, tv)) / (2*eps);
        mu_4 := (Mu(sol, r, tv + eps) - Mu(sol, r, tv - eps)) / (2*eps);
        phi_1 := (Phi(sol, r + eps, tv) - Phi(sol, r - eps, tv)) / (2*eps);
        phi_4 := (Phi(sol, r, tv + eps) - Phi(sol, r, tv - eps)) / (2*eps);

        # Second derivatives
        mu_11 := (Mu(sol, r + eps, tv) - 2*Mu(sol, r, tv) + Mu(sol, r - eps, tv)) / eps^2;
        mu_44 := (Mu(sol, r, tv + eps) - 2*Mu(sol, r, tv) + Mu(sol, r, tv - eps)) / eps^2;
        phi_11 := (Phi(sol, r + eps, tv) - 2*Phi(sol, r, tv) + Phi(sol, r - eps, tv)) / eps^2;
        phi_44 := (Phi(sol, r, tv + eps) - 2*Phi(sol, r, tv) + Phi(sol, r, tv - eps)) / eps^2;

        # Check equation (46): mu_11 - mu_44 + mu_1/rho = (e^mu/rho^2)*(phi_1^2 - phi_4^2)
        lhs_46 := mu_11 - mu_44 + mu_1/r;
        rhs_46 := (exp(mu_val) / r^2) * (phi_1^2 - phi_4^2);
        residual_46 := lhs_46 - rhs_46;

        # Check equation (47): phi_11 - phi_44 - phi_1/rho = -mu_1*phi_1 + mu_4*phi_4
        lhs_47 := phi_11 - phi_44 - phi_1/r;
        rhs_47 := -mu_1*phi_1 + mu_4*phi_4;
        residual_47 := lhs_47 - rhs_47;

        residuals := table();
        residuals["eq_46_residual"] := evalf(residual_46);
        residuals["eq_47_residual"] := evalf(residual_47);

        return eval(residuals);
    end proc;

end module:

# Save the module
save CaseIISolution, cat(currentdir(), "/CaseIISolution.m"):
