#
# CaseISolution.mpl
#
# Case (i): phi = 0 Solutions
#
# When phi = 0, only the psi component of the electromagnetic 4-potential is non-zero.
#
# Key equations from Misra & Radhakrishna (1962):
# - General relation (eq 22): g_33 = -alpha - beta*psi + (1/2)*psi^2
# - Wave equation for x (eq 29): x_11 - x_44 + x_1/rho = 0
# - Solution (eqs 30-33):
#     psi = beta + sqrt(2*alpha + beta^2) * tanh(x)
#     e^mu = (alpha + (1/2)*beta^2) * sech^2(x)
#     lambda_1 = 2*rho*(x_1^2 + x_4^2)
#     lambda_4 = 4*rho*x_1*x_4
#
# Variants:
# - 'bessel': x = A*J_0(k*rho)*cos(k*t + epsilon) - Bessel function solution
# - 't_only': psi depends only on t (eq 43)
# - 'rho_only': psi linear in t, metric depends on rho (eq 44)
#
# Reference: Misra, M. & Radhakrishna, L. (1962). Equations (30)-(33), (43)-(44).
#

CaseISolution := module()
    option package;

    export Create, Psi, Phi, Mu, Lambda,
           MetricAt, InverseMetricAt, FieldTensorAt, EnergyTensorAt,
           GetSymbolicExpressions, VerifyFieldEquations;

    local rho, t, BuildBesselVariant, BuildTOnlyVariant, BuildRhoOnlyVariant;

    rho := 'rho';
    t := 't';

    #
    # Create: Construct a Case I solution
    #
    # Parameters:
    #   alpha_val: Parameter alpha in the general relation (alpha > 0)
    #   beta_val: Parameter beta in the general relation
    #   variant: Solution variant ('bessel', 't_only', 'rho_only')
    #   Optional keyword parameters:
    #     m, n, a: Parameters for specific solutions
    #     p, q: Integration constants for lambda
    #     k_param, A_param, epsilon: Bessel solution parameters
    #     b_param: Additional parameter for rho_only variant
    #
    # Returns:
    #   Record containing solution data and functions
    #
    Create := proc(alpha_val, beta_val, variant := "bessel", {
        m := 1.0, n := 0.0, a := 1.0,
        p := 0.0, q := 0.0,
        k_param := 1.0, A_param := 1.0, epsilon := 0.0,
        b_param := 0.0
    })
        local psi_sym, phi_sym, mu_sym, lambda_sym, x_sym,
              sqrt_term, result;

        # phi is always 0 for Case I
        phi_sym := 0;

        # Derived quantity
        sqrt_term := sqrt(2*alpha_val + beta_val^2);

        # Build symbolic expressions based on variant
        if variant = "bessel" then
            (psi_sym, mu_sym, lambda_sym, x_sym) := BuildBesselVariant(
                alpha_val, beta_val, k_param, A_param, epsilon, q
            );
        elif variant = "t_only" then
            (psi_sym, mu_sym, lambda_sym) := BuildTOnlyVariant(m, n, a, p, q);
            x_sym := NULL;
        elif variant = "rho_only" then
            (psi_sym, mu_sym, lambda_sym) := BuildRhoOnlyVariant(m, n, a, p, b_param);
            x_sym := NULL;
        else
            error "Unknown variant: %1", variant;
        end if;

        result := Record(
            'alpha' = alpha_val,
            'beta' = beta_val,
            'variant' = variant,
            'm' = m, 'n' = n, 'a' = a,
            'p' = p, 'q' = q,
            'k' = k_param, 'A' = A_param, 'epsilon' = epsilon,
            'b' = b_param,
            'sqrt_term' = sqrt_term,
            'psi_symbolic' = psi_sym,
            'phi_symbolic' = phi_sym,
            'mu_symbolic' = mu_sym,
            'lambda_symbolic' = lambda_sym,
            'x_symbolic' = x_sym
        );

        return result;
    end proc;

    #
    # BuildBesselVariant: Bessel function solution (equations 30-33)
    #
    # x = A * J_0(k*rho) * cos(k*t + epsilon)
    # psi = beta + sqrt(2*alpha + beta^2) * tanh(x)
    # e^mu = (alpha + (1/2)*beta^2) * sech^2(x)
    #
    BuildBesselVariant := proc(alpha_val, beta_val, k_val, A_val, eps_val, q_val)
        local x, sqrt_term, psi_sym, mu_sym, lambda_sym, x1, x4;

        # x = A * J_0(k*rho) * cos(k*t + epsilon)
        x := A_val * BesselJ(0, k_val * rho) * cos(k_val * t + eps_val);

        # psi = beta + sqrt(2*alpha + beta^2) * tanh(x)  [eq 30]
        sqrt_term := sqrt(2*alpha_val + beta_val^2);
        psi_sym := beta_val + sqrt_term * tanh(x);

        # e^mu = (alpha + (1/2)*beta^2) * sech^2(x)  [eq 33]
        mu_sym := ln((alpha_val + (1/2)*beta_val^2) / cosh(x)^2);

        # lambda requires integration of (31) and (32)
        # lambda_1 = 2*rho*(x_1^2 + x_4^2), lambda_4 = 4*rho*x_1*x_4
        # For simplicity, use an approximate form based on x derivatives
        x1 := diff(x, rho);
        x4 := diff(x, t);

        # Approximate lambda (full integration is complex for Bessel functions)
        lambda_sym := rho * (x1^2 + x4^2) + q_val;

        return psi_sym, mu_sym, lambda_sym, x;
    end proc;

    #
    # BuildTOnlyVariant: Solution where psi depends only on t (equation 43)
    #
    # psi = p - (m/a) * tanh(-(m/2)*t + n)
    # mu = ln[(m^2/(2*a^2)) * sech^2(-(m/2)*t + n)]
    # lambda = (m^2*rho^2)/4 + q
    #
    BuildTOnlyVariant := proc(m_val, n_val, a_val, p_val, q_val)
        local arg, psi_sym, mu_sym, lambda_sym;

        arg := -m_val/2 * t + n_val;

        # psi from eq 43
        psi_sym := p_val - (m_val/a_val) * tanh(arg);

        # mu from eq 43
        mu_sym := ln((m_val^2 / (2*a_val^2)) / cosh(arg)^2);

        # lambda from eq 43
        lambda_sym := (m_val^2 * rho^2) / 4 + q_val;

        return psi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # BuildRhoOnlyVariant: Solution variant (equation 44)
    #
    # psi = a*t + b (linear in t)
    # mu = ln[(2*a^2/m^2) * rho^2 * cosh^2((m/2)*ln(rho) + n)]
    # lambda = (m^2/2)*ln(rho) + ln[rho^2 * cosh^4((m/2)*ln(rho) + n)] + p
    #
    BuildRhoOnlyVariant := proc(m_val, n_val, a_val, p_val, b_val)
        local arg, psi_sym, mu_sym, lambda_sym;

        arg := m_val/2 * ln(rho) + n_val;

        # psi = a*t + b (linear in t)
        psi_sym := a_val * t + b_val;

        # mu from eq 44
        mu_sym := ln((2*a_val^2 / m_val^2) * rho^2 * cosh(arg)^2);

        # lambda from eq 44
        lambda_sym := (m_val^2 / 2) * ln(rho) + ln(rho^2 * cosh(arg)^4) + p_val;

        return psi_sym, mu_sym, lambda_sym;
    end proc;

    #
    # Psi: Evaluate psi at (rho, t)
    #
    Psi := proc(sol, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, sol:-psi_symbolic));
    end proc;

    #
    # Phi: phi is always 0 for Case I
    #
    Phi := proc(sol, rho_val, t_val)
        return 0.0;
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
    # Returns: 4x4 Matrix of metric components g_ij
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
    # For Case (i) with phi = 0, from equation (9):
    #     F_12 = -phi_1/sqrt(8*Pi) = 0
    #     F_13 = -psi_1/sqrt(8*Pi)
    #     F_24 = phi_4/sqrt(8*Pi) = 0
    #     F_34 = psi_4/sqrt(8*Pi)
    #
    FieldTensorAt := proc(sol, rho_val, t_val)
        local eps, psi_1, psi_4, sqrt_8pi, F;

        eps := 1e-6;

        # Numerical derivatives
        psi_1 := (Psi(sol, rho_val + eps, t_val) - Psi(sol, rho_val - eps, t_val)) / (2*eps);
        psi_4 := (Psi(sol, rho_val, t_val + eps) - Psi(sol, rho_val, t_val - eps)) / (2*eps);

        sqrt_8pi := evalf(sqrt(8 * Pi));

        F := Matrix(4, 4, 0);

        # F_13 = -psi_1/sqrt(8*Pi) [indices: F[1,3]]
        F[1, 3] := -psi_1 / sqrt_8pi;
        F[3, 1] := -F[1, 3];

        # F_34 = psi_4/sqrt(8*Pi) [indices: F[3,4]]
        F[3, 4] := psi_4 / sqrt_8pi;
        F[4, 3] := -F[3, 4];

        return F;
    end proc;

    #
    # EnergyTensorAt: Compute electromagnetic energy tensor E^beta_alpha at a point
    #
    EnergyTensorAt := proc(sol, rho_val, t_val)
        local eps, mu_val, lam_val, psi_1, psi_4, factor, E44, E22, E;

        eps := 1e-6;

        mu_val := Mu(sol, rho_val, t_val);
        lam_val := Lambda(sol, rho_val, t_val);

        # Numerical derivatives of psi
        psi_1 := (Psi(sol, rho_val + eps, t_val) - Psi(sol, rho_val - eps, t_val)) / (2*eps);
        psi_4 := (Psi(sol, rho_val, t_val + eps) - Psi(sol, rho_val, t_val - eps)) / (2*eps);

        # From equation (10), for phi = 0:
        # E^4_4 = -E^1_1 = (1/16*Pi) * e^(-lambda) * [psi_1^2 + psi_4^2]
        # E^2_2 = -E^3_3 = (1/16*Pi) * e^(-lambda) * [-psi_1^2 + psi_4^2]
        factor := evalf(1 / (16 * Pi) * exp(-lam_val));
        E44 := factor * (psi_1^2 + psi_4^2);
        E22 := factor * (-psi_1^2 + psi_4^2);

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

        if sol:-x_symbolic <> NULL then
            exprs["x"] := sol:-x_symbolic;
        end if;

        return eval(exprs);
    end proc;

    #
    # VerifyFieldEquations: Check how well the solution satisfies the field equations
    #
    # Returns table of residuals (should be close to 0 for valid solutions)
    #
    VerifyFieldEquations := proc(sol, rho_val, t_val)
        local eps, r, tv, mu_val, mu_1, mu_4, psi_1, psi_4,
              mu_11, mu_44, psi_11, psi_44,
              lhs_18, rhs_18, residual_18,
              lhs_19, rhs_19, residual_19, residuals;

        eps := 1e-5;
        r := rho_val;
        tv := t_val;

        # Get function values
        mu_val := Mu(sol, r, tv);

        # First derivatives
        mu_1 := (Mu(sol, r + eps, tv) - Mu(sol, r - eps, tv)) / (2*eps);
        mu_4 := (Mu(sol, r, tv + eps) - Mu(sol, r, tv - eps)) / (2*eps);
        psi_1 := (Psi(sol, r + eps, tv) - Psi(sol, r - eps, tv)) / (2*eps);
        psi_4 := (Psi(sol, r, tv + eps) - Psi(sol, r, tv - eps)) / (2*eps);

        # Second derivatives
        mu_11 := (Mu(sol, r + eps, tv) - 2*Mu(sol, r, tv) + Mu(sol, r - eps, tv)) / eps^2;
        mu_44 := (Mu(sol, r, tv + eps) - 2*Mu(sol, r, tv) + Mu(sol, r, tv - eps)) / eps^2;
        psi_11 := (Psi(sol, r + eps, tv) - 2*Psi(sol, r, tv) + Psi(sol, r - eps, tv)) / eps^2;
        psi_44 := (Psi(sol, r, tv + eps) - 2*Psi(sol, r, tv) + Psi(sol, r, tv - eps)) / eps^2;

        # Check equation (18): mu_11 - mu_44 + mu_1/rho = -e^(-mu)*(psi_1^2 - psi_4^2)
        lhs_18 := mu_11 - mu_44 + mu_1/r;
        rhs_18 := -exp(-mu_val) * (psi_1^2 - psi_4^2);
        residual_18 := lhs_18 - rhs_18;

        # Check equation (19): psi_11 - psi_44 + psi_1/rho = mu_1*psi_1 - mu_4*psi_4
        lhs_19 := psi_11 - psi_44 + psi_1/r;
        rhs_19 := mu_1*psi_1 - mu_4*psi_4;
        residual_19 := lhs_19 - rhs_19;

        residuals := table();
        residuals["eq_18_residual"] := evalf(residual_18);
        residuals["eq_19_residual"] := evalf(residual_19);

        return eval(residuals);
    end proc;

end module:

# Save the module
save CaseISolution, cat(currentdir(), "/CaseISolution.m"):
