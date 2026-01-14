#
# EinsteinRosenMetric.mpl
#
# Einstein-Rosen Metric for Cylindrically Symmetric Spacetimes
#
# The metric (equation 1 from Misra & Radhakrishna 1962):
#     ds^2 = e^(lambda-mu)(dt^2 - drho^2) - rho^2*e^(-mu)*dPhi^2 - e^mu*dz^2
#
# Coordinates: (x^1, x^2, x^3, x^4) = (rho, Phi, z, t)
#
# Reference: Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields
#            of Cylindrical Symmetry." Proc. Nat. Inst. Sci. India 28A(4), 632-645.
#

EinsteinRosenMetric := module()
    option package;

    export Create, MetricAt, InverseMetricAt, LambdaAt, MuAt,
           DSSquared, GetSymbolicMetric, GetSymbolicInverse,
           GetDeterminant, ComputeChristoffel;

    local rho, t, Phi, z;

    # Initialize coordinate symbols
    rho := 'rho';
    t := 't';
    Phi := 'Phi';
    z := 'z';

    #
    # Create: Construct an Einstein-Rosen metric from lambda and mu expressions
    #
    # Parameters:
    #   lambda_expr: Symbolic expression for lambda(rho, t)
    #   mu_expr: Symbolic expression for mu(rho, t)
    #
    # Returns:
    #   Record containing symbolic metric, inverse metric, and evaluation functions
    #
    Create := proc(lambda_expr, mu_expr)
        local g, g_inv, det_g, result, lam, mu;

        lam := lambda_expr;
        mu := mu_expr;

        # Build 4x4 metric tensor g_ij (covariant)
        # From ds^2 = e^(lambda-mu)(dt^2 - drho^2) - rho^2*e^(-mu)*dPhi^2 - e^mu*dz^2
        # Coordinates: (rho, Phi, z, t) = indices (1, 2, 3, 4)
        g := Matrix(4, 4, [
            [-exp(lam - mu), 0, 0, 0],
            [0, -rho^2 * exp(-mu), 0, 0],
            [0, 0, -exp(mu), 0],
            [0, 0, 0, exp(lam - mu)]
        ]);

        # Inverse metric g^ij (contravariant)
        g_inv := Matrix(4, 4, [
            [-exp(mu - lam), 0, 0, 0],
            [0, -exp(mu) / rho^2, 0, 0],
            [0, 0, -exp(-mu), 0],
            [0, 0, 0, exp(mu - lam)]
        ]);

        # Determinant of metric
        det_g := LinearAlgebra[Determinant](g);

        # Return as record
        result := Record(
            'lambda_sym' = lam,
            'mu_sym' = mu,
            'g_symbolic' = g,
            'g_inv_symbolic' = g_inv,
            'det_g' = simplify(det_g)
        );

        return result;
    end proc;

    #
    # MetricAt: Evaluate the metric tensor numerically at a point
    #
    # Parameters:
    #   metric_record: Record returned by Create
    #   rho_val: Radial coordinate value (must be > 0)
    #   t_val: Time coordinate value
    #
    # Returns:
    #   4x4 Matrix of numerical values for g_ij
    #
    MetricAt := proc(metric_record, rho_val, t_val)
        local g_num, lam_val, mu_val, exp_lam_minus_mu, exp_minus_mu, exp_mu;

        # Evaluate lambda and mu at the point
        lam_val := evalf(subs({rho = rho_val, t = t_val}, metric_record:-lambda_sym));
        mu_val := evalf(subs({rho = rho_val, t = t_val}, metric_record:-mu_sym));

        # Compute exponentials
        exp_lam_minus_mu := evalf(exp(lam_val - mu_val));
        exp_minus_mu := evalf(exp(-mu_val));
        exp_mu := evalf(exp(mu_val));

        # Build numerical metric
        g_num := Matrix(4, 4, [
            [-exp_lam_minus_mu, 0, 0, 0],
            [0, -rho_val^2 * exp_minus_mu, 0, 0],
            [0, 0, -exp_mu, 0],
            [0, 0, 0, exp_lam_minus_mu]
        ]);

        return evalf(g_num);
    end proc;

    #
    # InverseMetricAt: Evaluate the inverse metric tensor numerically at a point
    #
    InverseMetricAt := proc(metric_record, rho_val, t_val)
        local g_inv_num, lam_val, mu_val, exp_lam_minus_mu, exp_minus_mu, exp_mu;

        lam_val := evalf(subs({rho = rho_val, t = t_val}, metric_record:-lambda_sym));
        mu_val := evalf(subs({rho = rho_val, t = t_val}, metric_record:-mu_sym));

        exp_lam_minus_mu := evalf(exp(lam_val - mu_val));
        exp_minus_mu := evalf(exp(-mu_val));
        exp_mu := evalf(exp(mu_val));

        g_inv_num := Matrix(4, 4, [
            [-1/exp_lam_minus_mu, 0, 0, 0],
            [0, -exp_mu / rho_val^2, 0, 0],
            [0, 0, -1/exp_mu, 0],
            [0, 0, 0, 1/exp_lam_minus_mu]
        ]);

        return evalf(g_inv_num);
    end proc;

    #
    # LambdaAt: Evaluate lambda numerically at a point
    #
    LambdaAt := proc(metric_record, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, metric_record:-lambda_sym));
    end proc;

    #
    # MuAt: Evaluate mu numerically at a point
    #
    MuAt := proc(metric_record, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, metric_record:-mu_sym));
    end proc;

    #
    # DSSquared: Compute the line element ds^2 for given coordinate differentials
    #
    DSSquared := proc(metric_record, rho_val, t_val, drho, dPhi_val, dz_val, dt)
        local g, dx, result, i, j;

        g := MetricAt(metric_record, rho_val, t_val);
        dx := Vector([drho, dPhi_val, dz_val, dt]);

        # Compute g_ij dx^i dx^j
        result := 0;
        for i from 1 to 4 do
            for j from 1 to 4 do
                result := result + g[i, j] * dx[i] * dx[j];
            end do;
        end do;

        return evalf(result);
    end proc;

    #
    # GetSymbolicMetric: Return the symbolic metric tensor
    #
    GetSymbolicMetric := proc(metric_record)
        return metric_record:-g_symbolic;
    end proc;

    #
    # GetSymbolicInverse: Return the symbolic inverse metric tensor
    #
    GetSymbolicInverse := proc(metric_record)
        return metric_record:-g_inv_symbolic;
    end proc;

    #
    # GetDeterminant: Return the symbolic metric determinant
    #
    GetDeterminant := proc(metric_record)
        return metric_record:-det_g;
    end proc;

    #
    # ComputeChristoffel: Compute Christoffel symbols Gamma^i_jk symbolically
    #
    # Parameters:
    #   metric_record: Record returned by Create
    #
    # Returns:
    #   Table with keys [i, j, k] mapping to SymPy expressions (1-indexed)
    #   Only non-zero symbols are stored
    #
    ComputeChristoffel := proc(metric_record)
        local g, g_inv, coords, christoffel, i, j, k, m, gamma_val, term;

        g := metric_record:-g_symbolic;
        g_inv := metric_record:-g_inv_symbolic;
        coords := [rho, Phi, z, t];

        christoffel := table();

        # Gamma^i_jk = (1/2) g^im (g_mj,k + g_mk,j - g_jk,m)
        for i from 1 to 4 do
            for j from 1 to 4 do
                for k from 1 to 4 do
                    gamma_val := 0;
                    for m from 1 to 4 do
                        term := g_inv[i, m] * (
                            diff(g[m, j], coords[k]) +
                            diff(g[m, k], coords[j]) -
                            diff(g[j, k], coords[m])
                        );
                        gamma_val := gamma_val + term;
                    end do;
                    gamma_val := simplify(gamma_val / 2);
                    if gamma_val <> 0 then
                        christoffel[i, j, k] := gamma_val;
                    end if;
                end do;
            end do;
        end do;

        return eval(christoffel);
    end proc;

end module:

# Save the module
save EinsteinRosenMetric, cat(currentdir(), "/EinsteinRosenMetric.m"):
