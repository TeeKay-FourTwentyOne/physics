#
# EnergyTensor.mpl
#
# Electromagnetic Energy-Momentum Tensor E^beta_alpha
#
# From equation (4) in Misra & Radhakrishna (1962):
#     E^beta_alpha = -F_mu_alpha F^mu_beta + (1/4)*delta^beta_alpha F_sigma_epsilon F^sigma_epsilon
#
# The non-vanishing independent components are given by equation (10):
#     E^4_4 = -E^1_1 = (1/16*Pi)*e^(-lambda)*[(1/rho^2)*e^(2*mu)*(phi_1^2 + phi_4^2) + psi_1^2 + psi_4^2]
#     E^2_2 = -E^3_3 = (1/16*Pi)*e^(-lambda)*[-(1/rho^2)*e^(2*mu)*(phi_1^2 - phi_4^2) + psi_1^2 - psi_4^2]
#     E^1_4 = E^4_1 = (1/8*Pi)*e^(-lambda)*[(1/rho^2)*e^(2*mu)*phi_1*phi_4 + psi_1*psi_4]
#     E^2_3 = (1/8*Pi)*e^(-lambda)*[-phi_1*psi_1 + phi_4*psi_4]
#
# Reference: Misra, M. & Radhakrishna, L. (1962). Equations (4) and (10).
#

EnergyTensor := module()
    option package;

    export Create, At, GetSymbolicComponents, Trace, EnergyConditions;

    local rho, t;

    rho := 'rho';
    t := 't';

    #
    # Create: Build the electromagnetic energy tensor from solution components
    #
    # Parameters:
    #   phi_expr: Symbolic expression for phi(rho, t)
    #   psi_expr: Symbolic expression for psi(rho, t)
    #   mu_expr: Symbolic expression for mu(rho, t)
    #   lambda_expr: Symbolic expression for lambda(rho, t)
    #
    # Returns:
    #   Record containing symbolic tensor components
    #
    Create := proc(phi_expr, psi_expr, mu_expr, lambda_expr)
        local phi_1, phi_4, psi_1, psi_4, mu, lam,
              exp_minus_lam, exp_2mu,
              E11_sym, E22_sym, E33_sym, E44_sym, E14_sym, E23_sym,
              result;

        # Compute derivatives
        phi_1 := diff(phi_expr, rho);
        phi_4 := diff(phi_expr, t);
        psi_1 := diff(psi_expr, rho);
        psi_4 := diff(psi_expr, t);

        mu := mu_expr;
        lam := lambda_expr;

        exp_minus_lam := exp(-lam);
        exp_2mu := exp(2*mu);

        # Component expressions from eq (10)
        # E^4_4 = -E^1_1
        E44_sym := (1/(16*Pi)) * exp_minus_lam * (
            (1/rho^2) * exp_2mu * (phi_1^2 + phi_4^2) + psi_1^2 + psi_4^2
        );
        E11_sym := -E44_sym;

        # E^2_2 = -E^3_3
        E22_sym := (1/(16*Pi)) * exp_minus_lam * (
            -(1/rho^2) * exp_2mu * (phi_1^2 - phi_4^2) + psi_1^2 - psi_4^2
        );
        E33_sym := -E22_sym;

        # E^1_4 = E^4_1
        E14_sym := (1/(8*Pi)) * exp_minus_lam * (
            (1/rho^2) * exp_2mu * phi_1 * phi_4 + psi_1 * psi_4
        );

        # E^2_3
        E23_sym := (1/(8*Pi)) * exp_minus_lam * (
            -phi_1 * psi_1 + phi_4 * psi_4
        );

        result := Record(
            'phi_sym' = phi_expr,
            'psi_sym' = psi_expr,
            'mu_sym' = mu_expr,
            'lambda_sym' = lambda_expr,
            'E11_symbolic' = E11_sym,
            'E22_symbolic' = E22_sym,
            'E33_symbolic' = E33_sym,
            'E44_symbolic' = E44_sym,
            'E14_symbolic' = E14_sym,
            'E23_symbolic' = E23_sym
        );

        return result;
    end proc;

    #
    # At: Evaluate the energy tensor E^beta_alpha numerically at a point
    #
    # Parameters:
    #   tensor_record: Record returned by Create
    #   rho_val: Radial coordinate value
    #   t_val: Time coordinate value
    #
    # Returns:
    #   4x4 Matrix of mixed tensor components E^beta_alpha
    #
    At := proc(tensor_record, rho_val, t_val)
        local E, subs_list;

        subs_list := {rho = rho_val, t = t_val};

        E := Matrix(4, 4, 0);

        E[1, 1] := evalf(subs(subs_list, tensor_record:-E11_symbolic));  # E^1_1
        E[2, 2] := evalf(subs(subs_list, tensor_record:-E22_symbolic));  # E^2_2
        E[3, 3] := evalf(subs(subs_list, tensor_record:-E33_symbolic));  # E^3_3
        E[4, 4] := evalf(subs(subs_list, tensor_record:-E44_symbolic));  # E^4_4
        E[1, 4] := evalf(subs(subs_list, tensor_record:-E14_symbolic));  # E^1_4
        E[4, 1] := E[1, 4];                                               # E^4_1 = E^1_4
        E[2, 3] := evalf(subs(subs_list, tensor_record:-E23_symbolic));  # E^2_3

        return E;
    end proc;

    #
    # GetSymbolicComponents: Return a table of symbolic tensor components
    #
    GetSymbolicComponents := proc(tensor_record)
        local components;

        components := table();
        components["E11"] := tensor_record:-E11_symbolic;
        components["E22"] := tensor_record:-E22_symbolic;
        components["E33"] := tensor_record:-E33_symbolic;
        components["E44"] := tensor_record:-E44_symbolic;
        components["E14"] := tensor_record:-E14_symbolic;
        components["E23"] := tensor_record:-E23_symbolic;

        return eval(components);
    end proc;

    #
    # Trace: Compute the trace E^alpha_alpha
    #
    # For electromagnetic fields, the trace should vanish (traceless).
    #
    Trace := proc(tensor_record, rho_val, t_val)
        local E;

        E := At(tensor_record, rho_val, t_val);

        return evalf(E[1, 1] + E[2, 2] + E[3, 3] + E[4, 4]);
    end proc;

    #
    # EnergyConditions: Check various energy conditions at a point
    #
    # Parameters:
    #   tensor_record: Record returned by Create
    #   rho_val, t_val: Point coordinates
    #
    # Returns:
    #   Record with boolean values for energy conditions
    #
    EnergyConditions := proc(tensor_record, rho_val, t_val)
        local E, rho_energy, trace_val, result;

        E := At(tensor_record, rho_val, t_val);

        # Energy density (E^4_4 component)
        rho_energy := E[4, 4];

        # Trace
        trace_val := E[1, 1] + E[2, 2] + E[3, 3] + E[4, 4];

        result := Record(
            'weak_energy' = evalb(rho_energy >= 0),
            'trace_free' = evalb(abs(trace_val) < 1e-10),
            'energy_density' = rho_energy,
            'trace' = trace_val
        );

        return result;
    end proc;

end module:

# Save the module
save EnergyTensor, cat(currentdir(), "/EnergyTensor.m"):
