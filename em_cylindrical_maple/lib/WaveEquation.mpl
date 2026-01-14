#
# WaveEquation.mpl
#
# Wave Equation Solutions for Cylindrical Symmetry
#
# The wave equation (29) from Misra & Radhakrishna (1962):
#     x_11 - x_44 + x_1/rho = 0
#
# where subscripts 1 and 4 denote partial differentiation with respect
# to rho and t respectively.
#
# This is the cylindrical wave equation in (2+1) dimensions, whose
# general solution can be expressed in terms of Bessel functions:
#     x = sum_n [A_n * J_0(k_n * rho) + B_n * Y_0(k_n * rho)] * cos(k_n * t + epsilon_n)
#
# Reference: Misra, M. & Radhakrishna, L. (1962). Equation (29).
#

WaveEquation := module()
    option package;

    export Create, CreateStandingWave, CreateSuperposition,
           EvaluateX, EvaluateX1, EvaluateX4,
           GetSymbolicX, VerifyWaveEquation;

    local rho, t;

    rho := 'rho';
    t := 't';

    #
    # Create: Construct a cylindrical wave solution from a list of modes
    #
    # Parameters:
    #   modes: List of [k, A, B, epsilon] lists for each mode
    #          k = wave number
    #          A = coefficient of J_0(k*rho)
    #          B = coefficient of Y_0(k*rho) (usually 0 for regularity at rho=0)
    #          epsilon = phase
    #
    # Returns:
    #   Record containing symbolic expressions and evaluation functions
    #
    Create := proc(modes)
        local x_expr, mode, k_val, A_val, B_val, eps_val,
              x_1_expr, x_4_expr, x_11_expr, x_44_expr, result;

        # Build symbolic expression for x
        x_expr := 0;

        for mode in modes do
            k_val := mode[1];
            A_val := mode[2];
            B_val := mode[3];
            eps_val := mode[4];

            # x = A * J_0(k*rho) * cos(k*t + epsilon) + B * Y_0(k*rho) * cos(k*t + epsilon)
            x_expr := x_expr + (A_val * BesselJ(0, k_val * rho) +
                                B_val * BesselY(0, k_val * rho)) * cos(k_val * t + eps_val);
        end do;

        # Compute symbolic derivatives
        x_1_expr := diff(x_expr, rho);
        x_4_expr := diff(x_expr, t);
        x_11_expr := diff(x_expr, rho, rho);
        x_44_expr := diff(x_expr, t, t);

        result := Record(
            'modes' = modes,
            'x_symbolic' = x_expr,
            'x_1_symbolic' = x_1_expr,
            'x_4_symbolic' = x_4_expr,
            'x_11_symbolic' = x_11_expr,
            'x_44_symbolic' = x_44_expr
        );

        return result;
    end proc;

    #
    # CreateStandingWave: Create a simple standing wave solution
    #
    # x = A * J_0(k*rho) * cos(k*t)
    #
    CreateStandingWave := proc(k_val, A_val := 1.0)
        return Create([[k_val, A_val, 0, 0]]);
    end proc;

    #
    # CreateSuperposition: Create a superposition of standing waves
    #
    # x = sum_n A_n * J_0(k_n * rho) * cos(k_n * t)
    #
    CreateSuperposition := proc(k_values, A_values)
        local modes, i, n;

        n := nops(k_values);
        modes := [];

        for i from 1 to n do
            modes := [op(modes), [k_values[i], A_values[i], 0, 0]];
        end do;

        return Create(modes);
    end proc;

    #
    # EvaluateX: Evaluate x at (rho, t)
    #
    EvaluateX := proc(wave_record, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, wave_record:-x_symbolic));
    end proc;

    #
    # EvaluateX1: Evaluate dx/drho at (rho, t)
    #
    EvaluateX1 := proc(wave_record, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, wave_record:-x_1_symbolic));
    end proc;

    #
    # EvaluateX4: Evaluate dx/dt at (rho, t)
    #
    EvaluateX4 := proc(wave_record, rho_val, t_val)
        return evalf(subs({rho = rho_val, t = t_val}, wave_record:-x_4_symbolic));
    end proc;

    #
    # GetSymbolicX: Return the symbolic expression for x
    #
    GetSymbolicX := proc(wave_record)
        return wave_record:-x_symbolic;
    end proc;

    #
    # VerifyWaveEquation: Verify that x satisfies the wave equation
    #
    # Returns the residual: x_11 - x_44 + x_1/rho (should be ~0)
    #
    # Note: This can be done symbolically (simplify should give 0) or numerically
    #
    VerifyWaveEquation := proc(wave_record, rho_val := NULL, t_val := NULL)
        local residual_expr, x_11, x_44, x_1;

        x_11 := wave_record:-x_11_symbolic;
        x_44 := wave_record:-x_44_symbolic;
        x_1 := wave_record:-x_1_symbolic;

        # Form the residual: x_11 - x_44 + x_1/rho
        residual_expr := x_11 - x_44 + x_1/rho;

        # If coordinates provided, evaluate numerically
        if rho_val <> NULL and t_val <> NULL then
            return evalf(subs({rho = rho_val, t = t_val}, residual_expr));
        else
            # Return simplified symbolic expression (should simplify to 0)
            return simplify(residual_expr);
        end if;
    end proc;

end module:

# Save the module
save WaveEquation, cat(currentdir(), "/WaveEquation.m"):
