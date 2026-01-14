#
# FieldTensor.mpl
#
# Electromagnetic Field Tensor F_mu_nu
#
# The field tensor is defined by equation (5):
#     F_mu_nu = A_mu,nu - A_nu,mu
#
# For the cylindrically symmetric case with 4-potential components
# A_2 = phi/sqrt(8*Pi) and A_3 = psi/sqrt(8*Pi), the non-vanishing components are
# given by equation (9):
#     F_12 = -phi_1/sqrt(8*Pi)  (phi derivative w.r.t. rho)
#     F_13 = -psi_1/sqrt(8*Pi)  (psi derivative w.r.t. rho)
#     F_24 = phi_4/sqrt(8*Pi)   (phi derivative w.r.t. t)
#     F_34 = psi_4/sqrt(8*Pi)   (psi derivative w.r.t. t)
#
# Reference: Misra, M. & Radhakrishna, L. (1962). Equation (9).
#

FieldTensor := module()
    option package;

    export Create, At, GetSymbolicTensor, Invariants, LeviCivita4D;

    local rho, t;

    rho := 'rho';
    t := 't';

    #
    # Create: Build the electromagnetic field tensor from potential expressions
    #
    # Parameters:
    #   phi_expr: Symbolic expression for phi(rho, t)
    #   psi_expr: Symbolic expression for psi(rho, t)
    #
    # Returns:
    #   Record containing symbolic tensor and derivatives
    #
    Create := proc(phi_expr, psi_expr)
        local sqrt_8pi, phi_1, phi_4, psi_1, psi_4, F_sym, result;

        sqrt_8pi := sqrt(8 * Pi);

        # Compute derivatives
        phi_1 := diff(phi_expr, rho);  # d(phi)/d(rho)
        phi_4 := diff(phi_expr, t);    # d(phi)/dt
        psi_1 := diff(psi_expr, rho);  # d(psi)/d(rho)
        psi_4 := diff(psi_expr, t);    # d(psi)/dt

        # Build F_mu_nu (covariant, antisymmetric)
        # Non-zero components from eq (9):
        # F_12 = -phi_1/sqrt(8*Pi)  -> F[1,2]
        # F_13 = -psi_1/sqrt(8*Pi)  -> F[1,3]
        # F_24 = phi_4/sqrt(8*Pi)   -> F[2,4]
        # F_34 = psi_4/sqrt(8*Pi)   -> F[3,4]
        # Plus antisymmetric counterparts

        F_sym := Matrix(4, 4, [
            [0, -phi_1/sqrt_8pi, -psi_1/sqrt_8pi, 0],
            [phi_1/sqrt_8pi, 0, 0, phi_4/sqrt_8pi],
            [psi_1/sqrt_8pi, 0, 0, psi_4/sqrt_8pi],
            [0, -phi_4/sqrt_8pi, -psi_4/sqrt_8pi, 0]
        ]);

        result := Record(
            'phi_sym' = phi_expr,
            'psi_sym' = psi_expr,
            'F_symbolic' = F_sym,
            'phi_1_sym' = phi_1,
            'phi_4_sym' = phi_4,
            'psi_1_sym' = psi_1,
            'psi_4_sym' = psi_4
        );

        return result;
    end proc;

    #
    # At: Evaluate the field tensor numerically at a point
    #
    # Parameters:
    #   tensor_record: Record returned by Create
    #   rho_val: Radial coordinate value
    #   t_val: Time coordinate value
    #
    # Returns:
    #   4x4 Matrix of numerical values for F_mu_nu
    #
    At := proc(tensor_record, rho_val, t_val)
        local F_num, i, j, val;

        F_num := Matrix(4, 4);

        for i from 1 to 4 do
            for j from 1 to 4 do
                val := subs({rho = rho_val, t = t_val}, tensor_record:-F_symbolic[i, j]);
                F_num[i, j] := evalf(val);
            end do;
        end do;

        return F_num;
    end proc;

    #
    # GetSymbolicTensor: Return the symbolic field tensor
    #
    GetSymbolicTensor := proc(tensor_record)
        return tensor_record:-F_symbolic;
    end proc;

    #
    # Invariants: Compute the electromagnetic invariants I_1 and I_2
    #
    # I_1 = F_mu_nu F^mu_nu (scalar invariant)
    # I_2 = F_mu_nu *F^mu_nu (pseudoscalar invariant)
    #
    # Parameters:
    #   tensor_record: Record returned by Create
    #   rho_val, t_val: Point coordinates
    #   g_inv: 4x4 inverse metric Matrix at the point
    #
    # Returns:
    #   List [I_1, I_2]
    #
    Invariants := proc(tensor_record, rho_val, t_val, g_inv)
        local F, F_upper, I1, I2, i, j, k, l, sqrt_neg_g, dual_F, eps_val;

        F := At(tensor_record, rho_val, t_val);

        # Raise indices: F^mu_nu = g^mu_alpha g^nu_beta F_alpha_beta
        F_upper := g_inv . F . LinearAlgebra[Transpose](g_inv);

        # I_1 = F_mu_nu F^mu_nu
        I1 := 0;
        for i from 1 to 4 do
            for j from 1 to 4 do
                I1 := I1 + F[i, j] * F_upper[i, j];
            end do;
        end do;
        I1 := evalf(I1);

        # I_2 requires dual tensor *F^mu_nu
        sqrt_neg_g := evalf(sqrt(-LinearAlgebra[Determinant](LinearAlgebra[MatrixInverse](g_inv))));

        dual_F := Matrix(4, 4, 0);
        for i from 1 to 4 do
            for j from 1 to 4 do
                for k from 1 to 4 do
                    for l from 1 to 4 do
                        eps_val := LeviCivita4D(i, j, k, l);
                        if eps_val <> 0 then
                            dual_F[i, j] := dual_F[i, j] + 0.5 * eps_val * F_upper[k, l] / sqrt_neg_g;
                        end if;
                    end do;
                end do;
            end do;
        end do;

        I2 := 0;
        for i from 1 to 4 do
            for j from 1 to 4 do
                I2 := I2 + F[i, j] * dual_F[i, j];
            end do;
        end do;
        I2 := evalf(I2);

        return [I1, I2];
    end proc;

    #
    # LeviCivita4D: Compute the 4D Levi-Civita symbol epsilon_ijkl
    #
    # Parameters:
    #   i, j, k, l: Indices (1-4)
    #
    # Returns:
    #   +1, -1, or 0
    #
    LeviCivita4D := proc(i, j, k, l)
        local indices, inversions, a, b, unique_count;

        indices := [i, j, k, l];

        # Check for repeated indices
        unique_count := nops({op(indices)});
        if unique_count <> 4 then
            return 0;
        end if;

        # Count inversions (pairs where earlier index > later index)
        inversions := 0;
        for a from 1 to 4 do
            for b from a + 1 to 4 do
                if indices[a] > indices[b] then
                    inversions := inversions + 1;
                end if;
            end do;
        end do;

        if modp(inversions, 2) = 0 then
            return 1;
        else
            return -1;
        end if;
    end proc;

end module:

# Save the module
save FieldTensor, cat(currentdir(), "/FieldTensor.m"):
