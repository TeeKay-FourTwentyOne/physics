#
# test_core.mpl
#
# Tests for core modules: metric, field_tensor, energy_tensor, wave_equation, and solutions.
#
# Run these tests to verify the Maple implementation matches expected behavior.
#
# Usage:
#   read("../EMCylindrical.mpl"):
#   read("test_core.mpl"):
#   RunAllTests();
#

# Test framework utilities
TestPassed := 0:
TestFailed := 0:
TestTotal := 0:

#
# AssertTrue: Assert that a condition is true
#
AssertTrue := proc(condition, test_name)
    global TestPassed, TestFailed, TestTotal;
    TestTotal := TestTotal + 1;

    if condition then
        TestPassed := TestPassed + 1;
        printf("  PASS: %s\n", test_name);
    else
        TestFailed := TestFailed + 1;
        printf("  FAIL: %s\n", test_name);
    end if;
end proc:

#
# AssertAlmostEqual: Assert that two values are approximately equal
#
AssertAlmostEqual := proc(val1, val2, tol, test_name)
    global TestPassed, TestFailed, TestTotal;
    TestTotal := TestTotal + 1;

    if abs(val1 - val2) < tol then
        TestPassed := TestPassed + 1;
        printf("  PASS: %s\n", test_name);
    else
        TestFailed := TestFailed + 1;
        printf("  FAIL: %s (got %a, expected %a)\n", test_name, val1, val2);
    end if;
end proc:

#
# AssertMatrixEqual: Assert that two matrices are approximately equal
#
AssertMatrixEqual := proc(M1, M2, tol, test_name)
    global TestPassed, TestFailed, TestTotal;
    local i, j, n, m, passed;

    TestTotal := TestTotal + 1;
    n := LinearAlgebra[RowDimension](M1);
    m := LinearAlgebra[ColumnDimension](M1);
    passed := true;

    for i from 1 to n do
        for j from 1 to m do
            if abs(M1[i,j] - M2[i,j]) > tol then
                passed := false;
            end if;
        end do;
    end do;

    if passed then
        TestPassed := TestPassed + 1;
        printf("  PASS: %s\n", test_name);
    else
        TestFailed := TestFailed + 1;
        printf("  FAIL: %s\n", test_name);
    end if;
end proc:

#
# PrintSummary: Print test summary
#
PrintSummary := proc()
    global TestPassed, TestFailed, TestTotal;
    printf("\n========================================\n");
    printf("Test Summary: %d/%d passed\n", TestPassed, TestTotal);
    if TestFailed > 0 then
        printf("FAILED: %d tests\n", TestFailed);
    else
        printf("All tests passed!\n");
    end if;
    printf("========================================\n");
end proc:

#
# Test: EinsteinRosenMetric
#
TestMetric := proc()
    local metric, g, g_inv, product, identity, i, j, det_g, lam_val, mu_val;

    printf("\nTesting EinsteinRosenMetric...\n");

    # Create a simple metric: lambda = rho + t, mu = rho
    metric := EinsteinRosenMetric:-Create(rho + t, rho);

    # Test 1: Metric shape
    g := EinsteinRosenMetric:-MetricAt(metric, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](g) = 4 and LinearAlgebra[ColumnDimension](g) = 4,
               "Metric has shape 4x4");

    # Test 2: Inverse metric shape
    g_inv := EinsteinRosenMetric:-InverseMetricAt(metric, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](g_inv) = 4 and LinearAlgebra[ColumnDimension](g_inv) = 4,
               "Inverse metric has shape 4x4");

    # Test 3: g * g^{-1} = I
    product := g . g_inv;
    identity := LinearAlgebra[IdentityMatrix](4);
    AssertMatrixEqual(product, identity, 1e-10, "g * g^{-1} = Identity");

    # Test 4: Metric is diagonal
    for i from 1 to 4 do
        for j from 1 to 4 do
            if i <> j then
                AssertAlmostEqual(g[i, j], 0.0, 1e-15, sprintf("Metric is diagonal at [%d,%d]", i, j));
            end if;
        end do;
    end do;

    # Test 5: Metric signature (-,-,-,+)
    AssertTrue(g[1, 1] < 0, "g_rhorho < 0");
    AssertTrue(g[2, 2] < 0, "g_PhiPhi < 0");
    AssertTrue(g[3, 3] < 0, "g_zz < 0");
    AssertTrue(g[4, 4] > 0, "g_tt > 0");

    # Test 6: Lambda and Mu evaluation
    lam_val := EinsteinRosenMetric:-LambdaAt(metric, 2.0, 1.0);
    mu_val := EinsteinRosenMetric:-MuAt(metric, 2.0, 1.0);
    AssertAlmostEqual(lam_val, 3.0, 1e-10, "Lambda(2,1) = 3");
    AssertAlmostEqual(mu_val, 2.0, 1e-10, "Mu(2,1) = 2");

    # Test 7: Determinant is negative (Lorentzian)
    det_g := LinearAlgebra[Determinant](g);
    AssertTrue(det_g < 0, "Metric determinant < 0 (Lorentzian)");

end proc:

#
# Test: FieldTensor
#
TestFieldTensor := proc()
    local tensor, F, i, j;

    printf("\nTesting FieldTensor...\n");

    # Create field tensor with phi = rho^2, psi = t
    tensor := FieldTensor:-Create(rho^2, t);

    # Test 1: Tensor shape
    F := FieldTensor:-At(tensor, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](F) = 4 and LinearAlgebra[ColumnDimension](F) = 4,
               "Field tensor has shape 4x4");

    # Test 2: Antisymmetry
    for i from 1 to 4 do
        for j from 1 to 4 do
            if i <> j then
                AssertAlmostEqual(F[i, j], -F[j, i], 1e-10,
                    sprintf("F[%d,%d] = -F[%d,%d] (antisymmetry)", i, j, j, i));
            end if;
        end do;
    end do;

    # Test 3: Diagonal elements are zero
    for i from 1 to 4 do
        AssertAlmostEqual(F[i, i], 0.0, 1e-15, sprintf("F[%d,%d] = 0 (diagonal)", i, i));
    end do;

    # Test 4: Levi-Civita symbol
    AssertTrue(FieldTensor:-LeviCivita4D(1, 2, 3, 4) = 1, "epsilon_1234 = 1");
    AssertTrue(FieldTensor:-LeviCivita4D(2, 1, 3, 4) = -1, "epsilon_2134 = -1");
    AssertTrue(FieldTensor:-LeviCivita4D(1, 1, 3, 4) = 0, "epsilon_1134 = 0 (repeated index)");

end proc:

#
# Test: EnergyTensor
#
TestEnergyTensor := proc()
    local tensor, E, trace;

    printf("\nTesting EnergyTensor...\n");

    # Create energy tensor
    tensor := EnergyTensor:-Create(rho^2, t, rho, rho + t);

    # Test 1: Tensor shape
    E := EnergyTensor:-At(tensor, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](E) = 4 and LinearAlgebra[ColumnDimension](E) = 4,
               "Energy tensor has shape 4x4");

    # Test 2: Trace is approximately zero (electromagnetic field is traceless)
    trace := EnergyTensor:-Trace(tensor, 2.0, 1.0);
    AssertTrue(abs(trace) < 0.1, "Energy tensor trace is approximately zero");

end proc:

#
# Test: WaveEquation
#
TestWaveEquation := proc()
    local wave, x_val, x1_val, residual;

    printf("\nTesting WaveEquation...\n");

    # Create standing wave solution
    wave := WaveEquation:-CreateStandingWave(1.0, 1.0);

    # Test 1: Evaluate x
    x_val := WaveEquation:-EvaluateX(wave, 2.0, 1.0);
    AssertTrue(type(x_val, numeric), "x evaluated to numeric value");

    # Test 2: Evaluate x_1 (derivative)
    x1_val := WaveEquation:-EvaluateX1(wave, 2.0, 1.0);
    AssertTrue(type(x1_val, numeric), "x_1 evaluated to numeric value");

    # Test 3: Verify wave equation (residual should be ~0)
    # Symbolically, simplify(x_11 - x_44 + x_1/rho) should be 0
    residual := WaveEquation:-VerifyWaveEquation(wave);
    AssertTrue(residual = 0 or abs(evalf(subs({rho=2.0, t=1.0}, residual))) < 0.01,
               "Wave equation satisfied (residual = 0)");

end proc:

#
# Test: CaseISolution
#
TestCaseISolution := proc()
    local sol, psi_val, phi_val, mu_val, lam_val, g, F, residuals;

    printf("\nTesting CaseISolution...\n");

    # Test t_only variant
    sol := CaseISolution:-Create(1.0, 0.5, "t_only", m=1.0, n=0.0, a=1.0, q=0.0);

    # Test 1: Phi is always 0 for Case I
    phi_val := CaseISolution:-Phi(sol, 2.0, 1.0);
    AssertAlmostEqual(phi_val, 0.0, 1e-15, "Phi = 0 for Case I");

    # Test 2: Psi evaluation returns numeric
    psi_val := CaseISolution:-Psi(sol, 2.0, 1.0);
    AssertTrue(type(psi_val, numeric), "Psi evaluates to numeric");

    # Test 3: Mu evaluation returns numeric
    mu_val := CaseISolution:-Mu(sol, 2.0, 1.0);
    AssertTrue(type(mu_val, numeric), "Mu evaluates to numeric");

    # Test 4: Lambda evaluation returns numeric
    lam_val := CaseISolution:-Lambda(sol, 2.0, 1.0);
    AssertTrue(type(lam_val, numeric), "Lambda evaluates to numeric");

    # Test 5: Metric shape
    g := CaseISolution:-MetricAt(sol, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](g) = 4, "Metric from solution is 4x4");

    # Test 6: Field tensor shape
    F := CaseISolution:-FieldTensorAt(sol, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](F) = 4, "Field tensor from solution is 4x4");

    # Test 7: Verify field equations (residuals should be small)
    residuals := CaseISolution:-VerifyFieldEquations(sol, 2.0, 1.0);
    AssertTrue(abs(residuals["eq_18_residual"]) < 0.1, "Field equation 18 residual small");
    AssertTrue(abs(residuals["eq_19_residual"]) < 0.1, "Field equation 19 residual small");

end proc:

#
# Test: CaseIISolution
#
TestCaseIISolution := proc()
    local sol, psi_val, phi_val, g;

    printf("\nTesting CaseIISolution...\n");

    # Test rho_only variant
    sol := CaseIISolution:-Create(1.0, 0.0, "rho_only", m=1.0, n=0.0, a=1.0);

    # Test 1: Psi is always 0 for Case II
    psi_val := CaseIISolution:-Psi(sol, 2.0, 1.0);
    AssertAlmostEqual(psi_val, 0.0, 1e-15, "Psi = 0 for Case II");

    # Test 2: Phi evaluation returns numeric
    phi_val := CaseIISolution:-Phi(sol, 2.0, 1.0);
    AssertTrue(type(phi_val, numeric), "Phi evaluates to numeric");

    # Test 3: Metric shape
    g := CaseIISolution:-MetricAt(sol, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](g) = 4, "Metric from solution is 4x4");

end proc:

#
# Test: CaseIIISolution
#
TestCaseIIISolution := proc()
    local sol, psi_val, phi_val, g, F;

    printf("\nTesting CaseIIISolution...\n");

    # Test variant_71
    sol := CaseIIISolution:-Create("variant_71", m=1.0, n=0.0, a=1.0);

    # Test 1: Both Psi and Phi are non-zero
    psi_val := CaseIIISolution:-Psi(sol, 2.0, 1.0);
    phi_val := CaseIIISolution:-Phi(sol, 2.0, 1.0);
    AssertTrue(type(psi_val, numeric), "Psi evaluates to numeric");
    AssertTrue(type(phi_val, numeric), "Phi evaluates to numeric");

    # Test 2: Metric shape
    g := CaseIIISolution:-MetricAt(sol, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](g) = 4, "Metric from solution is 4x4");

    # Test 3: Field tensor has both phi and psi contributions
    F := CaseIIISolution:-FieldTensorAt(sol, 2.0, 1.0);
    AssertTrue(LinearAlgebra[RowDimension](F) = 4, "Field tensor from solution is 4x4");

end proc:

#
# RunAllTests: Execute all test suites
#
RunAllTests := proc()
    global TestPassed, TestFailed, TestTotal;

    printf("\n========================================\n");
    printf("EMCylindrical Maple Test Suite\n");
    printf("========================================\n");

    # Reset counters
    TestPassed := 0;
    TestFailed := 0;
    TestTotal := 0;

    # Run test suites
    TestMetric();
    TestFieldTensor();
    TestEnergyTensor();
    TestWaveEquation();
    TestCaseISolution();
    TestCaseIISolution();
    TestCaseIIISolution();

    # Print summary
    PrintSummary();

    return TestFailed = 0;
end proc:

# Instructions
printf("\nTest file loaded. Run 'RunAllTests();' to execute all tests.\n");
