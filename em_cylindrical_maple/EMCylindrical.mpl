#
# EMCylindrical.mpl
#
# Einstein-Maxwell Cylindrical Symmetry Solutions
#
# Main package loader for Maple implementation of exact solutions to the
# Einstein-Maxwell field equations with cylindrical symmetry.
#
# Based on: Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields
#           of Cylindrical Symmetry." Proc. Nat. Inst. Sci. India 28A(4), 632-645.
#
# Usage:
#   read("EMCylindrical.mpl"):
#   with(EMCylindrical):
#
#   # Create a Case I solution (Bessel variant)
#   sol := CaseISolution:-Create(1.0, 0.5, "bessel", k_param=1.0, A_param=0.5);
#
#   # Evaluate at a point
#   psi_val := CaseISolution:-Psi(sol, 2.0, 1.0);
#   g := CaseISolution:-MetricAt(sol, 2.0, 1.0);
#
#   # Verify field equations
#   residuals := CaseISolution:-VerifyFieldEquations(sol, 2.0, 1.0);
#

# Load required packages
with(LinearAlgebra):

# Get the directory where this file is located
EMCylindrical_dir := currentdir():

#
# Load core library modules
#
read(cat(EMCylindrical_dir, "/lib/EinsteinRosenMetric.mpl")):
read(cat(EMCylindrical_dir, "/lib/FieldTensor.mpl")):
read(cat(EMCylindrical_dir, "/lib/EnergyTensor.mpl")):
read(cat(EMCylindrical_dir, "/lib/WaveEquation.mpl")):

#
# Load solution modules
#
read(cat(EMCylindrical_dir, "/solutions/CaseISolution.mpl")):
read(cat(EMCylindrical_dir, "/solutions/CaseIISolution.mpl")):
read(cat(EMCylindrical_dir, "/solutions/CaseIIISolution.mpl")):

#
# Load helper modules
#
read(cat(EMCylindrical_dir, "/helpers/CurrentDensity.mpl")):
read(cat(EMCylindrical_dir, "/helpers/CylindricalBoundary.mpl")):

#
# EMCylindrical: Main package wrapper
#
EMCylindrical := module()
    option package;

    # Export all submodules
    export EinsteinRosenMetric, FieldTensor, EnergyTensor, WaveEquation,
           CaseISolution, CaseIISolution, CaseIIISolution,
           CurrentDensity, CylindricalBoundary,
           Version, About;

    #
    # Version: Return package version
    #
    Version := proc()
        return "1.0.0";
    end proc;

    #
    # About: Print package information
    #
    About := proc()
        print("EMCylindrical - Einstein-Maxwell Cylindrical Symmetry Solutions");
        print("================================================================");
        print("");
        print("Version: 1.0.0");
        print("Maple port of Python/SymPy implementation");
        print("");
        print("Based on: Misra & Radhakrishna (1962)");
        print("'Some Electromagnetic Fields of Cylindrical Symmetry'");
        print("Proc. Nat. Inst. Sci. India 28A(4), 632-645");
        print("");
        print("Modules:");
        print("  EinsteinRosenMetric - Metric tensor and Christoffel symbols");
        print("  FieldTensor         - Electromagnetic field tensor F_mu_nu");
        print("  EnergyTensor        - Energy-momentum tensor E^beta_alpha");
        print("  WaveEquation        - Cylindrical wave equation solutions");
        print("  CaseISolution       - phi=0 solutions (3 variants)");
        print("  CaseIISolution      - psi=0 solutions (4 variants)");
        print("  CaseIIISolution     - Both non-zero (4 variants)");
        print("  CurrentDensity      - 4-current density calculations");
        print("  CylindricalBoundary - Surface currents at boundaries");
        print("");
        print("Quick Start:");
        print("  sol := CaseISolution:-Create(1.0, 0.5, 'bessel');");
        print("  psi := CaseISolution:-Psi(sol, 2.0, 1.0);");
        print("  g := CaseISolution:-MetricAt(sol, 2.0, 1.0);");
    end proc;

end module:

# Print confirmation
print("EMCylindrical package loaded. Use 'EMCylindrical:-About()' for information.");
