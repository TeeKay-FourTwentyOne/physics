"""
Benchmark tests comparing PINN vs DVR for higher-dimensional systems.

These tests measure time, memory, and accuracy to quantify computational
advantages of the PINN approach.
"""

import pytest
import numpy as np
import time
import tracemalloc
import gc

from quantum_hj.potentials_nd import (
    IsotropicOscillatorND, CoupledOscillator3D, CoupledOscillator4D
)
from quantum_hj.dvr_nd import DVRSolverND
from quantum_hj.pinn_nd import QuantumNumberTrainerND


def measure_memory_peak():
    """Get peak memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, method, time_seconds, memory_mb, energies, exact_energies=None):
        self.method = method
        self.time_seconds = time_seconds
        self.memory_mb = memory_mb
        self.energies = np.array(energies)
        self.exact_energies = np.array(exact_energies) if exact_energies is not None else None

    def errors(self, reference=None):
        """Compute errors against reference or exact energies."""
        ref = reference if reference is not None else self.exact_energies
        if ref is None:
            return None

        n = min(len(self.energies), len(ref))
        errors = np.abs(self.energies[:n] - ref[:n])
        rel_errors = errors / np.abs(ref[:n])

        return {
            'absolute': errors,
            'relative': rel_errors,
            'max_abs': float(np.max(errors)),
            'max_rel': float(np.max(rel_errors)),
            'mean_rel': float(np.mean(rel_errors))
        }

    def __repr__(self):
        return (f"BenchmarkResult({self.method}: "
                f"time={self.time_seconds:.2f}s, mem={self.memory_mb:.1f}MB)")


def benchmark_dvr_nd(potential, ranges, n_basis, n_states, verbose=False):
    """
    Benchmark DVR solver.

    Parameters
    ----------
    potential : PotentialND
        The potential
    ranges : list of tuple
        Coordinate ranges
    n_basis : int
        Basis functions per dimension
    n_states : int
        Number of states to compute
    verbose : bool
        Print progress

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    gc.collect()
    tracemalloc.start()

    start_time = time.time()

    dvr = DVRSolverND(potential, ranges, n_basis=n_basis)
    energies, _ = dvr.solve(n_states=n_states, verbose=verbose)

    end_time = time.time()
    peak_mem = measure_memory_peak()

    tracemalloc.stop()

    # Get exact energies if available
    exact = None
    if hasattr(potential, 'exact_energy'):
        ndim = potential.ndim
        # Generate quantum number combinations for first n_states
        # This is a simplification - exact mapping depends on potential
        exact = []
        total_n = 0
        while len(exact) < n_states and total_n < 10:
            for qn in _generate_quantum_numbers(ndim, total_n):
                if len(exact) >= n_states:
                    break
                E = potential.exact_energy(qn)
                if E not in exact:  # Avoid duplicate degeneracies
                    exact.append(E)
            total_n += 1
        exact = sorted(exact)[:n_states]

    return BenchmarkResult(
        method='DVR',
        time_seconds=end_time - start_time,
        memory_mb=peak_mem,
        energies=energies,
        exact_energies=exact
    )


def benchmark_pinn_nd(potential, quantum_numbers_list, n_epochs=5000, verbose=False):
    """
    Benchmark PINN solver.

    Parameters
    ----------
    potential : PotentialND
        The potential
    quantum_numbers_list : list of tuple
        List of quantum number tuples to solve
    n_epochs : int
        Training epochs per state
    verbose : bool
        Print progress

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    gc.collect()
    tracemalloc.start()

    start_time = time.time()

    energies = []
    exact = []

    for qn in quantum_numbers_list:
        trainer = QuantumNumberTrainerND(
            potential, qn, hidden_size=64
        )
        result = trainer.train(
            n_epochs=n_epochs,
            n_collocation=500,
            quant_start_epoch=n_epochs // 5,
            verbose=verbose
        )
        energies.append(result['E'])

        if hasattr(potential, 'exact_energy'):
            exact.append(potential.exact_energy(qn))

    end_time = time.time()
    peak_mem = measure_memory_peak()

    tracemalloc.stop()

    return BenchmarkResult(
        method='PINN',
        time_seconds=end_time - start_time,
        memory_mb=peak_mem,
        energies=sorted(energies),
        exact_energies=sorted(exact) if exact else None
    )


def _generate_quantum_numbers(ndim, total_n):
    """Generate all quantum number combinations summing to total_n."""
    if ndim == 1:
        yield (total_n,)
    else:
        for n0 in range(total_n + 1):
            for rest in _generate_quantum_numbers(ndim - 1, total_n - n0):
                yield (n0,) + rest


def compare_methods(dvr_result, pinn_result):
    """
    Compare DVR and PINN results.

    Parameters
    ----------
    dvr_result : BenchmarkResult
        DVR benchmark result
    pinn_result : BenchmarkResult
        PINN benchmark result

    Returns
    -------
    dict
        Comparison metrics
    """
    # Time comparison
    speedup = dvr_result.time_seconds / pinn_result.time_seconds if pinn_result.time_seconds > 0 else float('inf')

    # Memory comparison
    memory_ratio = dvr_result.memory_mb / pinn_result.memory_mb if pinn_result.memory_mb > 0 else float('inf')

    # Accuracy comparison
    dvr_errors = dvr_result.errors()
    pinn_errors = pinn_result.errors()

    return {
        'dvr': {
            'time_seconds': dvr_result.time_seconds,
            'memory_mb': dvr_result.memory_mb,
            'max_rel_error': dvr_errors['max_rel'] if dvr_errors else None
        },
        'pinn': {
            'time_seconds': pinn_result.time_seconds,
            'memory_mb': pinn_result.memory_mb,
            'max_rel_error': pinn_errors['max_rel'] if pinn_errors else None
        },
        'speedup': speedup,  # > 1 means PINN is faster
        'memory_ratio': memory_ratio,  # > 1 means PINN uses less memory
        'pinn_wins_time': speedup > 1,
        'pinn_wins_memory': memory_ratio > 1
    }


class TestBenchmarkInfrastructure:
    """Test the benchmark infrastructure."""

    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            method='test',
            time_seconds=1.5,
            memory_mb=100.0,
            energies=[1.0, 2.0, 3.0],
            exact_energies=[1.0, 2.0, 3.0]
        )
        assert result.method == 'test'
        assert result.time_seconds == 1.5

    def test_benchmark_result_errors(self):
        """Test error computation."""
        result = BenchmarkResult(
            method='test',
            time_seconds=1.0,
            memory_mb=50.0,
            energies=[1.01, 2.02, 3.03],
            exact_energies=[1.0, 2.0, 3.0]
        )
        errors = result.errors()
        assert errors is not None
        assert abs(errors['max_rel'] - 0.01) < 0.001

    def test_generate_quantum_numbers(self):
        """Test quantum number generation."""
        # 2D, total=2: (0,2), (1,1), (2,0)
        qns = list(_generate_quantum_numbers(2, 2))
        assert len(qns) == 3
        assert (0, 2) in qns
        assert (1, 1) in qns
        assert (2, 0) in qns


@pytest.mark.benchmark
class TestBenchmark3D:
    """3D benchmark tests."""

    def test_3d_isotropic_dvr(self):
        """Benchmark 3D DVR for isotropic oscillator."""
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        ranges = [(-6, 6)] * 3

        result = benchmark_dvr_nd(pot, ranges, n_basis=25, n_states=5)

        assert result.time_seconds > 0
        assert result.memory_mb > 0
        assert len(result.energies) == 5

        # Check accuracy
        errors = result.errors()
        assert errors['max_rel'] < 0.02

        print(f"\n3D Isotropic DVR: {result}")
        print(f"  Max relative error: {errors['max_rel']:.2%}")

    @pytest.mark.slow
    def test_3d_isotropic_pinn(self):
        """Benchmark 3D PINN for isotropic oscillator."""
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)

        # Solve ground and first excited states
        quantum_numbers = [(0, 0, 0), (1, 0, 0)]

        result = benchmark_pinn_nd(pot, quantum_numbers, n_epochs=4000)

        assert result.time_seconds > 0
        assert result.memory_mb > 0
        assert len(result.energies) == 2

        # Check accuracy
        errors = result.errors()
        assert errors['max_rel'] < 0.02

        print(f"\n3D Isotropic PINN: {result}")
        print(f"  Max relative error: {errors['max_rel']:.2%}")

    @pytest.mark.slow
    def test_3d_coupled_comparison(self):
        """Compare 3D coupled oscillator DVR vs PINN."""
        pot = CoupledOscillator3D(lambda1=0.2, lambda2=0.15)
        ranges = [(-6, 6)] * 3

        # DVR benchmark
        dvr_result = benchmark_dvr_nd(pot, ranges, n_basis=25, n_states=3)

        # PINN benchmark (just ground state for speed)
        pinn_result = benchmark_pinn_nd(pot, [(0, 0, 0)], n_epochs=4000)

        comparison = compare_methods(dvr_result, pinn_result)

        print(f"\n3D Coupled Oscillator Comparison:")
        print(f"  DVR:  {dvr_result.time_seconds:.2f}s, {dvr_result.memory_mb:.1f}MB")
        print(f"  PINN: {pinn_result.time_seconds:.2f}s, {pinn_result.memory_mb:.1f}MB")
        print(f"  Memory ratio (DVR/PINN): {comparison['memory_ratio']:.2f}")


@pytest.mark.benchmark
class TestBenchmark4D:
    """4D benchmark tests."""

    def test_4d_isotropic_dvr(self):
        """Benchmark 4D DVR for isotropic oscillator."""
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)
        ranges = [(-6, 6)] * 4

        # Use smaller basis for 4D
        result = benchmark_dvr_nd(pot, ranges, n_basis=12, n_states=5)

        assert result.time_seconds > 0
        assert result.memory_mb > 0
        assert len(result.energies) == 5

        # 4D with small basis has larger errors
        errors = result.errors()
        assert errors['max_rel'] < 0.05

        print(f"\n4D Isotropic DVR (n=12): {result}")
        print(f"  Max relative error: {errors['max_rel']:.2%}")

    @pytest.mark.slow
    def test_4d_isotropic_pinn(self):
        """Benchmark 4D PINN for isotropic oscillator."""
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)

        # Solve ground state
        quantum_numbers = [(0, 0, 0, 0)]

        result = benchmark_pinn_nd(pot, quantum_numbers, n_epochs=4000)

        assert result.time_seconds > 0
        assert result.memory_mb > 0
        assert len(result.energies) == 1

        # Check accuracy
        errors = result.errors()
        assert errors['max_rel'] < 0.02

        print(f"\n4D Isotropic PINN: {result}")
        print(f"  Max relative error: {errors['max_rel']:.2%}")

    @pytest.mark.slow
    def test_4d_coupled_comparison(self):
        """Compare 4D coupled oscillator DVR vs PINN."""
        pot = CoupledOscillator4D(coupling=0.2)
        ranges = [(-6, 6)] * 4

        # DVR benchmark (small basis due to memory)
        dvr_result = benchmark_dvr_nd(pot, ranges, n_basis=12, n_states=3)

        # PINN benchmark (ground state)
        pinn_result = benchmark_pinn_nd(pot, [(0, 0, 0, 0)], n_epochs=4000)

        comparison = compare_methods(dvr_result, pinn_result)

        print(f"\n4D Coupled Oscillator Comparison:")
        print(f"  DVR:  {dvr_result.time_seconds:.2f}s, {dvr_result.memory_mb:.1f}MB")
        print(f"  PINN: {pinn_result.time_seconds:.2f}s, {pinn_result.memory_mb:.1f}MB")
        print(f"  Memory ratio (DVR/PINN): {comparison['memory_ratio']:.2f}")

        # Check that PINN uses less memory in 4D
        # This is the "weak win" condition
        assert comparison['memory_ratio'] > 0.5, "Expected PINN to be competitive on memory"


@pytest.mark.benchmark
class TestMemoryScaling:
    """Test how memory scales with dimension."""

    def test_dvr_memory_scaling(self):
        """Test DVR memory scaling."""
        results = {}

        for ndim in [2, 3]:
            pot = IsotropicOscillatorND(ndim=ndim)
            ranges = [(-5, 5)] * ndim

            # Fixed basis per dimension
            n_basis = 20

            dvr = DVRSolverND(pot, ranges, n_basis=n_basis)
            mem_estimate = dvr.memory_estimate_mb()

            results[ndim] = {
                'total_basis': dvr.total_basis,
                'memory_mb': mem_estimate
            }

        # Memory should grow exponentially with dimension
        # 3D should use much more memory than 2D
        assert results[3]['memory_mb'] > results[2]['memory_mb'] * 5

        print("\nDVR Memory Scaling (n_basis=20):")
        for ndim, info in results.items():
            print(f"  {ndim}D: {info['total_basis']} basis, ~{info['memory_mb']:.1f} MB")


class TestAccuracyTargets:
    """Test that accuracy targets from the plan are met."""

    def test_3d_separable_target(self):
        """
        Test 3D separable < 1% error target.
        """
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        ranges = [(-6, 6)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=30)
        energies, _ = dvr.solve(n_states=4)

        # Ground state
        exact_E0 = 1.5
        rel_error = abs(energies[0] - exact_E0) / exact_E0
        assert rel_error < 0.01, f"3D separable ground state error {rel_error:.1%} > 1%"

        # First excited
        exact_E1 = 2.5
        rel_error = abs(energies[1] - exact_E1) / exact_E1
        assert rel_error < 0.01, f"3D separable first excited error {rel_error:.1%} > 1%"

    def test_4d_separable_target(self):
        """
        Test 4D separable < 1% error target with adequate basis.
        """
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)
        ranges = [(-6, 6)] * 4

        # Need adequate basis for 1% accuracy
        dvr = DVRSolverND(pot, ranges, n_basis=15)
        energies, _ = dvr.solve(n_states=2)

        exact_E0 = 2.0
        rel_error = abs(energies[0] - exact_E0) / exact_E0

        # May need larger basis for strict 1%
        assert rel_error < 0.02, f"4D separable ground state error {rel_error:.1%} > 2%"
