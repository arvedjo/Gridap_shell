# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Arved Hess <arved.hess@htwg-konstanz.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays, Gridap.CellData, Gridap.FESpaces
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Logging
using TimerOutputs

#Create a logger that only shows Info and higher
my_logger = ConsoleLogger(stdout, Logging.Info)
const to = TimerOutput()

# Apply it globally
global_logger(my_logger)

"""
Map model nodes via a function to produce a 3D model.
"""
function map_to_3D(fun, model_2d)
    labels = get_face_labeling(model_2d)
    unstructured = UnstructuredDiscreteModel(model_2d)
    grid = get_grid(unstructured)
    node_coordinates = Gridap.ReferenceFEs.get_node_coordinates(grid) # Physical coordinates of nodes
    cell_to_nodes = get_cell_node_ids(grid) # Node IDs for each cell
    cell_reffes = get_cell_reffe(grid)     # Get reference elements for each cell

    new_node_coordinates = map(fun, node_coordinates)
    reffes, cell_types = compress_cell_data(cell_reffes)
    new_grid = UnstructuredGrid(new_node_coordinates, cell_to_nodes, reffes, cell_types)
    return UnstructuredDiscreteModel(new_grid, UnstructuredGridTopology(new_grid), labels)
end

"""
Check that `df` matches a finite-difference tangent of `f` at `p`.
"""
function check_tangent_consistency(
    f, df, x, p=Point(1.0, 1.0); epsilon=sqrt(eps()), verbose=true, T=Float64
)
    function get_element_type(val)
        if isa(val, AbstractArray)
            return eltype(eltype(val))
        else
            return typeof(val)
        end
    end
    element_types = Tuple(get_element_type(xi(p)) for xi in x)
    reffes = Tuple(ReferenceFE(lagrangian, et, 3) for et in element_types)
    triangulations = Tuple(get_triangulation(xi) for xi in x)
    V_spaces = Tuple(FESpace(triangulations[i], reffes[i]) for i in eachindex(reffes))
    rand_vecs = Tuple(
        randn(eltype(et), num_free_dofs(V_spaces[i])) for
        (i, et) in enumerate(element_types)
    )
    δx = Tuple(FEFunction(V_spaces[i], rand_vecs[i]) for i in eachindex(V_spaces))

    d_analytical = df(x..., δx...)
    x_perturbed = Tuple(xi + epsilon * dxi for (xi, dxi) in zip(x, δx))
    d_fd = (f(x_perturbed...) - f(x...)) / epsilon

    err, idx = findmax(reduce(vcat, collect((norm ∘ (d_analytical - d_fd))(p))))
    if verbose
        if isa(p, Point)
            pos_str = "at point $p"
        else
            pos_str = "at index $idx"
        end
        @info "Maximum tangent consistency error: $err\n $pos_str"
    end
    @assert err < 1_000_000 * epsilon "Tangent consistency check failed with error $err"
    return err
end

# ──────────────────────────────────────────────────────────────────────────
# == Section 1: Parameters and Geometry ==
@timeit to "Setting up model parameters and geometry" begin
    @info "Setting up model parameters and geometry..."

    const ρ = 1.016 #radius of the cylinder
    const L = 3.048 #length of the cylinder
    const E = 2.0685e7
    const ν = 0.3
    const μ = E / (2.0 * (1.0 + ν))
    const λ = 2.0 * μ * ν / (1.0 - 2.0 * ν)   #1st Lame parameter
    const λ̄ = (2 * λ * μ) / (λ + 2 * μ)   #Plane stress parameter effective Lame parameter

    const t = 0.03 #thickness of the shell

    domain2D = (-π / 2, π / 2, 0.0, L)
    partition = (21, 21)

    model = CartesianDiscreteModel(domain2D, partition)
    # Simplexify the model to use Triangular elements (PSRI + Bubble relies on this)
    # model = simplexify(model)

    labels = get_face_labeling(model)

    add_tag_from_tags!(labels, "leftright_boundary", [3, 4, 7, 8])
    add_tag_from_tags!(labels, "up_boundary", [1, 2, 5])  # TOP boundary y=0

    degree = 4
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    Xq = get_cell_points(Ω)

    const p_load = Point(0.0, L)
    const δ = DiracDelta(model, p_load)

    const pressure = 0.0 # Surface pressure load in Pa
    # writevtk(model, "../model2D"; labels=labels)
end

# ──────────────────────────────────────────────────────────────────────────
# == Section 2: FE Spaces ==
@timeit to "Creating FE spaces" begin
    @info "Creating FE spaces..."

    reffe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, 3)
    V0_u = TestFESpace(
        model,
        reffe_u;
        conformity=:H1,
        dirichlet_tags=["leftright_boundary", "up_boundary"],
        dirichlet_masks=[(false, false, true), (true, true, true)],
    )

    g_u1(x) = VectorValue(0.0, 0.0, 0.0)
    g_u2(x) = VectorValue(0.0, 0.0, 0.0)
    U_u = TrialFESpace(V0_u, [g_u1, g_u2])

    reffe_β = ReferenceFE(lagrangian, VectorValue{2,Float64}, 3)
    V0_β = TestFESpace(
        model,
        reffe_β;
        conformity=:H1,
        dirichlet_tags=["leftright_boundary", "up_boundary"],
        dirichlet_masks=[(false, true), (true, true)],
    )
    const g_β1(x) = VectorValue(0.0, 0.0)
    const g_β2(x) = VectorValue(0.0, 0.0)
    U_β = TrialFESpace(V0_β, [g_β1, g_β2])

    X = MultiFieldFESpace([U_u, U_β])
    Y = MultiFieldFESpace([V0_u, V0_β])
end

# ──────────────────────────────────────────────────────────────────────────
# == Section 3: Kinematics and Geometry Functions ==
@timeit to "Setting up kinematics and geometry functions" begin
    @info "Setting up kinematics and geometry functions..."

    """
    Map 2D coordinates (φ, y) to 3D half-cylinder (3x1).
    """
    function get_Φ(x2D)
        phi, y = x2D
        x = ρ * sin(phi)
        z = ρ * cos(phi)
        return VectorValue(x, y, z)  # Convert to Cartesian coordinates
    end

    const Φ₀ = interpolate_everywhere(get_Φ, V0_u)

    """
    Compute initial (reference) director as unit normal (n) to shell surface (3x1).
    d₀ = (t₁ × t₂) / |t₁ × t₂|
    """
    function get_d₀(x)
        get_Φ_sv(x) = SVector(get_Φ(x))
        J = ForwardDiff.jacobian(get_Φ_sv, SVector(x))  # 3×2
        t1 = VectorValue(J[1, 1], J[2, 1], J[3, 1])
        t2 = VectorValue(J[1, 2], J[2, 2], J[3, 2])
        n_cross = t1 × t2
        return n_cross / norm(n_cross)
    end

    function get_d₀(x::VectorValue{2,<:Number})
        return get_d₀(SVector(x))
    end

    const d₀ = interpolate_everywhere(get_d₀, V0_u)

    """
    Compute initial rotation angles β₀ from director d₀ (2x1).
    """
    function get_β₀(d₀::VectorValue{3,<:Number})
        b1 = atan(-d₀[2], sqrt(d₀[1]^2 + d₀[3]^2))
        b2 = atan(d₀[1], d₀[3])
        return VectorValue(b1, b2)
    end

    const β₀ = interpolate_everywhere(get_β₀ ∘ get_d₀, V0_β)

    """
    Jacobian (transposed) of `d` w.r.t. β (2x3).
    """
    function Jt(β::VectorValue{2,T}) where {T<:Number}
        s1, c1 = sincos(β[1])
        s2, c2 = sincos(β[2])
        return TensorValue{2,3}(-s1 * s2, c1 * c2, -c1, zero(T), -s1 * c2, -c1 * s2)
    end

    """
    Directional derivative of `Jt` along dβ (2x3).
    """
    function dJt(β::VectorValue{2,T}, dβ::VectorValue{2,<:Number}) where {T<:Number}
        s1, c1 = sincos(β[1])
        s2, c2 = sincos(β[2])

        val11 = (-c1 * s2) * dβ[1] + (-s1 * c2) * dβ[2]
        val21 = (-s1 * c2) * dβ[1] + (-c1 * s2) * dβ[2]

        val12 = s1 * dβ[1]
        val22 = zero(T)

        val13 = (-c1 * c2) * dβ[1] + (s1 * s2) * dβ[2]
        val23 = (s1 * s2) * dβ[1] + (-c1 * c2) * dβ[2]

        return TensorValue{2,3}(val11, val21, val12, val22, val13, val23)
    end

    """
    Director vector from rotation angles β (3x1).
    """
    function director(β::VectorValue{2,T}) where {T<:Number}
        s1, c1 = sincos(β[1])
        s2, c2 = sincos(β[2])
        return VectorValue(c1 * s2, -s1, c1 * c2)
    end

    function director(β)
        return director ∘ (β)
    end

    """
    Gradient of the director vector (2x3).
    """
    function ∇director(β, ∇β)
        return ∇β ⋅ Jt(β)
    end

    """
    Variation of director with respect to β (3x1).
    """
    function δdirector(β, δβ)
        return ∇director(β, δβ)
    end

    """
    Variation of ∇director with respect to β (2x3).
    """
    function δ∇director(β, ∇β, δβ, ∇δβ)
        return ∇δβ ⋅ Jt(β) + ∇β ⋅ (dJt(β, δβ))
    end

    # Test fields for residual verification (used in assertions below)
    x_test = interpolate(x -> rand(VectorValue{2,Float64}), V0_β)
    y_test = interpolate(x -> rand(VectorValue{3,Float64}), V0_u)

    """
    Deformation gradient (∇u' + ∇Φ₀') (3x2).
    """
    F(∇u, ∇Φ₀) = ∇u' + ∇Φ₀'

    """
    Variation of the deformation gradient (3x2).
    """
    dF(∇du) = ∇du' # Variation of F (no constant part)

    a₀(∇Φ₀) = ∇Φ₀ ⋅ ∇Φ₀'
    b₀(∇Φ₀, ∇d₀) = -0.5 * ((∇Φ₀ ⋅ ∇d₀') + (∇d₀ ⋅ ∇Φ₀'))
    j₀ = (sqrt ∘ det ∘ a₀) ∘ ∇(Φ₀)
end

#──────────────────────────────────────────────────────────────────────────
# == Section 4: Strain and Stress Definitions ==
@timeit to "Defining strain and stress tensors" begin
    @info "Defining strain and stress tensors..."

    """
    Nonlinear membrane strain (2x2).
    """
    ε_nl(∇u, ∇Φ₀) = 0.5 * ((F(∇u, ∇Φ₀)' ⋅ F(∇u, ∇Φ₀)) - a₀(∇Φ₀))   # 2x2 matrix membrane strain
    ε_nl(∇u) = ε_nl ∘ (∇u, ∇(Φ₀))

    """
    Variation of nonlinear membrane strain (2x2).
    """
    function dε_nl(∇u, ∇du, ∇Φ₀)
        return 0.5 * (dF(∇du)' ⋅ F(∇u, ∇Φ₀) + F(∇u, ∇Φ₀)' ⋅ dF(∇du))
    end
    dε_nl(∇u, ∇du) = dε_nl ∘ (∇u, ∇du, ∇(Φ₀))

    """
    Bending strain (2x2).
    """
    function κ(∇u, β, ∇β, β₀, ∇β₀, ∇Φ₀, ∇d₀)
        return -0.5 * (
            F(∇u, ∇Φ₀)' ⋅ ∇director(β + β₀, ∇β + ∇β₀)' +
            ∇director(β + β₀, ∇β + ∇β₀) ⋅ F(∇u, ∇Φ₀)
        ) - b₀(∇Φ₀, ∇d₀)
    end # 2x2 matrix bending strain
    κ(∇u, β, ∇β) = κ ∘ (∇u, β, ∇β, β₀, ∇(β₀), ∇(Φ₀), ∇(d₀))

    """
    Variation of bending strain w.r.t u (2x2).
    """
    function dκ_u(β, ∇β, ∇δu, β₀, ∇β₀)
        return -0.5 * (
            dF(∇δu)' ⋅ (∇director(β + β₀, ∇β + ∇β₀))' +
            ∇director(β + β₀, ∇β + ∇β₀) ⋅ dF(∇δu)
        )
    end
    dκ_u(β, ∇β, ∇δu) = dκ_u ∘ (β, ∇β, ∇δu, β₀, ∇(β₀))

    """
    Variation of bending strain w.r.t β (2x2).
    """
    function dκ_β(∇u, β, ∇β, δβ, ∇δβ, β₀, ∇β₀, ∇Φ₀)
        return -0.5 * (
            F(∇u, ∇Φ₀)' ⋅ (δ∇director(β + β₀, ∇β + ∇β₀, δβ, ∇δβ))' +
            δ∇director(β + β₀, ∇β + ∇β₀, δβ, ∇δβ) ⋅ F(∇u, ∇Φ₀)
        )
    end
    dκ_β(∇u, β, ∇β, δβ, ∇δβ) = dκ_β ∘ (∇u, β, ∇β, δβ, ∇δβ, β₀, ∇(β₀), ∇(Φ₀))

    """
    Variation of bending strain (2x2).
    """
    function dκ(∇u, β, ∇β, ∇δu, δβ, ∇δβ)
        return dκ_u(β, ∇β, ∇δu) + dκ_β(∇u, β, ∇β, δβ, ∇δβ)
    end

    """
    Shear strain (2x1).
    """
    γ(∇u, β, β₀, ∇Φ₀) = F(∇u, ∇Φ₀)' ⋅ director(β + β₀) - ∇Φ₀ ⋅ director(β₀)
    γ(∇u, β) = γ ∘ (∇u, β, β₀, ∇(Φ₀))

    """
    Variation of shear strain w.r.t u (2x1).
    """
    function dγ_u(β, ∇du, β₀)
        return dF(∇du)' ⋅ director(β + β₀)
    end
    dγ_u(β, ∇du) = dγ_u ∘ (β, ∇du, β₀)

    """
    Variation of shear strain w.r.t β (2x1).
    """
    function dγ_β(∇u, β, dβ, β₀, ∇Φ₀)
        return F(∇u, ∇Φ₀)' ⋅ δdirector(β + β₀, dβ)
    end
    dγ_β(∇u, β, dβ) = dγ_β ∘ (∇u, β, dβ, β₀, ∇(Φ₀))

    """
    Variation of shear strain (2x1).
    """
    function dγ(∇u, β, ∇du, dβ)
        return dγ_u(β, ∇du) + dγ_β(∇u, β, dβ)
    end

    """
    Plane-stress constitutive tensor Ā (sym 4th-order) (2x2x2x2).
    """
    function getĀ(a₀⁻¹::TensorValue{2,2,T}) where {T<:Number}
        A = @SArray [
            (
                λ̄ * a₀⁻¹[i, j] * a₀⁻¹[k, l] +
                μ * (a₀⁻¹[i, k] * a₀⁻¹[j, l] + a₀⁻¹[i, l] * a₀⁻¹[j, k])
            ) for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        ]

        return Gridap.TensorValues.SymFourthOrderTensorValue(
            A[1, 1, 1, 1],
            A[1, 1, 1, 2],
            A[1, 1, 2, 2],
            A[1, 2, 1, 1],
            A[1, 2, 1, 2],
            A[1, 2, 2, 2],
            A[2, 2, 1, 1],
            A[2, 2, 1, 2],
            A[2, 2, 2, 2],
        )
    end

    """
    Plane-stress constitutive tensor field on the surface (2x2x2x2).
    """
    Ā = (getĀ ∘ inv ∘ a₀) ∘ ∇(Φ₀)

    """
    Surface metric inverse used for shear terms (2x2).
    """
    B̄ = (inv ∘ a₀) ∘ ∇(Φ₀)

    """
    Membrane stress resultant from nonlinear strain (2x2).
    N(ε) = (t * Ā) ⊙ ε    #   Normal Stress SymTensorValue 2x2
    """
    N(∇u) = (t * Ā) ⊙ (ε_nl ∘ (∇u, ∇(Φ₀)))

    """
    Bending moment resultant from curvature strain (2x2).
    M(κ) = ((t^3 / 12) * Ā) ⊙ κ  #   Bending Moment SymTensorValue 2x2
    """
    M(∇u, β, ∇β) = ((t^3 / 12) * Ā) ⊙ (κ ∘ (∇u, β, ∇β, β₀, ∇(β₀), ∇(Φ₀), ∇(d₀)))

    """
    Shear force resultant from shear strain (2x1).
    T(γ) = (t * μ * B̄) ⋅ γ  #   Shear Force VectorValue 2x1
    """
    T(∇u, β) = (t * μ * B̄) ⋅ (γ ∘ (∇u, β, β₀, ∇(Φ₀)))
end

# Energy densities (unused - for reference):
# psi_m(u) = 0.5 * (N(ε_nl(∇(u), ∇(Φ₀))) ⊙ ε_nl(∇(u), ∇(Φ₀)))     # Membrane
# psi_b(u, β) = 0.5 * (M(κ(∇(u), β, ∇(β), ∇(Φ₀))) ⊙ κ(∇(u), β, ∇(β), ∇(Φ₀)))   # Bending
# psi_s(u, β) = 0.5 * (T(γ(∇(u), β, ∇(Φ₀))) ⊙ γ(∇(u), β, ∇(Φ₀)))   # Shear

# ──────────────────────────────────────────────────────────────────────────
# == Section 5: Weak Form and Solver ==
@timeit to "Setting up weak form" begin
    @info "Setting up weak form..."

    """
    Internal residual of the weak form (scalar).
    """
    function res_internal((u, β), (δu, δβ))
        membrane_residual = ∫((N(∇(u)) ⊙ dε_nl(∇(u), ∇(δu))) * j₀) * dΩ
        bending_residual =
            ∫((M(∇(u), β, ∇(β)) ⊙ dκ(∇(u), β, ∇(β), ∇(δu), δβ, ∇(δβ))) * j₀) * dΩ
        shear_residual = ∫((T(∇(u), β) ⋅ dγ(∇(u), β, ∇(δu), δβ)) * j₀) * dΩ

        return membrane_residual + bending_residual + shear_residual
    end

    """
    External load residual (point load only) (scalar).
    """
    function get_res_external(load=0.0, pressure=0.0)
        f_lp(x) = VectorValue(0.0, 0.0, load)
        return function ((δu, δβ))
            return ∫((-pressure * (d₀ ⋅ δu)) * j₀) * dΩ +
                   ∫((VectorValue(0.0, 0.0) ⋅ δβ) * j₀) * dΩ +
                   δ(f_lp ⋅ δu)
        end
    end

    # res_external = get_res_external(10.0, 50.0)
    # @assert sum(res_external((y_test, x_test))) ≈ 10.0 "External load test failed"

end
# ──────────────────────────────────────────────────────────────────────────
# == Section 6: Nonlinear Solver and Output ==
function output_data(uh, βh, step; output_stress=false)
    @timeit to "Writing output data" begin
        if output_stress
            @info "Computing stress for output..."
            # Compute stress resultants and map to output space for smooth visualization
            # Direct Visulatisation of N, M, T fields is possible but may be noisy 
            # due to their dependence on derivatives of FE functions which are not smooth across elements.
            function get_stress_output(stress, reffe)
                dΩ_rec = Measure(Ω, 2)
                V_rec = TestFESpace(
                    model, ReferenceFE(lagrangian, reffe, 1); conformity=:H1
                )
                U_rec = TrialFESpace(V_rec)
                a_N(σ, φ) = ∫((σ ⊙ φ) * j₀) * dΩ_rec
                l_N(φ) = ∫((stress ⊙ φ) * j₀) * dΩ
                op = AffineFEOperator(a_N, l_N, U_rec, V_rec)
                return solve(op)
            end

            σ_N = get_stress_output(N(∇(uh)), Gridap.TensorValues.SymTensorValue{2,Float64})
            σ_M = get_stress_output(
                M(∇(uh), βh, ∇(βh)), Gridap.TensorValues.SymTensorValue{2,Float64}
            )
            σ_T = get_stress_output(T(∇(uh), βh), VectorValue{2,Float64})

            writevtk(
                Ω,
                joinpath(@__DIR__, "solution_shell_$(lpad(step,3,'0'))");
                cellfields=["uh" => uh, "βh" => βh, "N" => σ_N, "M" => σ_M, "T" => σ_T],
            )
        else
            writevtk(
                Ω,
                joinpath(@__DIR__, "solution_shell_$(lpad(step,3,'0'))");
                cellfields=["uh" => uh, "βh" => βh],
            )
        end

        model_3D = map_to_3D(Φ₀, model)
        Ω_3D = Triangulation(model_3D)

        V_curved_u = TestFESpace(
            model_3D,
            reffe_u;
            conformity=:H1,
            dirichlet_tags=["leftright_boundary", "up_boundary"],
            dirichlet_masks=[(false, false, true), (true, true, true)],
        )
        V_curved_β = TestFESpace(
            model_3D,
            reffe_β;
            conformity=:H1,
            dirichlet_tags=["leftright_boundary", "up_boundary"],
            dirichlet_masks=[(false, true), (true, true)],
        )

        uh_3D = FEFunction(V_curved_u, get_free_dof_values(uh))
        βh_3D = FEFunction(V_curved_β, get_free_dof_values(βh))

        return writevtk(
            Ω_3D,
            joinpath(@__DIR__, "solution_curved_shell_$(lpad(step,3,'0'))");
            cellfields=["uh" => uh_3D, "βh" => βh_3D],
        )
    end
end

# Setup nonlinear solver
nls = NLSolver(; show_trace=true, extended_trace=false, method=:newton, iterations=50)
solver = FESolver(nls)
#initial guess
wh0 = interpolate([VectorValue(0.0, 0.0, 0.0), VectorValue(0.0, 0.0)], X)
W = Float64[]   # store load point displacements for each load step

function run(wh, load_z, step, nsteps, cache; output_stress=false)
    println("\n+++ Solving for load $load_z in step $step of $nsteps +++\n")
    res_external = get_res_external(load_z, pressure)
    if wh === nothing
        wh = wh0
    end

    function res((u, β), (δu, δβ))
        return res_internal((u, β), (δu, δβ)) - res_external((δu, δβ))
    end
    op = FEOperator(res, X, Y)
    @timeit to "Load step $step with load $load_z solve!" begin
        wh, cache = solve!(wh, solver, op, cache)
    end
    (uh, βh) = wh
    try
        output_data(uh, βh, step; output_stress=output_stress)
    catch e
        @warn "Output failed for load step $step: $e"
    end

    #output data for load location
    u_lp = uh(p_load)
    push!(W, u_lp[3])
    @info "Displacement at load point $p_load: $u_lp"
    return wh, cache
end

function res((u, β), (δu, δβ))
    return res_internal((u, β), (δu, δβ))
end
op = FEOperator(res, X, Y)

# Reference load solution data in Newton for validation and simulation
y_ref =
    2000 * [
        0.0,
        0.05,
        0.1,
        0.125,
        0.15,
        0.175,
        0.2,
        0.225,
        0.25,
        0.275,
        0.3,
        0.325,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]

# Tangent consistency checks
check_tangent_consistency(ε_nl, dε_nl, (∇(y_test),), Xq)
check_tangent_consistency(κ, dκ, (∇(y_test), x_test, ∇(x_test)), Xq)
check_tangent_consistency(γ, dγ, (∇(y_test), x_test), Xq)

Wh = Any[]  # To store the solution at each load step
Cache = Any[]  # To store the solver cache at each load step (if needed)

#Start Simulation with load stepping and nonlinear solver
try
    @timeit to "Starting nonlinear solver" begin
        cache = nothing
        wh = nothing
        @info "Starting nonlinear solver..."
        for (step, load_z) in enumerate(y_ref)
            @timeit to "Load step $step with load $load_z" begin
                wh, cache = run(wh, -load_z, step, length(y_ref), cache; output_stress=true)
                push!(Wh, wh)
                push!(Cache, cache)
                show(to["Starting nonlinear solver"]["Load step $step with load $load_z"])
            end
        end
    end
finally
    @save joinpath(@__DIR__, "model_state.jld2") W Wh model
    show(to)
end

## Plotting load-displacement curve at the load point
using Plots

x_ref =
    1e-2 * [
        0.0,
        5.421,
        16.1,
        22.195,
        27.657,
        32.7,
        37.582,
        42.633,
        48.537,
        56.355,
        66.410,
        79.810,
        94.669,
        113.704,
        124.751,
        132.653,
        138.920,
        144.185,
        148.770,
        152.863,
        156.584,
        160.015,
        163.211,
        166.200,
        168.973,
        171.505,
    ]

using Plots
plot(x_ref, y_ref; label="Reference Abaqus", xlabel="Displacement at load point [m]")
plot!(
    -W,
    y_ref;
    label="Gridap Simulation",
    xlabel="Displacement at load point [m]",
    linestyle=:dash,
    marker=:star,
)