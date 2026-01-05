# TODO: Check if ∇β has to be transposed and consider using nabla(beta) from Gridap.

using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays
using LineSearches: BackTracking
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
map_to_3D(fun, model_2d)

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

# ──────────────────────────────────────────────────────────────────────────
# == Main Solver Function ==
"""
solve()

Run the nonlinear Naghdi shell solver and write results.
"""
function get_solver()
    # ──────────────────────────────────────────────────────────────────────────
    # == Section 1: Parameters and Geometry ==
    @timeit to "Setting up model parameters and geometry" begin
        @info "Setting up model parameters and geometry..."

        ρ = 1.016 #radius of the cylinder
        L = 3.048 #length of the cylinder
        E = 2.0685e7
        ν = 0.3
        μ = E / (2.0 * (1.0 + ν))
        λ = 2.0 * μ * ν / (1.0 - 2.0 * ν)   #1st Lame parameter
        λ̄ = (2 * λ * μ) / (λ + 2 * μ)   #Plane stress parameter effective Lame parameter

        t = 0.03 #thickness of the shell

        domain2D = (-π / 2, π / 2, 0.0, L)
        partition = (21, 21)

        model = CartesianDiscreteModel(domain2D, partition)

        labels = get_face_labeling(model)
        # topo = get_grid_topology(model)  # Unused - removed

        add_tag_from_tags!(labels, "leftright_boundary", [3, 4, 7, 8])
        add_tag_from_tags!(labels, "up_boundary", [1, 2, 5])  # TOP boundary y=0

        degree = 4
        Ω = Triangulation(model)
        dΩ = Measure(Ω, degree)
        dΩ_h = Measure(Ω, degree - 2)
        # Get characteristic cell size h 
        h = CellField(get_cell_measure(Ω) .^ (1 / num_cell_dims(Ω)), Ω)

        #Weighing factor alpha for curing locking
        α = t^2 / (h * h)

        Xq = get_cell_points(Ω)

        p_load = Point(0.0, L)
        δ = DiracDelta(model, p_load)

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
        # Bubble enrichment (currently unused - for future MITC-type stabilization)
        # reffe_u_bubble = ReferenceFE(bubble, VectorValue{3,Float64})
        # V0_u_bubble = TestFESpace(model, reffe_u_bubble)
        # U_bubble = TrialFESpace(V0_u_bubble)

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
        g_β1(x) = VectorValue(0.0, 0.0)
        g_β2(x) = VectorValue(0.0, 0.0)
        U_β = TrialFESpace(V0_β, [g_β1, g_β2])

        #for bubble not used
        # X = MultiFieldFESpace([U_u, U_bubble, U_β])
        # Y = MultiFieldFESpace([V0_u, V0_bubble, V0_β])
        X = MultiFieldFESpace([U_u, U_β])
        Y = MultiFieldFESpace([V0_u, V0_β])
    end

    # ──────────────────────────────────────────────────────────────────────────
    # == Section 3: Kinematics and Geometry Functions ==
    @timeit to "Setting up kinematics and geometry functions" begin
        @info "Setting up kinematics and geometry functions..."

        """
        get_Φ(x2D)

        Map 2D coordinates (φ, y) to 3D half-cylinder.
        """
        function get_Φ(x2D)
            phi, y = x2D
            x = ρ * sin(phi)
            z = ρ * cos(phi)
            return VectorValue(x, y, z)  # Convert to Cartesian coordinates
        end

        Φ₀ = CellField(get_Φ, Ω)

        """
        tangent_vectors(Φ)

        Compute the two tangent vectors from ∇Φ (2×3).
        """
        function tangent_vectors(Φ::Gridap.CellField)
            function fun(∇Φ)
                @assert size(∇Φ)[1] == 2 "The underlying grid must have dimension 2"
                @assert size(∇Φ)[2] == 3 "The vector length must be 3"
                # ∇Φ is a 2x3 matrix: rows are tangent vectors in 3D
                t1 = VectorValue(∇Φ[1, 1], ∇Φ[1, 2], ∇Φ[1, 3])
                t2 = VectorValue(∇Φ[2, 1], ∇Φ[2, 2], ∇Φ[2, 3])
                return t1, t2
            end
            return fun ∘ (∇(Φ))
        end

        """
        normal_vector(T)

        Unit normal from two tangent vectors.
        """
        function normal_vector(
            T::Tuple{VectorValue{3,Tv},VectorValue{3,Tv}}
        ) where {Tv<:Number}
            n = T[1] × T[2]  # Compute cross product once
            return n / norm(n)
        end

        """
        normal_vector(Φ)

        Unit normal field from a geometry map Φ.
        """
        function normal_vector(Φ::Gridap.CellField)
            T₀ = tangent_vectors(Φ)
            return normal_vector ∘ (T₀)
        end

        N₀ = normal_vector(Φ₀)

        """
        get_β₀(n)

        Reference rotation angles from normal `n`.
        """
        function get_β₀(n::VectorValue{3,T}) where {T<:Number}
            return VectorValue(atan(-n[2], sqrt(n[1]^2 + n[3]^2)), atan(n[1], n[3]))
        end
        β₀ = CellField(get_β₀ ∘ (N₀), Ω)

        # Analytical gradient of β₀ using ForwardDiff
        function analytical_∇β₀(x)
            function compute_beta(x_vec)
                function get_Φ_svec(x)
                    v = get_Φ(x)
                    return SVector(v[1], v[2], v[3])
                end
                J = ForwardDiff.jacobian(get_Φ_svec, x_vec) # 3x2
                t1 = VectorValue(J[1, 1], J[2, 1], J[3, 1])
                t2 = VectorValue(J[1, 2], J[2, 2], J[3, 2])
                n_cross = t1 × t2
                n = n_cross / norm(n_cross)
                b1 = atan(-n[2], sqrt(n[1]^2 + n[3]^2))
                b2 = atan(n[1], n[3])
                return SVector(b1, b2)
            end
            x_svec = SVector(x[1], x[2])
            J_beta = ForwardDiff.jacobian(compute_beta, x_svec)
            return TensorValue{2,2}(J_beta[1, 1], J_beta[1, 2], J_beta[2, 1], J_beta[2, 2])
        end
        ∇β₀ = CellField(analytical_∇β₀, Ω)

        """
        d(β)

        Director vector from rotation angles β (3×1).
        """
        function d(β::VectorValue{2,T}) where {T<:Number}
            s1, c1 = sincos(β[1])
            s2, c2 = sincos(β[2])
            return VectorValue(c1 * s2, -s1, c1 * c2)
        end

        function d(β)
            return d ∘ (β)
        end

        """
        ∇d(β, ∇β)

        Gradient of the director vector (2×3).
        """
        function ∇d(β, ∇β)
            return ∇β ⋅ (Jt ∘ β)
        end

        """
        dd(β, dβ)

        Variation of `d` with respect to β.
        """

        function dd(β, dβ)
            return dβ ⋅ (Jt ∘ (β))
        end

        """
        d∇d(β, ∇β, dβ, ∇dβ)

        Variation of ∇d with respect to β.
        """
        function d∇d(β, ∇β, dβ, ∇dβ)
            return ∇dβ ⋅ (Jt ∘ β) + ∇β ⋅ (dJt ∘ (β, dβ))
        end

        """
        Jt(β)

        Jacobian (transposed) of `d` w.r.t. β (2×3).
        """
        function Jt(β::VectorValue{2,T}) where {T<:Number}
            s1, c1 = sincos(β[1])
            s2, c2 = sincos(β[2])
            return TensorValue{2,3}(-s1 * s2, c1 * c2, -c1, zero(T), -s1 * c2, -c1 * s2)
        end

        """
        dJt(β, dβ)

        Directional derivative of `Jt` along dβ.
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

        # Test fields for residual verification (used in assertions below)
        x_test = interpolate(VectorValue(1.0, 1.0), V0_β)
        y_test = interpolate(VectorValue(1.0, 1.0, 1.0), V0_u)

        """
        F(∇u)

        Deformation gradient (∇u' + ∇Φ₀').
        """
        F(∇u) = ∇u' + ∇(Φ₀)'

        """
        dF(∇du)

        Variation of the deformation gradient.
        """
        dF(∇du) = ∇du' # Variation of F (no constant part)

        a₀ = ∇(Φ₀) ⋅ ∇(Φ₀)'
        a₀⁻¹ = inv(a₀)
        b₀ = -0.5 * ((∇(Φ₀) ⋅ ∇d(β₀, ∇β₀)') + (∇d(β₀, ∇β₀) ⋅ ∇(Φ₀)'))
        j₀ = sqrt ∘ det(a₀)
    end

    # ──────────────────────────────────────────────────────────────────────────
    # == Section 4: Strain and Stress Definitions ==
    @timeit to "Defining strain and stress tensors" begin
        @info "Defining strain and stress tensors..."

        """
        ε_nl(∇u)

        Nonlinear membrane strain (2×2).
        """
        ε_nl(∇u) = 0.5 * (F(∇u)' ⋅ F(∇u) - a₀)    # 2x2 matrix membrane strain

        """
        dε_nl(∇u, ∇du)

        Variation of nonlinear membrane strain.
        """
        function dε_nl(∇u, ∇du)
            return 0.5 * (dF(∇du)' ⋅ F(∇u) + F(∇u)' ⋅ dF(∇du))
        end

        """
        κ(∇u, β, ∇β)

        Bending strain (2×2).
        """
        function κ(∇u, β, ∇β)
            return -0.5 * (F(∇u)' ⋅ ∇d(β + β₀, ∇β + ∇β₀)' + ∇d(β + β₀, ∇β + ∇β₀) ⋅ F(∇u)) -
                   b₀
        end # 2x2 matrix bending strain

        """
        dκ(∇u, β, ∇β, ∇du, dβ, ∇dβ)

        Variation of bending strain.
        """
        function dκ(∇u, β, ∇β, ∇du, dβ, ∇dβ)
            return -0.5 * (
                dF(∇du)' ⋅ (∇d(β + β₀, ∇β + ∇β₀))' +
                F(∇u)' ⋅ (d∇d(β + β₀, ∇β + ∇β₀, dβ, ∇dβ))' +
                d∇d(β + β₀, ∇β + ∇β₀, dβ, ∇dβ) ⋅ F(∇u) +
                ∇d(β + β₀, ∇β + ∇β₀) ⋅ dF(∇du)
            )
        end

        """
        γ(∇u, β)

        Shear strain (2×1).
        """
        γ(∇u, β) = F(∇u)' ⋅ d(β + β₀) - ∇(Φ₀) ⋅ d(β₀)

        """
        dγ(∇u, β, ∇du, dβ)

        Variation of shear strain.
        """
        function dγ(∇u, β, ∇du, dβ)
            return dF(∇du)' ⋅ d(β + β₀) + F(∇u)' ⋅ dd(β + β₀, dβ)
        end

        """
        getĀ(a₀⁻¹)

        Plane-stress constitutive tensor Ā (sym 4th-order).
        """
        function getĀ(a₀⁻¹::TensorValue{2,2,T}) where {T<:Number}
            A = zeros(MArray{Tuple{2,2,2,2},T}) #preallocate
            for i in 1:2, j in 1:2, k in 1:2, l in 1:2
                A[i, j, k, l] =
                    λ̄ * a₀⁻¹[i, j] * a₀⁻¹[k, l] +
                    μ * (a₀⁻¹[i, k] * a₀⁻¹[j, l] + a₀⁻¹[i, l] * a₀⁻¹[j, k])
            end

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

        Ā = getĀ ∘ (a₀⁻¹)

        N(ε) = t * Ā ⊙ ε    #   Normal Stress SymTensorValue 2x2
        M(κ) = (t^3 / 12) * Ā ⊙ κ  #   Bending Moment SymTensorValue 2x2
        T(γ) = t * μ * a₀⁻¹ ⋅ γ#   Shear Force TensorValue 2x1
    end

    # Energy densities (unused - for reference):
    # psi_m(u) = 0.5 * (N(ε_nl(u)) ⊙ ε_nl(u))     # Membrane
    # psi_b(u, β) = 0.5 * (M(κ(u, β)) ⊙ κ(u, β))   # Bending
    # psi_s(u, β) = 0.5 * (T(γ(u, β)) ⊙ γ(u, β))   # Shear

    # ──────────────────────────────────────────────────────────────────────────
    # == Section 5: Weak Form and Solver ==
    @timeit to "Setting up weak form" begin
        @info "Setting up weak form..."

        """
        res_internal((u, β), (δu, δβ))

        Internal residual of the weak form.
        """
        function res_internal((u, β), (δu, δβ))
            return ∫(
                (
                    α * (N(ε_nl(∇(u))) ⊙ dε_nl(∇(u), ∇(δu))) +
                    M(κ(∇(u), β, ∇(β))) ⊙ dκ(∇(u), β, ∇(β), ∇(δu), δβ, ∇(δβ)) +
                    α * (T(γ(∇(u), β)) ⋅ dγ(∇(u), β, ∇(δu), δβ))
                ) * j₀,
            ) * dΩ +
                   ∫(
                (
                    (1 - α) * (N(ε_nl(∇(u))) ⊙ dε_nl(∇(u), ∇(δu))) +
                    (1 - α) * T(γ(∇(u), β)) ⋅ dγ(∇(u), β, ∇(δu), δβ)
                ) * j₀,
            ) * dΩ_h
        end

        #test
        # isa(sum(res_internal((y, x), (y, x))), Number)

        """
        res_external((δu, δβ))

        External load residual (point load only).
        """
        function get_res_external(load)
            f_lp(x) = VectorValue(0.0, 0.0, load)
            return function res_external((δu, δβ))
                return ∫(VectorValue(0.0, 0.0, 0.0) ⋅ δu) * dΩ +
                       ∫(VectorValue(0.0, 0.0) ⋅ δβ) * dΩ +
                       δ(f_lp ⋅ δu)
            end
        end

        res_external = get_res_external(10.0)
        @assert sum(res_external((y_test, x_test))) ≈ 10.0 "External load test failed"

        #test
        # isa(sum(res((y, y, x), (y, y, x))), Number)
    end
    # jacob = jacobian(op, wh)
    # K = get_matrix(jacob)
    # evd = eigen(Matrix(jacob); sortby=abs) #trying to see if the stiffness matrix is pos def and looking for spurious modes

    function output_data(uh, βh, step)
        @timeit to "Writing output data" begin
            #get stresses
            σm = N(ε_nl(∇(uh)))
            writevtk(
                Ω,
                joinpath(@__DIR__, "solution_shell_$(lpad(step,3,'0'))");
                cellfields=["uh" => uh, "βh" => βh, "σm" => σm],
            )

            newModel = map_to_3D(Φ₀, model)
            newΩ = Triangulation(newModel)

            V_curved_u = TestFESpace(
                newModel,
                reffe_u;
                conformity=:H1,
                dirichlet_tags=["leftright_boundary", "up_boundary"],
                dirichlet_masks=[(false, false, true), (true, true, true)],
            )
            V_curved_β = TestFESpace(
                newModel,
                reffe_β;
                conformity=:H1,
                dirichlet_tags=["leftright_boundary", "up_boundary"],
                dirichlet_masks=[(false, true), (true, true)],
            )

            uh_new = FEFunction(V_curved_u, get_free_dof_values(uh))
            βh_new = FEFunction(V_curved_β, get_free_dof_values(βh))

            return writevtk(
                newΩ,
                joinpath(@__DIR__, "solution_curved_shell_$(lpad(step,3,'0'))");
                cellfields=["uh" => uh_new, "βh" => βh_new],
            )
        end
    end

    nls = NLSolver(;
        show_trace=true,
        extended_trace=false,
        method=:newton,
        linesearch=BackTracking(),
        iterations=50,
    )
    solver = FESolver(nls)
    #initial guess
    wh0 = interpolate([VectorValue(0.0, 0.0, 0.0), VectorValue(0.0, 0.0)], X)

    function run(wh, load_z, step, nsteps, cache)
        println("\n+++ Solving for load $load_z in step $step of $nsteps +++\n")
        res_external = get_res_external(load_z)
        if wh === nothing
            wh = wh0
        end
        #TODO with bubble all bubble results are always zero - check if bubble enrichment is working
        # function res((u_l, u_b, β), (δu_l, δu_b, δβ))
        #     u = u_l + u_b
        #     δu = δu_l + δu_b
        #     return res_internal((u, β), (δu, δβ)) - res_external((δu, δβ))
        # end
        function res((u, β), (δu, δβ))
            return res_internal((u, β), (δu, δβ)) - res_external((δu, δβ))
        end
        op = FEOperator(res, X, Y)
        @timeit to "Load step $step with load $load_z solve!" begin
            wh, cache = solve!(wh, solver, op, cache)
        end
        (uh, βh) = wh

        output_data(uh, βh, step)

        return wh, cache
    end
    return run
end

try
    cache = nothing
    wh = nothing
    #for bubble enrichment (currently unused - for future MITC-type stabilization)
    # wh = interpolate(
    #     [VectorValue(0.0, 0.0, 0.0), VectorValue(0.0, 0.0, 0.0), VectorValue(0.0, 0.0)], X
    # )
    run = get_solver()
    # Incremental loading: load goes from 0 → -2000 in 21 steps
    # Step 0: 0, Step 1: -100, ..., Step 20: -2000
    range = 0:2
    @timeit to "Starting nonlinear solver" begin
        @info "Starting nonlinear solver..."
        for step in range
            load_z = step * -100.0
            @timeit to "Load step $step with load $load_z" begin
                wh, cache = run(wh, load_z, step + 1, length(range), cache)
            end
        end
    end
finally
    show(to)
end