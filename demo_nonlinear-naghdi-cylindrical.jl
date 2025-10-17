# Naghdi shell model in 2D - nonlinear version
# Based on https://fenics-shells.readthedocs.io/en/latest/demo/nonlinear-naghdi-cylindrical/demo_nonlinear-naghdi-cylindrical.py.html
# Author: Arved H.
# Date: October 2025

using Gridap
using Gridap.Geometry, Gridap.ReferenceFEs, Gridap.Arrays

ρ = 1.016 #radius of the cylinder
L = 3.048 #length of the cylinder
E = 2.0685e7
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = 2.0 * μ * ν / (1.0 - 2.0 * ν)   #1st Lame parameter
λ̄ = (2 * λ * μ) / (λ + 2 * μ)   #Plane stress parameter effective Lame parameter

t = 0.03 #thickness of the shell

domain2D = (-π / 2, π / 2, 0.0, L)
partition = (20, 20)

model = CartesianDiscreteModel(domain2D, partition)

labels = get_face_labeling(model)
topo = get_grid_topology(model)

add_tag_from_tags!(labels, "leftright_boundary", [7, 8])
add_tag_from_tags!(labels, "up_boundary", [5])  # TOP boundary y=0

order = 2
#TODO not clear how I could have enriched the space with bubble functions here
reffe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, order + 1)
V0_u = TestFESpace(
    model,
    reffe_u;
    conformity=:H1,
    dirichlet_tags=["leftright_boundary", "up_boundary"],
    dirichlet_masks=[(false, false, true), (true, true, true)],
)
U_u = TrialFESpace(V0_u, VectorValue(0.0, 0.0, 0.0))

reffe_β = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V0_β = TestFESpace(
    model,
    reffe_β;
    conformity=:H1,
    dirichlet_tags=["leftright_boundary"],
    dirichlet_masks=[(true, false)],
)
U_β = TrialFESpace(V0_β, VectorValue(0.0, 0.0))

X = MultiFieldFESpace([U_u, U_β])
Y = MultiFieldFESpace([V0_u, V0_β])

degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
dΩ_h = Measure(Ω, degree - 2)
#Get characteristic cell size h 
h = CellField(get_cell_measure(Ω) .^ (1 / num_cell_dims(Ω)), Ω)

#function for half_cylinder
function funΦ(x2D)
    phi, y = x2D
    x = ρ * sin(phi)
    z = ρ * cos(phi)
    return VectorValue(x, y, z)  # Convert to Cartesian coordinates
end

Φ₀ = interpolate_everywhere(funΦ, V0_u)

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

function normal_vector(T::Tuple{VectorValue{3,Float64},VectorValue{3,Float64}})
    return (T[1] × T[2]) / (norm(T[1] × T[2]))
end

function normal_vector(Φ::Gridap.CellField)
    T₀ = tangent_vectors(Φ₀)
    return normal_vector ∘ (T₀)
end

N₀ = normal_vector(Φ₀)

function get_β₀(n::VectorValue{3,<:Number})
    return VectorValue(atan(-n[2], sqrt(n[1]^2 + n[3]^2)), atan(n[1], n[3]))
end

β₀ = get_β₀ ∘ (N₀)

function director(β::VectorValue{2,T}) where {T<:Number}
    return VectorValue(cos(β[1]) * sin(β[2]), -sin(β[1]), cos(β[1]) * cos(β[2]))
end

function d(β)
    return director ∘ (β + β₀)
end

function ∇d(β)
    function J(β::VectorValue{2,T}) where {T<:Number}
        return TensorValue(
            [
                -sin(β[1])*cos(β[2]) -sin(β[1])*sin(β[2]) -cos(β[1])
                -cos(β[1])*sin(β[2]) cos(β[1])*cos(β[2]) zero(T)
            ],
        )
    end
    # chain Rule switched because Jacobian is defined as Jtransposed in gridap.jl
    return ∇(β) ⋅ (J ∘ (β + β₀))
end

x = interpolate_everywhere(β₀, V0_β)
y = interpolate_everywhere(d(β₀), V0_u)

# is equivalent with N₀
D₀ = director ∘ (β₀)

# test equivalence
e = D₀ - N₀
@assert sqrt(sum(∫(e ⋅ e) * dΩ)) < 1e-10 "The initial director field does not match the normal field"
d₀ = interpolate_everywhere(D₀, V0_u)

F(u) = ∇(u) + ∇(Φ₀)
a₀ = ∇(Φ₀) ⋅ ∇(Φ₀)'
a₀⁻¹ = inv(a₀)
b₀ = -0.5 * ((∇(Φ₀) ⋅ ∇(d₀)') + (∇(d₀) ⋅ ∇(Φ₀)'))
j₀ = sqrt ∘ det(a₀)

ε_nl(u) = 0.5 * (F(u) ⋅ F(u)' - a₀)    # 2x2 matrix membrane strain
κ(u, β) = -0.5 * (F(u) ⋅ ∇d(β)' + ∇d(β) ⋅ F(u)') - b₀ # 2x2 matrix bending strain
γ(u, β) = F(u) ⋅ d(β) - ∇(Φ₀) ⋅ d₀ # 2x1 vector shear strain

function getĀ(a₀⁻¹)
    A1111 = (λ̄ + 2 * μ) * a₀⁻¹[1, 1]^2
    A1112 = (λ̄ + 2 * μ) * a₀⁻¹[1, 1] * a₀⁻¹[1, 2]
    A1122 = (λ̄ + μ) * a₀⁻¹[1, 1] * a₀⁻¹[2, 2] + μ * a₀⁻¹[1, 2]^2
    A1211 = A1112
    A1212 = μ * a₀⁻¹[1, 1] * a₀⁻¹[2, 2] + (λ̄ + μ) * a₀⁻¹[1, 2]^2
    A1222 = (λ̄ + 2 * μ) * a₀⁻¹[1, 2] * a₀⁻¹[2, 2]
    A2211 = A1122
    A2212 = A1222
    A2222 = (λ̄ + 2 * μ) * a₀⁻¹[2, 2]^2
    return Gridap.TensorValues.SymFourthOrderTensorValue(
        A1111, A1112, A1122, A1211, A1212, A1222, A2211, A2212, A2222
    )
end

Ā = getĀ ∘ (a₀⁻¹)

N(ε) = t * Ā ⊙ ε    #   Normal Stress SymTensorValue 2x2
M(κ) = (t^3 / 12) * Ā ⊙ κ  #   Bending Moment SymTensorValue 2x2
T(γ) = t * μ * a₀⁻¹ ⋅ γ #   Shear Force TensorValue 2x1

# psi_m(u) = 0.5 * (N(ε_nl(u)) ⊙ ε_nl(u))
# psi_b(u, β) = 0.5 * (M(κ(u, β)) ⊙ κ(u, β))
# psi_s(u, β) = 0.5 * (T(γ(u, β)) ⊙ γ(u, β))

#tag cell for load

pointLoadLocation = Point(0.0, L)
cells = vec(get_cell_coordinates(Ω))
cellId = argmin(norm.(mean.(cells) .- pointLoadLocation))
cell_to_tag = fill(1, num_cells(Ω))
cell_to_tag[cellId] = 2
additional_labels = Gridap.Geometry.face_labeling_from_cell_tags(
    topo, cell_to_tag, ["all", "load_area"]
)
merge!(labels, additional_labels)
writevtk(model, "model2D"; labels=labels)

Γ_lp = BoundaryTriangulation(model; tags=["load_area"])
dΓ_lp = Measure(Γ_lp, degree)

# a force shall applied
f_lp(x) = VectorValue(0.0, 1.0, 0.0)

#Weighing factor alpha for curing locking
α = (t ./ h) .* (t ./ h)

#internal residual
function R_internal((u, β), (du, dβ))
    return ∫(
        (α * N(ε_nl(u)) ⊙ ε_nl(du) + M(κ(u, β)) ⊙ κ(du, dβ) + α * T(γ(u, β)) ⋅ γ(du, dβ)) *
        j₀,
    ) * dΩ +
           ∫(((1 - α) * N(ε_nl(u)) ⊙ ε_nl(du) + (1 - α) * T(γ(u, β)) ⋅ γ(du, dβ)) * j₀) *
           dΩ_h
end

#test
isa(sum(R_internal((y, x), (y, x))), Number)

#external residual
function R_external((du, dβ))
    return ∫(VectorValue(0.0, 0.0, 0.0) ⋅ du + VectorValue(0.0, 0.0) ⋅ dβ) * dΩ +
           ∫(f_lp ⋅ du) * dΓ_lp
end

#total residual
function R_total((u, β), (du, dβ))
    return R_internal((u, β), (du, dβ)) - R_external((du, dβ))
end

#test
isa(sum(R_total((y, x), (y, x))), Number)

#initial guess
wh = interpolate_everywhere([VectorValue(0.0, 0.0, 0.0), VectorValue(0.0, 0.0)], X)

op = FEOperator(R_total, X, Y)

## test jacobian
res, jac = Gridap.Algebra.residual_and_jacobian(op, wh)
using LinearAlgebra
cond(collect(jac)) > 10e10 && @warn "Jacobian is ill-conditioned"

## solve
using LineSearches: BackTracking
nls = NLSolver(; show_trace=true, method=:newton, linesearch=BackTracking(), iterations=20)
solver = FESolver(nls)

wh_new, _ = solve!(wh, solver, op)
(uh, βh) = wh_new

writevtk(Ω, "test"; cellfields=Dict("Φ₀" => Φ₀, "u" => uh, "β" => βh))

## visualization in 3D
function map_to_3D(fun, model)
    model = UnstructuredDiscreteModel(model)
    grid = get_grid(model)
    node_coordinates = Gridap.ReferenceFEs.get_node_coordinates(grid) # Physical coordinates of nodes
    cell_to_nodes = get_cell_node_ids(grid) # Node IDs for each cell
    cell_reffes = get_cell_reffe(grid)     # Get reference elements for each cell

    new_node_coordinates = map(fun, node_coordinates)
    reffes, cell_types = compress_cell_data(cell_reffes)
    new_grid = UnstructuredGrid(new_node_coordinates, cell_to_nodes, reffes, cell_types)
    return UnstructuredDiscreteModel(new_grid, UnstructuredGridTopology(new_grid), labels)
end

newModel = map_to_3D(Φ₀, model)
vtkPath = "data/half_cylinder_naghdi"
mkpath(dirname(vtkPath))
writevtk(newModel, basename(vtkPath))