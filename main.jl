# Naghdi shell model in 2D - nonlinear version
# Based on https://fenics-shells.readthedocs.io/en/latest/demo/nonlinear-naghdi-cylindrical/demo_nonlinear-naghdi-cylindrical.py.html

using Gridap
using Gridap.CellData, Gridap.Geometry

R = 1.0 #Radius of the cylinder

E = 2.0685e7
ρ = 1.016
ν = 0.3
μ = E / (2.0 * (1.0 + ν))
λ = 2.0 * μ * ν / (1.0 - 2.0 * ν)   #1st Lame parameter
λ̄ = (2 * λ * μ) / (λ + 2 * μ)   #Plane stress parameter effective Lame parameter

t = 0.03 #Thickness of the shell

domain2D = (0.0, π, 0.0, 1.0)
partition = (20, 10)

# model = CartesianDiscreteModel(domain2D, partition; isperiodic=(true, false))
model = CartesianDiscreteModel(domain2D, partition)

# model = CartesianDiscreteModel(domain2D, partition)

labels = get_face_labeling(model)
topo = get_grid_topology(model)
#boundary for left side and right side
add_tag_from_tags!(labels, "Dirichlet-in", [5])
add_tag_from_tags!(labels, "Dirichlet-out", [6])
writevtk(model, "model2D")

order = 2
reffe_u = ReferenceFE(lagrangian, VectorValue{3,Float64}, order)
V0_u = TestFESpace(model, reffe_u; conformity=:H1, dirichlet_tags=["boundary"])
U_u = TrialFESpace(V0_u, VectorValue(0.0, 0.0, 0.0))

reffe_β = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)
V0_β = TestFESpace(model, reffe_β; conformity=:H1, dirichlet_tags=["Dirichlet-in"])
U_β = TrialFESpace(V0_β, VectorValue(0.0, 0.0))

X = MultiFieldFESpace([U_u, U_β])
Y = MultiFieldFESpace([V0_u, V0_β])

#Numerical integration
#higher degree of numerical integration leads to errors at the boundary due to discontinuity and large errors!
degree = 2 * order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

Xq = get_cell_points(Ω)

x = [0.0, 0.0]
function rho(θ, R)
    return R  # Constant radius
end

#function for circle
function funΦ(data)
    phi, z = data
    x = rho(phi, R) * -cos(phi)   # Map x-coordinate to angle [0,π]
    y = rho(phi, R) * sin(phi)   # Keep y-coordinate as height
    return VectorValue(x, y, z)  # Convert to Cartesian coordinates
end

Φ₀ = interpolate_everywhere(funΦ, V0_u)

#Second Section

px = get_physical_coordinate(Ω)

# function n(x)
#     return VectorValue(cos(x[1]), sin(x[1]), 0.0)
# end

function tangent_vectors(Φ::Gridap.FESpaces.SingleFieldFEFunction)
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

function normal_vector(Φ::Gridap.FESpaces.SingleFieldFEFunction)
    T₀ = tangent_vectors(Φ₀)
    return normal_vector ∘ (T₀)
end

T₀ = tangent_vectors(Φ₀)

N₀ = normal_vector(Φ₀)

function get_β₀(n::VectorValue{3,<:Number})
    return VectorValue(atan(-n[3], sqrt(n[1]^2 + n[2]^2)), atan(n[2], n[1]))
end
β₀ = get_β₀ ∘ (N₀)

function director(β::VectorValue{2,T}) where {T<:Number}
    return VectorValue(cos(β[1]) * cos(β[2]), cos(β[1]) * sin(β[2]), -sin(β[1]))
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
d₀ = director ∘ (β₀)

e = d₀ - N₀
@assert sqrt(sum(∫(e ⋅ e) * dΩ)) < 1e-10 "The initial director field does not match the normal field"

# F(∇u) = ∇u - ∇(Φ₀)
F(∇u) = ∇u

n₀ = interpolate_everywhere(N₀, V0_u)
a₀ = ∇(Φ₀) ⋅ ∇(Φ₀)'
a₀⁻¹ = inv(a₀)
b₀ = -0.5 * ((∇(Φ₀) ⋅ ∇(n₀)') + (∇(n₀) ⋅ ∇(Φ₀)'))
j₀ = sqrt ∘ det(a₀)

function ε_nl(∇u)
    return 0.5 * (F(∇u) ⋅ F(∇u)' - a₀)    # 2x2 matrix membrane strain
end
κ(u, β) = 0.5 * (F(u) ⋅ ∇d(β)' + ∇d(β) ⋅ F(u)') - b₀ # 2x2 matrix bending strain
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
A = Ā(Xq)

# N(ε) = t * Ā ⊙ ε    #   Normal Stress SymTensorValue 2x2
N(ε) = t * ε    #   Normal Stress SymTensorValue 2x2

M(κ) = (t^3 / 12) * Ā ⊙ κ  #   Bending Moment SymTensorValue 2x2
T(γ) = t * μ * a₀⁻¹ ⋅ γ #   Shear Force TensorValue 2x1

psi_m(u) = 0.5 * (N(ε_nl(u)) ⊙ ε_nl(u))
psi_b(u, β) = 0.5 * (M(κ(u, β)) ⊙ κ(u, β))
psi_s(u, β) = 0.5 * (T(γ(u, β)) ⊙ γ(u, β))

#test
# @time (psi_m(b) + psi_b(b, a) + psi_s(b, a))(Xq)[1]

#Third Section

#add filter for point load
additional_labels = Gridap.Geometry.face_labeling_from_vertex_filter(
    topo, "load_area", (x) -> (0.8 < x[1] < 1.2) && (0.3 < x[2] < 0.7)
)
merge!(labels, additional_labels)

Γ_lp = BoundaryTriangulation(model; tags=["load_area"])
dΓ_lp = Measure(Γ_lp, degree)

# a force shall applied
f_lp(x) = VectorValue(0.0, 0.01, 0.0)

#internal residual
# function R_internal((u, β), (du, dβ))
#     return ∫((N(ε(u)) ⊙ ε(du) + M(κ(u, β)) ⊙ κ(du, dβ) + T(γ(u, β)) ⊙ γ(du, dβ)) * j₀) * dΩ
# end
function R_internal(u, du)
    return ∫(N(ε_nl ∘ (∇(u))) ⊙ ε_nl ∘ (∇(du))) * dΩ
end
#external residual
# function R_external((du, dβ))
#     return ∫(VectorValue(0.0, 0.0, 0.0) ⋅ du + VectorValue(0.0, 0.0) ⋅ dβ) * dΩ +
#            ∫(f_lp ⋅ du) * dΓ_lp
# end
function R_external(du)
    return ∫((VectorValue(0.0, 0.0, 0.0) ⋅ du) * j₀) * dΩ
end

#total residual
# function R_total((u, β), (du, dβ))
#     return R_internal((u, β), (du, dβ)) - R_external((du, dβ))
# end
function R_total(u, du)
    return R_internal(u, du) - R_external(du)
end

# wh = interpolate_everywhere([VectorValue(0.0, 1.0, 0.0), VectorValue(0.0, 0.0)], X)
# wh2 = interpolate_everywhere([VectorValue(0.0, 2.0, 0.0), VectorValue(0.0, 0.0)], X)
wh = interpolate_everywhere(VectorValue(0.0, 1.0, 0.0), U_u)
wh2 = interpolate_everywhere(VectorValue(1.0, 2.0, 1.0), U_u)

# op = FEOperator(R_total, X, Y)
op = FEOperator(R_internal, U_u, V0_u)

#test jacobian
A = Gridap.Algebra.residual_and_jacobian(op, wh)
B = Gridap.Algebra.residual_and_jacobian(op, wh2)
maximum(A[1] - B[1])
maximum(A[2])
det(B[2])

## solve
using LineSearches: BackTracking
nls = NLSolver(; show_trace=true, method=:newton, linesearch=BackTracking(), iterations=20)
solver = FESolver(nls)

wh_new, _ = solve!(wh, solver, op)
(uh, βh) = wh_new

writevtk(Ω, "test"; cellfields=Dict("Φ₀" => Φ₀, "u" => uh, "β" => βh))
