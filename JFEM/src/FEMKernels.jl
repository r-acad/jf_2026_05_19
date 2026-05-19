
module FEM

using LinearAlgebra
using Statistics
using StaticArrays

# Tunable phi2 shear correction: alpha coefficient in phi2 = min(1, alpha*(h/L)^2)
# Set to 0.0 to use default alpha=10. Otherwise overrides alpha.
const PHI2_ALPHA = Ref(10.0)

# When true, allow the MITC4+ assumed-strain membrane formulation to apply even
# when curvature_membrane is supplied. Default false preserves the legacy
# "flat-only MITC4+" behavior; Ko-Lee-Bathe 2016 motivates enabling it on
# curved/distorted 4-node shells where standard MITC4's rs-term causes
# membrane locking (HTP_launch etc.). Env: JFEM_SOL105_EIG_MITC4PLUS_ALLOW_CURVED.
const MITC4PLUS_ALLOW_CURVED = Ref(
    lowercase(strip(get(ENV, "JFEM_SOL105_EIG_MITC4PLUS_ALLOW_CURVED", "false"))) in ("1", "true", "yes", "on")
)

@inline function fem_env_float(name::AbstractString, default::Float64)
    raw = get(ENV, name, "")
    isempty(strip(raw)) && return default
    parsed = tryparse(Float64, strip(raw))
    return parsed === nothing ? default : parsed
end

@inline function fem_env_bool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, "")))
    isempty(raw) && return default
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    return default
end

# Pre-allocated workspace for QUAD4 element stiffness computation.
# Allocate once per thread, reuse across all elements to eliminate ~5M heap allocations.
struct Quad4Workspace
    # Accumulated per-element (cleared at start of each element)
    Ke::Matrix{Float64}          # 24×24 element stiffness
    K_ab::Matrix{Float64}        # 24×4  membrane incompatible mode coupling
    K_bb::Matrix{Float64}        # 4×4   membrane incompatible self-coupling
    K_ab_bend::Matrix{Float64}   # 24×4  bending incompatible mode coupling
    K_bb_bend::Matrix{Float64}   # 4×4   bending incompatible self-coupling
    # MITC4 tying point B-matrices (filled per element)
    Bs_tp::Matrix{Float64}       # 4×24  rows: [Bs_xi_A; Bs_xi_C; Bs_eta_B; Bs_eta_D]
    Bs_row::Vector{Float64}      # 24    temporary for tying point computation
    # Gauss-point matrices (reused each GP, cleared with fill!)
    Bm::Matrix{Float64}          # 3×24  membrane strain-displacement
    Bb::Matrix{Float64}          # 3×24  bending strain-displacement
    Bd::Matrix{Float64}          # 1×24  drilling B-matrix
    Bi::Matrix{Float64}          # 3×4   membrane incompatible mode B-matrix
    Bi_bend::Matrix{Float64}     # 3×4   bending incompatible mode B-matrix
    Bs_cov::Matrix{Float64}      # 2×24  covariant shear B-matrix
    # Temporaries for in-place mul!
    tmp3x24::Matrix{Float64}     # for Cm*Bm, Cb*Bb products
    tmp3x4::Matrix{Float64}      # for Cm*Bi, Cb*Bi_bend products
    tmp2x24::Matrix{Float64}     # for Cs_cov*Bs_cov products
    tmp2x2::Matrix{Float64}      # for Cs_cov = invJ'*Cs*invJ
    # B matrix coupling workspace (reused when Bmb != nothing)
    K_ab_cross::Matrix{Float64}      # 24×4
    K_ab_bend_cross::Matrix{Float64} # 24×4
    K_mb_incomp::Matrix{Float64}     # 4×4
    # Coordinate transform workspace (used in Solver.jl assembly)
    Ke_global::Matrix{Float64}   # 24×24  T'*Ke*T result
    tmp24x24::Matrix{Float64}    # 24×24  temporary for transform
    Rel_t::Matrix{Float64}       # 3×3   element-to-global rotation
    # Thread-local constitutive matrix buffers (copied from flat arrays, avoids Vector{Matrix} reads)
    Cm_buf::Matrix{Float64}      # 3×3   membrane constitutive
    Cb_buf::Matrix{Float64}      # 3×3   bending constitutive
    Cs_buf::Matrix{Float64}      # 2×2   shear constitutive
    Bmb_buf::Matrix{Float64}     # 3×3   membrane-bending coupling
    # Assembly buffers (used in Solver.jl parallel loop)
    T_buf::Matrix{Float64}       # 24×24 transformation matrix
    lc::Matrix{Float64}          # 4×2   local coordinates
    dofs::Vector{Int}            # 24    DOF indices
end

function create_quad4_workspace()
    Quad4Workspace(
        zeros(24,24), zeros(24,4), zeros(4,4), zeros(24,4), zeros(4,4),   # Ke, K_ab, K_bb, K_ab_bend, K_bb_bend
        zeros(4,24), zeros(24),                                             # Bs_tp, Bs_row
        zeros(3,24), zeros(3,24), zeros(1,24), zeros(3,4), zeros(3,4), zeros(2,24), # Bm..Bs_cov
        zeros(3,24), zeros(3,4), zeros(2,24), zeros(2,2),                  # tmp buffers
        zeros(24,4), zeros(24,4), zeros(4,4),                              # B coupling
        zeros(24,24), zeros(24,24), zeros(3,3),                            # transform
        zeros(3,3), zeros(3,3), zeros(2,2), zeros(3,3),                    # constitutive buffers
        zeros(24,24), zeros(4,2), Vector{Int}(undef, 24)                   # assembly buffers
    )
end

# Thread-safe matrix multiplication replacing BLAS mul! (which is NOT re-entrant on Windows).
# C += alpha * A * B   (A is m×k, B is k×n, C is m×n)
@inline function ts_mul_add!(C, A, B, alpha)
    m, k = size(A)
    _, n = size(B)
    @inbounds @fastmath for j in 1:n
        for l in 1:k
            val = alpha * B[l,j]
            for i in 1:m
                C[i,j] += A[i,l] * val
            end
        end
    end
end

# C += alpha * A' * B   (A is k×m, A' is m×k, B is k×n, C is m×n)
@inline function ts_mul_At_add!(C, A, B, alpha)
    k, m = size(A)
    _, n = size(B)
    @inbounds @fastmath for j in 1:n
        for l in 1:k
            val = alpha * B[l,j]
            for i in 1:m
                C[i,j] += A[l,i] * val
            end
        end
    end
end

# C = A * B   (overwrite, no accumulate)
@inline function ts_mul!(C, A, B)
    m, k = size(A)
    _, n = size(B)
    fill!(C, 0.0)
    @inbounds @fastmath for j in 1:n
        for l in 1:k
            val = B[l,j]
            for i in 1:m
                C[i,j] += A[i,l] * val
            end
        end
    end
end

@inline function _is_infinite_like(x)
    return x == Inf || x == -Inf
end

function stiffness_frame3d_generic(L, A, Iy, Iz, J, E, G; As_y=Inf, As_z=Inf, I12=0.0)
    T = promote_type(typeof(L), typeof(A), typeof(Iy), typeof(Iz), typeof(J), typeof(E), typeof(G), typeof(I12))
    k = zeros(T, 12, 12)
    if L < 1e-9; return k; end

    X = E * A / L
    k[1,1] = X;  k[1,7] = -X; k[7,1] = -X; k[7,7] = X

    T = G * J / L
    k[4,4] = T;  k[4,10] = -T; k[10,4] = -T; k[10,10] = T

    # Timoshenko shear parameters (Φ=0 reduces to Euler-Bernoulli)
    # NASTRAN: K1=shear area factor for plane 1 (y-dir), K2=for plane 2 (z-dir)
    # As_y = K1*A (shear area in y), As_z = K2*A (shear area in z)
    # xz-plane bending: deflection in z → shear in z → uses As_z (K2*A)
    # xy-plane bending: deflection in y → shear in y → uses As_y (K1*A)
    Phi_y = _is_infinite_like(As_z) ? zero(T) : 12*E*Iy/(G*As_z*L^2)
    Phi_z = _is_infinite_like(As_y) ? zero(T) : 12*E*Iz/(G*As_y*L^2)

    # Bending in xz-plane (uses Iy, shear via As_z=K2*A)
    a_y = 12*E*Iy / (L^3*(1+Phi_y))
    b_y = 6*E*Iy / (L^2*(1+Phi_y))
    c_y = (4+Phi_y)*E*Iy / (L*(1+Phi_y))
    d_y = (2-Phi_y)*E*Iy / (L*(1+Phi_y))
    k[3,3] = a_y;  k[3,9] = -a_y; k[9,3] = -a_y; k[9,9] = a_y
    k[3,5] = -b_y; k[3,11] = -b_y; k[5,3] = -b_y; k[11,3] = -b_y
    k[9,5] = b_y;  k[9,11] = b_y;  k[5,9] = b_y;  k[11,9] = b_y
    k[5,5] = c_y;  k[5,11] = d_y;  k[11,5] = d_y;  k[11,11] = c_y

    # Bending in xy-plane (uses Iz, shear via As_y=K1*A)
    a_z = 12*E*Iz / (L^3*(1+Phi_z))
    b_z = 6*E*Iz / (L^2*(1+Phi_z))
    c_z = (4+Phi_z)*E*Iz / (L*(1+Phi_z))
    d_z = (2-Phi_z)*E*Iz / (L*(1+Phi_z))
    k[2,2] = a_z;  k[2,8] = -a_z; k[8,2] = -a_z; k[8,8] = a_z
    k[2,6] = b_z;  k[2,12] = b_z;  k[6,2] = b_z;  k[12,2] = b_z
    k[8,6] = -b_z; k[8,12] = -b_z; k[6,8] = -b_z; k[12,8] = -b_z
    k[6,6] = c_z;  k[6,12] = d_z;  k[12,6] = d_z;  k[12,12] = c_z

    # Cross-coupling from I12 (product of inertia)
    # Couples xy-plane bending {v: 2,8; θz: 6,12} with xz-plane bending {w: 3,9; θy: 5,11}
    if abs(I12) > 0.0
        a_yz = 12*E*I12 / L^3
        b_yz = 6*E*I12 / L^2
        c_yz = 4*E*I12 / L
        d_yz = 2*E*I12 / L
        # v-w coupling: DOFs (2,3), (2,9), (8,3), (8,9)
        k[2,3] += a_yz;  k[3,2] += a_yz
        k[2,9] += -a_yz; k[9,2] += -a_yz
        k[8,3] += -a_yz; k[3,8] += -a_yz
        k[8,9] += a_yz;  k[9,8] += a_yz
        # v-θy coupling: DOFs (2,5), (2,11), (8,5), (8,11)
        k[2,5] += -b_yz; k[5,2] += -b_yz
        k[2,11] += -b_yz; k[11,2] += -b_yz
        k[8,5] += b_yz;  k[5,8] += b_yz
        k[8,11] += b_yz; k[11,8] += b_yz
        # w-θz coupling: DOFs (3,6), (3,12), (9,6), (9,12)
        k[3,6] += b_yz;  k[6,3] += b_yz
        k[3,12] += b_yz; k[12,3] += b_yz
        k[9,6] += -b_yz; k[6,9] += -b_yz
        k[9,12] += -b_yz; k[12,9] += -b_yz
        # θy-θz coupling: DOFs (5,6), (5,12), (11,6), (11,12)
        k[5,6] += -c_yz; k[6,5] += -c_yz
        k[5,12] += -d_yz; k[12,5] += -d_yz
        k[11,6] += -d_yz; k[6,11] += -d_yz
        k[11,12] += -c_yz; k[12,11] += -c_yz
    end

    return k
end

function stiffness_frame3d(L, A, Iy, Iz, J, E, G; As_y=Inf, As_z=Inf, I12=0.0)
    return stiffness_frame3d_generic(L, A, Iy, Iz, J, E, G; As_y=As_y, As_z=As_z, I12=I12)
end

function forces_frame3d(u_elem, L, A, Iy, Iz, J, E, G; As_y=Inf, As_z=Inf, I12=0.0)
    k = stiffness_frame3d(L, A, Iy, Iz, J, E, G; As_y=As_y, As_z=As_z, I12=I12)
    f_local = k * u_elem
    
    return Dict(
        "axial"     => f_local[7],
        "shear_1"   => -f_local[2],
        "shear_2"   => -f_local[3],
        "torque"    => -f_local[4],
        "moment_a1" => -f_local[6],
        "moment_a2" => f_local[5],
        "moment_b1" => f_local[12],
        "moment_b2" => -f_local[11]
    )
end

function stress_frame3d(u_elem, L, A, E)
    return E * (u_elem[7] - u_elem[1]) / L
end



@inline function shape_derivs_quad(xi, eta)
    dN_dxi  = SVector{4}(-0.25*(1-eta), 0.25*(1-eta), 0.25*(1+eta), -0.25*(1+eta))
    dN_deta = SVector{4}(-0.25*(1-xi), -0.25*(1+xi), 0.25*(1+xi), 0.25*(1-xi))
    return dN_dxi, dN_deta
end

@inline function shape_values_quad(xi, eta)
    return SVector{4}(
        0.25*(1.0-xi)*(1.0-eta),
        0.25*(1.0+xi)*(1.0-eta),
        0.25*(1.0+xi)*(1.0+eta),
        0.25*(1.0-xi)*(1.0+eta),
    )
end

@inline function _mitc4_3d_node_vec(mat::AbstractMatrix, a::Int)
    return SVector{3,Float64}(mat[a,1], mat[a,2], mat[a,3])
end

@inline function _mitc4_3d_jacobian(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
    zeta::Float64,
)
    N = shape_values_quad(xi, eta)
    dNr, dNs = shape_derivs_quad(xi, eta)
    half_h = 0.5*h
    g_r = SVector(0.0, 0.0, 0.0)
    g_s = SVector(0.0, 0.0, 0.0)
    g_z = SVector(0.0, 0.0, 0.0)
    @inbounds for a in 1:4
        x_a = _mitc4_3d_node_vec(coords3d, a)
        n_a = _mitc4_3d_node_vec(directors, a)
        fiber_a = x_a + (zeta*half_h)*n_a
        g_r += dNr[a] * fiber_a
        g_s += dNs[a] * fiber_a
        g_z += (N[a]*half_h) * n_a
    end
    detJ = dot(g_r, cross(g_s, g_z))
    return N, dNr, dNs, g_r, g_s, g_z, detJ
end

@inline function _mitc4_3d_contravariant(
    g_r::SVector{3,Float64},
    g_s::SVector{3,Float64},
    g_z::SVector{3,Float64},
    detJ::Float64,
)
    det_safe = abs(detJ) < 1e-14 ? (detJ < 0.0 ? -1e-14 : 1e-14) : detJ
    gr = cross(g_s, g_z) / det_safe
    gs = cross(g_z, g_r) / det_safe
    gz = cross(g_r, g_s) / det_safe
    return gr, gs, gz, det_safe
end

@inline function _mitc4_3d_grad_from_natural(
    u_r::SVector{3,Float64},
    u_s::SVector{3,Float64},
    u_z::SVector{3,Float64},
    gr::SVector{3,Float64},
    gs::SVector{3,Float64},
    gz::SVector{3,Float64},
)
    return SMatrix{3,3,Float64,9}(
        u_r[1]*gr[1] + u_s[1]*gs[1] + u_z[1]*gz[1],
        u_r[2]*gr[1] + u_s[2]*gs[1] + u_z[2]*gz[1],
        u_r[3]*gr[1] + u_s[3]*gs[1] + u_z[3]*gz[1],
        u_r[1]*gr[2] + u_s[1]*gs[2] + u_z[1]*gz[2],
        u_r[2]*gr[2] + u_s[2]*gs[2] + u_z[2]*gz[2],
        u_r[3]*gr[2] + u_s[3]*gs[2] + u_z[3]*gz[2],
        u_r[1]*gr[3] + u_s[1]*gs[3] + u_z[1]*gz[3],
        u_r[2]*gr[3] + u_s[2]*gs[3] + u_z[2]*gz[3],
        u_r[3]*gr[3] + u_s[3]*gs[3] + u_z[3]*gz[3],
    )
end

"""
    quad4_mitc4_3d_B_displacement(coords3d, directors, h, xi, eta, zeta)

Build the displacement-based 3D small-strain B matrix for the opt-in
degenerate-3D MITC4 development path. Inputs are already in one Cartesian frame
(normally the element-local frame). The returned rows are
`[eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz]`.

This helper is intentionally not wired into default assembly yet; it is the
Phase-1 flat-reduction scaffold for `JFEM_Q4_KERNEL=mitc4_3d`.
"""
function quad4_mitc4_3d_B_displacement(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
    zeta::Float64,
)
    B = zeros(6, 24)
    N, dNr, dNs, g_r, g_s, g_z, detJ = _mitc4_3d_jacobian(coords3d, directors, h, xi, eta, zeta)
    gr, gs, gz, det_safe = _mitc4_3d_contravariant(g_r, g_s, g_z, detJ)
    half_h_zeta = 0.5*h*zeta
    half_h = 0.5*h
    e = (SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0))

    @inbounds for a in 1:4
        n_a = _mitc4_3d_node_vec(directors, a)
        base = (a-1)*6
        for d in 1:3
            ed = e[d]
            grad_u = _mitc4_3d_grad_from_natural(dNr[a]*ed, dNs[a]*ed, SVector(0.0,0.0,0.0), gr, gs, gz)
            col = base + d
            B[1,col] = grad_u[1,1]
            B[2,col] = grad_u[2,2]
            B[3,col] = grad_u[3,3]
            B[4,col] = grad_u[1,2] + grad_u[2,1]
            B[5,col] = grad_u[2,3] + grad_u[3,2]
            B[6,col] = grad_u[1,3] + grad_u[3,1]
        end
        for d in 1:3
            rvec = cross(e[d], n_a)
            grad_u = _mitc4_3d_grad_from_natural(
                (dNr[a]*half_h_zeta)*rvec,
                (dNs[a]*half_h_zeta)*rvec,
                (N[a]*half_h)*rvec,
                gr, gs, gz,
            )
            col = base + 3 + d
            B[1,col] = grad_u[1,1]
            B[2,col] = grad_u[2,2]
            B[3,col] = grad_u[3,3]
            B[4,col] = grad_u[1,2] + grad_u[2,1]
            B[5,col] = grad_u[2,3] + grad_u[3,2]
            B[6,col] = grad_u[1,3] + grad_u[3,1]
        end
    end
    return B, det_safe
end

@inline function _mitc4_3d_shear_row_at!(
    row::AbstractVector,
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
    zeta::Float64,
    component::Int,
)
    fill!(row, 0.0)
    N, dNr, dNs, g_r, g_s, g_z, detJ = _mitc4_3d_jacobian(coords3d, directors, h, xi, eta, zeta)
    half_h_zeta = 0.5*h*zeta
    half_h = 0.5*h
    e = (SVector(1.0,0.0,0.0), SVector(0.0,1.0,0.0), SVector(0.0,0.0,1.0))
    @inbounds for a in 1:4
        n_a = _mitc4_3d_node_vec(directors, a)
        base = (a-1)*6
        for d in 1:3
            ed = e[d]
            val = component == 1 ? dot(dNr[a]*ed, g_z) : dot(dNs[a]*ed, g_z)
            row[base+d] = val
        end
        for d in 1:3
            rvec = cross(e[d], n_a)
            u_nat = component == 1 ? (dNr[a]*half_h_zeta)*rvec : (dNs[a]*half_h_zeta)*rvec
            u_zeta = (N[a]*half_h)*rvec
            val = component == 1 ? dot(u_nat, g_z) + dot(u_zeta, g_r) :
                                   dot(u_nat, g_z) + dot(u_zeta, g_s)
            row[base+3+d] = val
        end
    end
    return row
end

"""
    quad4_mitc4_3d_shear_tying_rows(coords3d, directors, h, xi, eta; zeta=0.0)

Return the 2x24 MITC4 covariant transverse-shear tying rows at `(xi, eta)`.
Rows are `[gamma_rzeta, gamma_szeta]` before Cartesian shear transformation.
"""
function quad4_mitc4_3d_shear_tying_rows(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64;
    zeta::Float64=0.0,
)
    Bs_tp = zeros(4, 24)
    row = zeros(24)
    tying_pts = (SVector(0.0, -1.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(1.0, 0.0))
    _mitc4_3d_shear_row_at!(row, coords3d, directors, h, tying_pts[1][1], tying_pts[1][2], zeta, 1)
    @views copyto!(Bs_tp[1, :], row)
    _mitc4_3d_shear_row_at!(row, coords3d, directors, h, tying_pts[2][1], tying_pts[2][2], zeta, 1)
    @views copyto!(Bs_tp[2, :], row)
    _mitc4_3d_shear_row_at!(row, coords3d, directors, h, tying_pts[3][1], tying_pts[3][2], zeta, 2)
    @views copyto!(Bs_tp[3, :], row)
    _mitc4_3d_shear_row_at!(row, coords3d, directors, h, tying_pts[4][1], tying_pts[4][2], zeta, 2)
    @views copyto!(Bs_tp[4, :], row)

    Bs = zeros(2, 24)
    w_eta_m = 0.5*(1.0-eta)
    w_eta_p = 0.5*(1.0+eta)
    w_xi_m = 0.5*(1.0-xi)
    w_xi_p = 0.5*(1.0+xi)
    @inbounds for j in 1:24
        Bs[1,j] = w_eta_m*Bs_tp[1,j] + w_eta_p*Bs_tp[2,j]
        Bs[2,j] = w_xi_m*Bs_tp[3,j] + w_xi_p*Bs_tp[4,j]
    end
    return Bs
end

function quad4_mitc4_3d_physical_shear_rows(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
)
    tying_pts = (SVector(0.0, -1.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(1.0, 0.0))
    Bs_tp = zeros(4, 24)
    B, _ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, tying_pts[1][1], tying_pts[1][2], 0.0)
    @views copyto!(Bs_tp[1, :], B[6, :]) # gamma_xz at A
    B, _ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, tying_pts[2][1], tying_pts[2][2], 0.0)
    @views copyto!(Bs_tp[2, :], B[6, :]) # gamma_xz at B
    B, _ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, tying_pts[3][1], tying_pts[3][2], 0.0)
    @views copyto!(Bs_tp[3, :], B[5, :]) # gamma_yz at C
    B, _ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, tying_pts[4][1], tying_pts[4][2], 0.0)
    @views copyto!(Bs_tp[4, :], B[5, :]) # gamma_yz at D

    Bs = zeros(2, 24)
    w_eta_m = 0.5*(1.0-eta)
    w_eta_p = 0.5*(1.0+eta)
    w_xi_m = 0.5*(1.0-xi)
    w_xi_p = 0.5*(1.0+xi)
    @inbounds for j in 1:24
        Bs[1,j] = w_eta_m*Bs_tp[1,j] + w_eta_p*Bs_tp[2,j]
        Bs[2,j] = w_xi_m*Bs_tp[3,j] + w_xi_p*Bs_tp[4,j]
    end
    return Bs
end

function quad4_mitc4_3d_covariant_physical_shear_rows(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
)
    Bs_cov = quad4_mitc4_3d_shear_tying_rows(coords3d, directors, h, xi, eta)
    _, _, _, _, ar, as = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
    t1, t2 = _mitc4_3d_tangent_basis(coords3d, directors, h, xi, eta)
    c1r = dot(ar, t1)
    c1s = dot(as, t1)
    c2r = dot(ar, t2)
    c2s = dot(as, t2)
    scale = 2.0 / max(h, 1e-30)
    Bs = zeros(2, 24)
    @inbounds for j in 1:24
        Bs[1,j] = scale * (c1r*Bs_cov[1,j] + c1s*Bs_cov[2,j])
        Bs[2,j] = scale * (c2r*Bs_cov[1,j] + c2s*Bs_cov[2,j])
    end
    return Bs
end

function quad4_mitc4_3d_covariant_flatdirector_shear_rows(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
)
    _, _, _, g_r, g_s, _, _ = _mitc4_3d_jacobian(coords3d, directors, h, 0.0, 0.0, 0.0)
    n_raw = cross(g_r, g_s)
    n_len = norm(n_raw)
    n = n_len > 1e-30 ? n_raw / n_len : SVector(0.0, 0.0, 1.0)
    flat_dirs = zeros(4, 3)
    @inbounds for a in 1:4
        flat_dirs[a, 1] = n[1]
        flat_dirs[a, 2] = n[2]
        flat_dirs[a, 3] = n[3]
    end
    return quad4_mitc4_3d_covariant_physical_shear_rows(coords3d, flat_dirs, h, xi, eta)
end

@inline function quad4_mitc4_3d_selected_shear_rows(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
)
    mode = lowercase(strip(get(ENV, "JFEM_Q4_MITC4_3D_SHEAR_MODE", "covariant")))
    if mode in ("physical", "cartesian", "direct")
        return quad4_mitc4_3d_physical_shear_rows(coords3d, directors, h, xi, eta)
    elseif mode in ("flatdirector", "flat-director", "midnormal", "constant_director")
        return quad4_mitc4_3d_covariant_flatdirector_shear_rows(coords3d, directors, h, xi, eta)
    end
    return quad4_mitc4_3d_covariant_physical_shear_rows(coords3d, directors, h, xi, eta)
end

@inline function _mitc4_3d_surface_area_and_grads(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
)
    N, dNr, dNs, g_r, g_s, _, _ = _mitc4_3d_jacobian(coords3d, directors, h, xi, eta, 0.0)
    g11 = dot(g_r, g_r)
    g12 = dot(g_r, g_s)
    g22 = dot(g_s, g_s)
    detg = g11*g22 - g12*g12
    detg_safe = abs(detg) < 1e-14 ? (detg < 0.0 ? -1e-14 : 1e-14) : detg
    ar = (g22/detg_safe)*g_r + (-g12/detg_safe)*g_s
    as = (-g12/detg_safe)*g_r + (g11/detg_safe)*g_s
    dA = norm(cross(g_r, g_s))
    return N, dNr, dNs, dA, ar, as
end

@inline function _mitc4_3d_strain_row_project(
    B::AbstractMatrix,
    a::SVector{3,Float64},
    b::SVector{3,Float64},
    j::Int,
)
    return a[1]*b[1]*B[1,j] +
           a[2]*b[2]*B[2,j] +
           a[3]*b[3]*B[3,j] +
           0.5*(a[1]*b[2] + a[2]*b[1]) * B[4,j] +
           0.5*(a[2]*b[3] + a[3]*b[2]) * B[5,j] +
           0.5*(a[1]*b[3] + a[3]*b[1]) * B[6,j]
end

@inline function _mitc4_3d_tangent_basis(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
)
    _, _, _, g_r, g_s, _, _ = _mitc4_3d_jacobian(coords3d, directors, h, xi, eta, 0.0)
    n_raw = cross(g_r, g_s)
    n_len = norm(n_raw)
    n = n_len > 1e-30 ? n_raw / n_len : SVector(0.0, 0.0, 1.0)
    e1 = SVector(1.0, 0.0, 0.0)
    e2 = SVector(0.0, 1.0, 0.0)
    t1_raw = e1 - dot(e1, n) * n
    if norm(t1_raw) <= 1e-12
        t1_raw = e2 - dot(e2, n) * n
    end
    t1 = norm(t1_raw) > 1e-30 ? t1_raw / norm(t1_raw) : e1
    t2_raw = cross(n, t1)
    t2 = norm(t2_raw) > 1e-30 ? t2_raw / norm(t2_raw) : e2
    return t1, t2
end

function quad4_mitc4_3d_membrane_bending_rows(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    h::Float64,
    xi::Float64,
    eta::Float64,
    bend_delta::Float64,
)
    B0, _ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, xi, eta, 0.0)
    Bp, _ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, xi, eta, bend_delta)
    Bm_z, _ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, xi, eta, -bend_delta)
    Bm = zeros(3, 24)
    Bb = zeros(3, 24)
    mode = lowercase(strip(get(ENV, "JFEM_Q4_MITC4_3D_STRAIN_BASIS", "tangent")))
    if mode in ("cartesian", "fixed", "element")
        @inbounds for j in 1:24
            Bm[1,j] = B0[1,j]
            Bm[2,j] = B0[2,j]
            Bm[3,j] = B0[4,j]
            Bb[1,j] = (Bp[1,j] - Bm_z[1,j]) / (h * bend_delta)
            Bb[2,j] = (Bp[2,j] - Bm_z[2,j]) / (h * bend_delta)
            Bb[3,j] = (Bp[4,j] - Bm_z[4,j]) / (h * bend_delta)
        end
    else
        t1, t2 = _mitc4_3d_tangent_basis(coords3d, directors, h, xi, eta)
        @inbounds for j in 1:24
            Bm[1,j] = _mitc4_3d_strain_row_project(B0, t1, t1, j)
            Bm[2,j] = _mitc4_3d_strain_row_project(B0, t2, t2, j)
            Bm[3,j] = 2.0 * _mitc4_3d_strain_row_project(B0, t1, t2, j)
            Bb[1,j] = (_mitc4_3d_strain_row_project(Bp, t1, t1, j) -
                       _mitc4_3d_strain_row_project(Bm_z, t1, t1, j)) / (h * bend_delta)
            Bb[2,j] = (_mitc4_3d_strain_row_project(Bp, t2, t2, j) -
                       _mitc4_3d_strain_row_project(Bm_z, t2, t2, j)) / (h * bend_delta)
            Bb[3,j] = 2.0 * (_mitc4_3d_strain_row_project(Bp, t1, t2, j) -
                             _mitc4_3d_strain_row_project(Bm_z, t1, t2, j)) / (h * bend_delta)
        end
    end
    bmode = lowercase(strip(get(ENV, "JFEM_Q4_MITC4_3D_BENDING_MODE", "fiber")))
    if bmode in ("plate", "rotation", "rotgrad", "classical")
        _, dNr, dNs, _, ar, as = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
        fill!(Bb, 0.0)
        @inbounds for a in 1:4
            gradN = dNr[a] * ar + dNs[a] * as
            base = (a - 1) * 6
            dNdx = gradN[1]
            dNdy = gradN[2]
            Bb[1, base+5] = dNdx
            Bb[2, base+4] = -dNdy
            Bb[3, base+5] = dNdy
            Bb[3, base+4] = -dNdx
        end
    end
    return Bm, Bb
end

"""
    stiffness_quad4_mitc4_3d_resultant_matrices(...)

Opt-in development stiffness for `JFEM_Q4_KERNEL=mitc4_3d`. This is a
degenerate-3D kinematic shell using the new director-based B-matrix, integrated
against JFEM's existing shell resultants (`Cm/Cb/Cs/Bmb`). It is intentionally a
bridge implementation: enough to run probes and compare mode sets before adding
ply-by-ply through-thickness material callbacks.
"""
function stiffness_quad4_mitc4_3d_resultant_matrices(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    Cm,
    Cb,
    Cs,
    h,
    E_ref;
    k6rot=100.0,
    drill_scale::Float64=1.0,
    Bmb=nothing,
    shear_center_only::Bool=false,
    bending_incomp::Bool=false,
)
    Ke = zeros(24, 24)
    h < 1e-30 && return Ke
    membrane_scale = fem_env_float("JFEM_Q4_MITC4_3D_MEMBRANE_SCALE", 1.0)
    bending_scale = fem_env_float("JFEM_Q4_MITC4_3D_BENDING_SCALE", 1.0)
    shear_scale = fem_env_float("JFEM_Q4_MITC4_3D_SHEAR_SCALE", 1.0)
    drill_scale_diag = fem_env_float("JFEM_Q4_MITC4_3D_DRILL_SCALE", 1.0)
    bend_delta = min(max(abs(fem_env_float("JFEM_Q4_MITC4_3D_BENDING_ZETA_DELTA", 1e-4)), 1e-8), 1.0)
    shear_mode = lowercase(strip(get(ENV, "JFEM_Q4_MITC4_3D_SHEAR_MODE", "covariant")))
    macneal_shear = shear_mode in ("macneal", "rbf", "macneal_rbf", "plate_rbf")
    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt,-pt), SVector(pt,-pt), SVector(pt,pt), SVector(-pt,pt))

    tmp3x24 = zeros(3, 24)
    tmp2x24 = zeros(2, 24)
    tmp3x4 = zeros(3, 4)
    Bm = zeros(3, 24)
    Bb = zeros(3, 24)
    Bs = zeros(2, 24)
    Bd = zeros(1, 24)
    Bi_bend = zeros(3, 4)
    K_ab_bend = zeros(24, 4)
    K_bb_bend = zeros(4, 4)

    G_drill = h > 0.0 ? Cm[3,3] / h : 0.0
    if G_drill < 1e-6
        G_drill = E_ref / (2 * 3.0)
    end
    alpha_drill = drill_scale * (k6rot / 1e5) * G_drill * h

    @inbounds for gp in gauss_pts
        xi, eta = gp[1], gp[2]
        _, _, _, dA, ar, as = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
        dA = max(dA, 1e-12)
        dphi1 = (-2.0 * xi) * ar
        dphi2 = (-2.0 * eta) * as

        Bm_new, Bb_new = quad4_mitc4_3d_membrane_bending_rows(
            coords3d, directors, h, xi, eta, bend_delta
        )
        Bm .= Bm_new
        Bb .= Bb_new
        fill!(Bd, 0.0)
        fill!(Bi_bend, 0.0)

        if membrane_scale != 0.0
            ts_mul!(tmp3x24, Cm, Bm)
            ts_mul_At_add!(Ke, Bm, tmp3x24, dA * membrane_scale)
        end
        if bending_scale != 0.0
            ts_mul!(tmp3x24, Cb, Bb)
            ts_mul_At_add!(Ke, Bb, tmp3x24, dA * bending_scale)
        end
        if Bmb !== nothing
            ts_mul!(tmp3x24, Bmb, Bb)
            ts_mul_At_add!(Ke, Bm, tmp3x24, dA * sqrt(abs(membrane_scale * bending_scale)))
            ts_mul!(tmp3x24, Bmb, Bm)
            ts_mul_At_add!(Ke, Bb, tmp3x24, dA * sqrt(abs(membrane_scale * bending_scale)))
        end

        if !macneal_shear && !shear_center_only && shear_scale != 0.0 && maximum(abs, Cs) >= 1e-30
            Bs .= quad4_mitc4_3d_selected_shear_rows(coords3d, directors, h, xi, eta)
            ts_mul!(tmp2x24, Cs, Bs)
            ts_mul_At_add!(Ke, Bs, tmp2x24, dA * shear_scale)
        end

        if bending_incomp && bending_scale != 0.0
            Bi_bend[2,1] = -dphi1[2]
            Bi_bend[2,2] = -dphi2[2]
            Bi_bend[1,3] =  dphi1[1]
            Bi_bend[1,4] =  dphi2[1]
            Bi_bend[3,1] = -dphi1[1]
            Bi_bend[3,2] = -dphi2[1]
            Bi_bend[3,3] =  dphi1[2]
            Bi_bend[3,4] =  dphi2[2]
            ts_mul!(tmp3x4, Cb, Bi_bend)
            ts_mul_At_add!(K_ab_bend, Bb, tmp3x4, dA * bending_scale)
            ts_mul_At_add!(K_bb_bend, Bi_bend, tmp3x4, dA * bending_scale)
        end

        N, dNr, dNs, _, _, _ = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
        for a in 1:4
            gradN = dNr[a]*ar + dNs[a]*as
            base = (a-1)*6
            Bd[1, base+1] = 0.5*gradN[2]
            Bd[1, base+2] = -0.5*gradN[1]
            Bd[1, base+6] = N[a]
        end
        ts_mul_At_add!(Ke, Bd, Bd, dA * alpha_drill * drill_scale_diag)
    end

    if macneal_shear && shear_scale != 0.0 && maximum(abs, Cs) >= 1e-30
        coords2d = zeros(4, 2)
        K_shear = zeros(24, 24)
        @inbounds for a in 1:4
            coords2d[a, 1] = coords3d[a, 1]
            coords2d[a, 2] = coords3d[a, 2]
        end
        add_quad4_macneal_shear_rbf!(K_shear, coords2d, Cb, Cs, h)
        @inbounds @fastmath for j in 1:24, i in 1:24
            Ke[i, j] += shear_scale * K_shear[i, j]
        end
    elseif shear_center_only && shear_scale != 0.0 && maximum(abs, Cs) >= 1e-30
        _, _, _, dA, _, _ = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, 0.0, 0.0)
        Bs .= quad4_mitc4_3d_selected_shear_rows(coords3d, directors, h, 0.0, 0.0)
        ts_mul!(tmp2x24, Cs, Bs)
        ts_mul_At_add!(Ke, Bs, tmp2x24, 4.0 * max(dA, 1e-12) * shear_scale)
    end

    if bending_incomp && maximum(abs, K_bb_bend) > 0.0
        inv_Kbb_b = Matrix(inv(SMatrix{4,4}(K_bb_bend)))
        @inbounds @fastmath for j in 1:24, i in 1:24
            sb = 0.0
            for l in 1:4
                tmp_b = 0.0
                for q in 1:4
                    tmp_b += inv_Kbb_b[l,q] * K_ab_bend[j,q]
                end
                sb += K_ab_bend[i,l] * tmp_b
            end
            Ke[i,j] -= sb
        end
    end

    return Ke
end

function stiffness_quad4_mitc4_3d_ply_matrices(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    ply_data,
    Cs,
    h,
    E_ref;
    k6rot=100.0,
    drill_scale::Float64=1.0,
    shear_center_only::Bool=false,
    material_rotation::Float64=0.0,
    local_bending_scale::Float64=1.0,
)
    Ke = zeros(24, 24)
    h < 1e-30 && return Ke
    if ply_data === nothing || isempty(ply_data)
        return Ke
    end
    ply_scale = fem_env_float("JFEM_Q4_MITC4_3D_PLY_SCALE", 1.0)
    membrane_scale = fem_env_float("JFEM_Q4_MITC4_3D_MEMBRANE_SCALE", 1.0)
    bending_scale = fem_env_float("JFEM_Q4_MITC4_3D_BENDING_SCALE", 1.0) * local_bending_scale
    shear_scale = fem_env_float("JFEM_Q4_MITC4_3D_SHEAR_SCALE", 1.0)
    drill_scale_diag = fem_env_float("JFEM_Q4_MITC4_3D_DRILL_SCALE", 1.0)
    bend_delta = min(max(abs(fem_env_float("JFEM_Q4_MITC4_3D_BENDING_ZETA_DELTA", 1e-4)), 1e-8), 1.0)
    shear_mode = lowercase(strip(get(ENV, "JFEM_Q4_MITC4_3D_SHEAR_MODE", "covariant")))
    macneal_shear = shear_mode in ("macneal", "rbf", "macneal_rbf", "plate_rbf")
    ply_split = fem_env_bool("JFEM_Q4_MITC4_3D_PLY_SPLIT", false) ||
                membrane_scale != 1.0 || bending_scale != 1.0

    pt = 1.0 / sqrt(3.0)
    gp2 = (SVector(-pt, 1.0), SVector(pt, 1.0))
    gauss_pts = (SVector(-pt,-pt), SVector(pt,-pt), SVector(pt,pt), SVector(-pt,pt))

    tmp3x24 = zeros(3, 24)
    tmp2x24 = zeros(2, 24)
    Bps = zeros(3, 24)
    Bm = zeros(3, 24)
    Bb = zeros(3, 24)
    Bs = zeros(2, 24)
    Bd = zeros(1, 24)

    G_drill = E_ref / (2 * 3.0)
    alpha_drill = drill_scale * (k6rot / 1e5) * G_drill * h

    @inbounds for gp in gauss_pts
        xi, eta = gp[1], gp[2]
        if ply_split
            Bm_new, Bb_new = quad4_mitc4_3d_membrane_bending_rows(
                coords3d, directors, h, xi, eta, bend_delta
            )
            Bm .= Bm_new
            Bb .= Bb_new
        end
        _, _, _, dA_mid, _, _ = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
        for ply in ply_data
            z_bot = Float64(ply["z_bot"])
            z_top = Float64(ply["z_top"])
            z_mid = 0.5 * (z_bot + z_top)
            z_half = 0.5 * (z_top - z_bot)
            z_half <= 0.0 && continue
            Qbar = copy(ply["Qbar"])
            if abs(material_rotation) > 1e-12
                rotate_constitutive_3x3!(Qbar, material_rotation)
            end
            for gz in gp2
                z_phys = z_mid + z_half * gz[1]
                scale = max(dA_mid, 1e-12) * gz[2] * z_half * ply_scale
                fill!(Bps, 0.0)
                if ply_split
                    mfac = sqrt(max(membrane_scale, 0.0))
                    bfac = sqrt(max(bending_scale, 0.0))
                    for j in 1:24
                        Bps[1,j] = mfac * Bm[1,j] + bfac * z_phys * Bb[1,j]
                        Bps[2,j] = mfac * Bm[2,j] + bfac * z_phys * Bb[2,j]
                        Bps[3,j] = mfac * Bm[3,j] + bfac * z_phys * Bb[3,j]
                    end
                else
                    zeta = 2.0 * z_phys / h
                    B3, detJ = quad4_mitc4_3d_B_displacement(coords3d, directors, h, xi, eta, zeta)
                    scale = abs(detJ) * gz[2] * (2.0 * z_half / h) * ply_scale
                    mode = lowercase(strip(get(ENV, "JFEM_Q4_MITC4_3D_STRAIN_BASIS", "tangent")))
                    if mode in ("cartesian", "fixed", "element")
                        for j in 1:24
                            Bps[1,j] = B3[1,j]
                            Bps[2,j] = B3[2,j]
                            Bps[3,j] = B3[4,j]
                        end
                    else
                        t1, t2 = _mitc4_3d_tangent_basis(coords3d, directors, h, xi, eta)
                        for j in 1:24
                            Bps[1,j] = _mitc4_3d_strain_row_project(B3, t1, t1, j)
                            Bps[2,j] = _mitc4_3d_strain_row_project(B3, t2, t2, j)
                            Bps[3,j] = 2.0 * _mitc4_3d_strain_row_project(B3, t1, t2, j)
                        end
                    end
                end
                ts_mul!(tmp3x24, Qbar, Bps)
                ts_mul_At_add!(Ke, Bps, tmp3x24, scale)
            end
        end

        if !macneal_shear && !shear_center_only && shear_scale != 0.0 && maximum(abs, Cs) >= 1e-30
            _, _, _, dA, _, _ = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
            Bs .= quad4_mitc4_3d_selected_shear_rows(coords3d, directors, h, xi, eta)
            ts_mul!(tmp2x24, Cs, Bs)
            ts_mul_At_add!(Ke, Bs, tmp2x24, max(dA, 1e-12) * shear_scale)
        end

        N, dNr, dNs, _, ar, as = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
        fill!(Bd, 0.0)
        for a in 1:4
            gradN = dNr[a]*ar + dNs[a]*as
            base = (a-1)*6
            Bd[1, base+1] = 0.5*gradN[2]
            Bd[1, base+2] = -0.5*gradN[1]
            Bd[1, base+6] = N[a]
        end
        _, _, _, dA, _, _ = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
        ts_mul_At_add!(Ke, Bd, Bd, max(dA, 1e-12) * alpha_drill * drill_scale_diag)
    end

    if macneal_shear && shear_scale != 0.0 && maximum(abs, Cs) >= 1e-30
        coords2d = zeros(4, 2)
        Cb_eff = zeros(3, 3)
        K_shear = zeros(24, 24)
        @inbounds for a in 1:4
            coords2d[a, 1] = coords3d[a, 1]
            coords2d[a, 2] = coords3d[a, 2]
        end
        for ply in ply_data
            z_bot = Float64(ply["z_bot"])
            z_top = Float64(ply["z_top"])
            z_mid = 0.5 * (z_bot + z_top)
            z_half = 0.5 * (z_top - z_bot)
            z_half <= 0.0 && continue
            Qbar = copy(ply["Qbar"])
            if abs(material_rotation) > 1e-12
                rotate_constitutive_3x3!(Qbar, material_rotation)
            end
            @inbounds for j in 1:3, i in 1:3
                Cb_eff[i, j] += (z_top^3 - z_bot^3) / 3.0 * Qbar[i, j]
            end
        end
        add_quad4_macneal_shear_rbf!(K_shear, coords2d, Cb_eff, Cs, h)
        @inbounds @fastmath for j in 1:24, i in 1:24
            Ke[i, j] += shear_scale * K_shear[i, j]
        end
    elseif shear_center_only && shear_scale != 0.0 && maximum(abs, Cs) >= 1e-30
        _, _, _, dA, _, _ = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, 0.0, 0.0)
        Bs .= quad4_mitc4_3d_selected_shear_rows(coords3d, directors, h, 0.0, 0.0)
        ts_mul!(tmp2x24, Cs, Bs)
        ts_mul_At_add!(Ke, Bs, tmp2x24, 4.0 * max(dA, 1e-12) * shear_scale)
    end

    return Ke
end

function quad4_mitc4_3d_membrane_force_field(
    coords3d::AbstractMatrix,
    directors::AbstractMatrix,
    u_elem::AbstractVector,
    Cm,
    h;
    Bmb=nothing,
)
    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt,-pt), SVector(pt,-pt), SVector(pt,pt), SVector(-pt,pt))
    bend_delta = min(max(abs(fem_env_float("JFEM_Q4_MITC4_3D_BENDING_ZETA_DELTA", 1e-4)), 1e-8), 1.0)
    N_gp = zeros(4, 3)
    N_avg = zeros(3)
    area_w = zeros(4)
    Bm = zeros(3, 24)
    Bb = zeros(3, 24)
    eps_m = zeros(3)
    kappa = zeros(3)
    total_area = 0.0

    @inbounds for (igp, gp) in enumerate(gauss_pts)
        xi, eta = gp[1], gp[2]
        _, _, _, dA, _, _ = _mitc4_3d_surface_area_and_grads(coords3d, directors, h, xi, eta)
        dA = max(dA, 1e-12)

        Bm_new, Bb_new = quad4_mitc4_3d_membrane_bending_rows(
            coords3d, directors, h, xi, eta, bend_delta
        )
        Bm .= Bm_new
        Bb .= Bb_new
        fill!(eps_m, 0.0)
        fill!(kappa, 0.0)
        for j in 1:24
            eps_m[1] += Bm[1,j] * u_elem[j]
            eps_m[2] += Bm[2,j] * u_elem[j]
            eps_m[3] += Bm[3,j] * u_elem[j]
            if Bmb !== nothing
                kappa[1] += Bb[1,j] * u_elem[j]
                kappa[2] += Bb[2,j] * u_elem[j]
                kappa[3] += Bb[3,j] * u_elem[j]
            end
        end

        for i in 1:3
            val = 0.0
            for j in 1:3
                val += Cm[i,j] * eps_m[j]
                if Bmb !== nothing
                    val += Bmb[i,j] * kappa[j]
                end
            end
            N_gp[igp, i] = val
            N_avg[i] += val * dA
        end
        area_w[igp] = dA
        total_area += dA
    end

    if total_area > 1e-30
        @inbounds for i in 1:3
            N_avg[i] /= total_area
        end
    end
    return N_gp, N_avg, area_w
end

const QUAD4_D2N_DXIDETA = SVector{4}(0.25, -0.25, 0.25, -0.25)

@inline function quad4_center_frame_from_coords3d(coords_3d::AbstractMatrix)
    p1 = SVector(coords_3d[1,1], coords_3d[1,2], coords_3d[1,3])
    p2 = SVector(coords_3d[2,1], coords_3d[2,2], coords_3d[2,3])
    p3 = SVector(coords_3d[3,1], coords_3d[3,2], coords_3d[3,3])
    p4 = SVector(coords_3d[4,1], coords_3d[4,2], coords_3d[4,3])

    d13 = p3 - p1
    d24 = p4 - p2
    v3_raw = cross(d13, d24)
    v3_len = norm(v3_raw)
    if v3_len <= 1e-30
        v3_raw = cross(p2 - p1, p4 - p1)
        v3_len = norm(v3_raw)
    end
    v3 = v3_len > 1e-30 ? v3_raw / v3_len : SVector(0.0, 0.0, 1.0)

    edge1 = p2 - p1
    v1_raw = edge1 - dot(edge1, v3) * v3
    v1_len = norm(v1_raw)
    if v1_len <= 1e-30
        edge2 = p4 - p1
        v1_raw = edge2 - dot(edge2, v3) * v3
        v1_len = norm(v1_raw)
    end
    v1 = v1_len > 1e-30 ? v1_raw / v1_len : SVector(1.0, 0.0, 0.0)
    v2 = cross(v3, v1)
    return v1, v2, v3
end

@inline function quad4_gp_local_frame_from_coords3d(coords_3d::AbstractMatrix,
                                                    xi::Float64, eta::Float64)
    dNr, dNs = shape_derivs_quad(xi, eta)

    a1 = SVector(
        dNr[1] * coords_3d[1,1] + dNr[2] * coords_3d[2,1] + dNr[3] * coords_3d[3,1] + dNr[4] * coords_3d[4,1],
        dNr[1] * coords_3d[1,2] + dNr[2] * coords_3d[2,2] + dNr[3] * coords_3d[3,2] + dNr[4] * coords_3d[4,2],
        dNr[1] * coords_3d[1,3] + dNr[2] * coords_3d[2,3] + dNr[3] * coords_3d[3,3] + dNr[4] * coords_3d[4,3],
    )
    a2 = SVector(
        dNs[1] * coords_3d[1,1] + dNs[2] * coords_3d[2,1] + dNs[3] * coords_3d[3,1] + dNs[4] * coords_3d[4,1],
        dNs[1] * coords_3d[1,2] + dNs[2] * coords_3d[2,2] + dNs[3] * coords_3d[3,2] + dNs[4] * coords_3d[4,2],
        dNs[1] * coords_3d[1,3] + dNs[2] * coords_3d[2,3] + dNs[3] * coords_3d[3,3] + dNs[4] * coords_3d[4,3],
    )

    cross_a = cross(a1, a2)
    area_elem = norm(cross_a)
    n_gp = area_elem > 1e-30 ? cross_a / area_elem : SVector(0.0, 0.0, 1.0)
    a1_len = norm(a1)
    t1 = a1_len > 1e-30 ? a1 / a1_len : SVector(1.0, 0.0, 0.0)
    t2 = cross(n_gp, t1)

    J11 = a1_len
    J12 = 0.0
    J21 = dot(a2, t1)
    J22 = dot(a2, t2)
    return n_gp, area_elem, t1, t2, J11, J12, J21, J22
end

"""
Compute per-corner out-of-plane (z) coordinate of the element midsurface,
in the element-center local frame (v3 = diagonal-cross-product normal).
For a perfectly flat element this is zero on every corner. For a warped or
curved element, it captures the linear warp pattern (cylindrical, twist,
or combination) that the bilinear-shape-function curvature term misses.

Returns SVector{4} of local-z corner coordinates in the element-center frame.
"""
@inline function quad4_local_z_from_coords3d(coords_3d::AbstractMatrix)
    p1 = SVector(coords_3d[1,1], coords_3d[1,2], coords_3d[1,3])
    p2 = SVector(coords_3d[2,1], coords_3d[2,2], coords_3d[2,3])
    p3 = SVector(coords_3d[3,1], coords_3d[3,2], coords_3d[3,3])
    p4 = SVector(coords_3d[4,1], coords_3d[4,2], coords_3d[4,3])
    c = (p1 + p2 + p3 + p4) / 4
    n_raw = cross(p3 - p1, p4 - p2)
    nrm = norm(n_raw)
    v3 = nrm > 1e-30 ? n_raw / nrm : SVector(0.0, 0.0, 1.0)
    return SVector{4,Float64}(dot(p1 - c, v3), dot(p2 - c, v3),
                              dot(p3 - c, v3), dot(p4 - c, v3))
end

@inline function quad4_gp_curvature_membrane_from_coords3d(coords_3d::AbstractMatrix,
                                                           xi::Float64, eta::Float64)
    n_gp, _, _, _, J11, J12, J21, J22 =
        quad4_gp_local_frame_from_coords3d(coords_3d, xi, eta)

    a_rr = SVector(0.0, 0.0, 0.0)
    a_ss = SVector(0.0, 0.0, 0.0)
    a_rs = SVector(
        QUAD4_D2N_DXIDETA[1] * coords_3d[1,1] + QUAD4_D2N_DXIDETA[2] * coords_3d[2,1] +
        QUAD4_D2N_DXIDETA[3] * coords_3d[3,1] + QUAD4_D2N_DXIDETA[4] * coords_3d[4,1],
        QUAD4_D2N_DXIDETA[1] * coords_3d[1,2] + QUAD4_D2N_DXIDETA[2] * coords_3d[2,2] +
        QUAD4_D2N_DXIDETA[3] * coords_3d[3,2] + QUAD4_D2N_DXIDETA[4] * coords_3d[4,2],
        QUAD4_D2N_DXIDETA[1] * coords_3d[1,3] + QUAD4_D2N_DXIDETA[2] * coords_3d[2,3] +
        QUAD4_D2N_DXIDETA[3] * coords_3d[3,3] + QUAD4_D2N_DXIDETA[4] * coords_3d[4,3],
    )

    b_rr = dot(n_gp, a_rr)
    b_ss = dot(n_gp, a_ss)
    b_rs = dot(n_gp, a_rs)

    detJ = J11 * J22 - J12 * J21
    if abs(detJ) <= 1e-30
        return SVector(0.0, 0.0, 0.0)
    end

    inv_det = 1.0 / detJ
    iJ11 =  J22 * inv_det
    iJ12 = -J12 * inv_det
    iJ21 = -J21 * inv_det
    iJ22 =  J11 * inv_det

    k11 = iJ11 * iJ11 * b_rr + 2.0 * iJ11 * iJ12 * b_rs + iJ12 * iJ12 * b_ss
    k22 = iJ21 * iJ21 * b_rr + 2.0 * iJ21 * iJ22 * b_rs + iJ22 * iJ22 * b_ss
    k12 = iJ11 * iJ21 * b_rr + (iJ11 * iJ22 + iJ12 * iJ21) * b_rs + iJ12 * iJ22 * b_ss
    return SVector(k11, k22, k12)
end

@inline function quad4_gp_bending_connection_from_coords3d(coords_3d::AbstractMatrix,
                                                           xi::Float64, eta::Float64)
    dNr, dNs = shape_derivs_quad(xi, eta)

    a_r = SVector(
        dNr[1]*coords_3d[1,1] + dNr[2]*coords_3d[2,1] + dNr[3]*coords_3d[3,1] + dNr[4]*coords_3d[4,1],
        dNr[1]*coords_3d[1,2] + dNr[2]*coords_3d[2,2] + dNr[3]*coords_3d[3,2] + dNr[4]*coords_3d[4,2],
        dNr[1]*coords_3d[1,3] + dNr[2]*coords_3d[2,3] + dNr[3]*coords_3d[3,3] + dNr[4]*coords_3d[4,3],
    )
    a_s = SVector(
        dNs[1]*coords_3d[1,1] + dNs[2]*coords_3d[2,1] + dNs[3]*coords_3d[3,1] + dNs[4]*coords_3d[4,1],
        dNs[1]*coords_3d[1,2] + dNs[2]*coords_3d[2,2] + dNs[3]*coords_3d[3,2] + dNs[4]*coords_3d[4,2],
        dNs[1]*coords_3d[1,3] + dNs[2]*coords_3d[2,3] + dNs[3]*coords_3d[3,3] + dNs[4]*coords_3d[4,3],
    )
    a_rs = SVector(
        QUAD4_D2N_DXIDETA[1]*coords_3d[1,1] + QUAD4_D2N_DXIDETA[2]*coords_3d[2,1] +
        QUAD4_D2N_DXIDETA[3]*coords_3d[3,1] + QUAD4_D2N_DXIDETA[4]*coords_3d[4,1],
        QUAD4_D2N_DXIDETA[1]*coords_3d[1,2] + QUAD4_D2N_DXIDETA[2]*coords_3d[2,2] +
        QUAD4_D2N_DXIDETA[3]*coords_3d[3,2] + QUAD4_D2N_DXIDETA[4]*coords_3d[4,2],
        QUAD4_D2N_DXIDETA[1]*coords_3d[1,3] + QUAD4_D2N_DXIDETA[2]*coords_3d[2,3] +
        QUAD4_D2N_DXIDETA[3]*coords_3d[3,3] + QUAD4_D2N_DXIDETA[4]*coords_3d[4,3],
    )

    cross_a = cross(a_r, a_s)
    area_elem = norm(cross_a)
    a_r_len = norm(a_r)
    if area_elem <= 1e-30 || a_r_len <= 1e-30
        return SVector(0.0, 0.0)
    end

    n_gp = cross_a / area_elem
    t1 = a_r / a_r_len
    t2 = cross(n_gp, t1)

    J11 = a_r_len
    J12 = 0.0
    J21 = dot(a_s, t1)
    J22 = dot(a_s, t2)
    detJ = J11 * J22 - J12 * J21
    if abs(detJ) <= 1e-30
        return SVector(0.0, 0.0)
    end

    cross_r = cross(a_r, a_rs)
    cross_s = cross(a_rs, a_s)
    n_r = (cross_r - n_gp * dot(n_gp, cross_r)) / area_elem
    n_s = (cross_s - n_gp * dot(n_gp, cross_s)) / area_elem

    t1_r = SVector(0.0, 0.0, 0.0)
    t1_s = (a_rs - t1 * dot(t1, a_rs)) / a_r_len
    t2_r = cross(n_r, t1)
    t2_s = cross(n_s, t1) + cross(n_gp, t1_s)

    inv_det = 1.0 / detJ
    iJ11 =  J22 * inv_det
    iJ12 = -J12 * inv_det
    iJ21 = -J21 * inv_det
    iJ22 =  J11 * inv_det

    t1_x = iJ11 * t1_r + iJ12 * t1_s
    t1_y = iJ21 * t1_r + iJ22 * t1_s
    eta1 = dot(t1_x, t2)
    eta2 = dot(t1_y, t2)
    return SVector(eta1, eta2)
end

@inline function rotate_constitutive_3x3!(C::AbstractMatrix{Float64}, beta::Float64)
    abs(beta) <= 1e-12 && return C
    cb = cos(beta)
    sb = sin(beta)
    c2 = cb * cb
    s2 = sb * sb
    cs = cb * sb
    T11 = c2
    T12 = s2
    T13 = cs
    T21 = s2
    T22 = c2
    T23 = -cs
    T31 = -2.0 * cs
    T32 = 2.0 * cs
    T33 = c2 - s2

    t11 = C[1,1]*T11 + C[1,2]*T21 + C[1,3]*T31
    t12 = C[1,1]*T12 + C[1,2]*T22 + C[1,3]*T32
    t13 = C[1,1]*T13 + C[1,2]*T23 + C[1,3]*T33
    t21 = C[2,1]*T11 + C[2,2]*T21 + C[2,3]*T31
    t22 = C[2,1]*T12 + C[2,2]*T22 + C[2,3]*T32
    t23 = C[2,1]*T13 + C[2,2]*T23 + C[2,3]*T33
    t31 = C[3,1]*T11 + C[3,2]*T21 + C[3,3]*T31
    t32 = C[3,1]*T12 + C[3,2]*T22 + C[3,3]*T32
    t33 = C[3,1]*T13 + C[3,2]*T23 + C[3,3]*T33

    C[1,1] = T11*t11 + T21*t21 + T31*t31
    C[1,2] = T11*t12 + T21*t22 + T31*t32
    C[1,3] = T11*t13 + T21*t23 + T31*t33
    C[2,1] = T12*t11 + T22*t21 + T32*t31
    C[2,2] = T12*t12 + T22*t22 + T32*t32
    C[2,3] = T12*t13 + T22*t23 + T32*t33
    C[3,1] = T13*t11 + T23*t21 + T33*t31
    C[3,2] = T13*t12 + T23*t22 + T33*t32
    C[3,3] = T13*t13 + T23*t23 + T33*t33
    return C
end

@inline function rotate_constitutive_2x2!(C::AbstractMatrix{Float64}, beta::Float64)
    abs(beta) <= 1e-12 && return C
    cb = cos(beta)
    sb = sin(beta)
    a11 = C[1,1]
    a12 = C[1,2]
    a22 = C[2,2]
    C[1,1] = cb^2*a11 + 2.0*cb*sb*a12 + sb^2*a22
    C[1,2] = -cb*sb*a11 + (cb^2 - sb^2)*a12 + cb*sb*a22
    C[2,1] = C[1,2]
    C[2,2] = sb^2*a11 - 2.0*cb*sb*a12 + cb^2*a22
    return C
end

@inline function quad4_gp_rotation_from_element!(R::AbstractMatrix{Float64},
                                                 v1::SVector{3,Float64},
                                                 v2::SVector{3,Float64},
                                                 v3::SVector{3,Float64},
                                                 t1::SVector{3,Float64},
                                                 t2::SVector{3,Float64},
                                                 n_gp::SVector{3,Float64})
    R[1,1] = dot(t1, v1); R[1,2] = dot(t1, v2); R[1,3] = dot(t1, v3)
    R[2,1] = dot(t2, v1); R[2,2] = dot(t2, v2); R[2,3] = dot(t2, v3)
    R[3,1] = dot(n_gp, v1); R[3,2] = dot(n_gp, v2); R[3,3] = dot(n_gp, v3)

    v1_proj = v1 - dot(v1, n_gp) * n_gp
    v1_proj_len = norm(v1_proj)
    if v1_proj_len <= 1e-30
        v1_proj = v2 - dot(v2, n_gp) * n_gp
        v1_proj_len = norm(v1_proj)
    end
    if v1_proj_len <= 1e-30
        return 0.0
    end
    v1_proj_unit = v1_proj / v1_proj_len
    return atan(dot(v1_proj_unit, t2), dot(v1_proj_unit, t1))
end

@inline function rotate_quad4_dof_blocks!(B::AbstractMatrix{Float64}, R::AbstractMatrix{Float64})
    nrows = size(B, 1)
    @inbounds for k in 1:4
        idx = (k - 1) * 6
        for i in 1:nrows
            t1 = B[i, idx + 1]
            t2 = B[i, idx + 2]
            t3 = B[i, idx + 3]
            B[i, idx + 1] = t1*R[1,1] + t2*R[2,1] + t3*R[3,1]
            B[i, idx + 2] = t1*R[1,2] + t2*R[2,2] + t3*R[3,2]
            B[i, idx + 3] = t1*R[1,3] + t2*R[2,3] + t3*R[3,3]

            r1 = B[i, idx + 4]
            r2 = B[i, idx + 5]
            r3 = B[i, idx + 6]
            B[i, idx + 4] = r1*R[1,1] + r2*R[2,1] + r3*R[3,1]
            B[i, idx + 5] = r1*R[1,2] + r2*R[2,2] + r3*R[3,2]
            B[i, idx + 6] = r1*R[1,3] + r2*R[2,3] + r3*R[3,3]
        end
    end
    return B
end

@inline function rotate_quad4_dof_blocks!(b::AbstractVector{Float64}, R::AbstractMatrix{Float64})
    @inbounds for k in 1:4
        idx = (k - 1) * 6
        t1 = b[idx + 1]
        t2 = b[idx + 2]
        t3 = b[idx + 3]
        b[idx + 1] = t1*R[1,1] + t2*R[2,1] + t3*R[3,1]
        b[idx + 2] = t1*R[1,2] + t2*R[2,2] + t3*R[3,2]
        b[idx + 3] = t1*R[1,3] + t2*R[2,3] + t3*R[3,3]

        r1 = b[idx + 4]
        r2 = b[idx + 5]
        r3 = b[idx + 6]
        b[idx + 4] = r1*R[1,1] + r2*R[2,1] + r3*R[3,1]
        b[idx + 5] = r1*R[1,2] + r2*R[2,2] + r3*R[3,2]
        b[idx + 6] = r1*R[1,3] + r2*R[2,3] + r3*R[3,3]
    end
    return b
end

function stiffness_quad4(coords, E, nu, h; bend_ratio=1.0, ts_t=5.0/6.0, k6rot=100.0, ws::Union{Nothing,Quad4Workspace}=nothing)
    const_mem = E * h / (1 - nu^2)
    Cm = const_mem .* [1 nu 0; nu 1 0; 0 0 (1-nu)/2]

    const_bend = bend_ratio * (E * h^3) / (12 * (1 - nu^2))
    Cb = const_bend .* [1 nu 0; nu 1 0; 0 0 (1-nu)/2]

    G = E / (2*(1+nu))
    k_shear = ts_t * G * h
    Cs = k_shear .* [1 0; 0 1]

    return stiffness_quad4_matrices(coords, Cm, Cb, Cs, h, E; bend_ratio=bend_ratio, k6rot=k6rot, ws=ws)
end

function stiffness_quad4_generic(coords, E, nu, h; bend_ratio=1.0, ts_t=5.0/6.0, k6rot=100.0)
    T = promote_type(typeof(E), typeof(nu), typeof(h))
    oneT = one(T)
    zeroT = zero(T)

    const_mem = E * h / (oneT - nu^2)
    Cm = const_mem .* T[oneT nu zeroT; nu oneT zeroT; zeroT zeroT (oneT - nu) / T(2)]

    const_bend = T(bend_ratio) * (E * h^3) / (T(12) * (oneT - nu^2))
    Cb = const_bend .* T[oneT nu zeroT; nu oneT zeroT; zeroT zeroT (oneT - nu) / T(2)]

    G = E / (T(2) * (oneT + nu))
    Cs = (T(ts_t) * G * h) .* T[oneT zeroT; zeroT oneT]

    return stiffness_quad4_default_generic(coords, Cm, Cb, Cs, h, G; k6rot=k6rot)
end

function stiffness_quad4_default_generic(coords, Cm, Cb, Cs, h, G_ref; k6rot=100.0)
    T = promote_type(eltype(Cm), eltype(Cb), eltype(Cs), typeof(h), typeof(G_ref))
    zeroT = zero(T)
    oneT = one(T)

    Ke = zeros(T, 24, 24)
    K_ab = zeros(T, 24, 4)
    K_bb = zeros(T, 4, 4)

    Bs_tp = zeros(Float64, 4, 24)
    Bm = zeros(Float64, 3, 24)
    Bb = zeros(Float64, 3, 24)
    Bd = zeros(Float64, 1, 24)
    Bi = zeros(Float64, 3, 4)
    Bs_cov = zeros(Float64, 2, 24)

    tying_pts = ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))
    for tp_idx in 1:4
        xi_tp, eta_tp = tying_pts[tp_idx]
        dNr, dNs = shape_derivs_quad(xi_tp, eta_tp)
        J11 = dNr[1] * coords[1,1] + dNr[2] * coords[2,1] + dNr[3] * coords[3,1] + dNr[4] * coords[4,1]
        J12 = dNr[1] * coords[1,2] + dNr[2] * coords[2,2] + dNr[3] * coords[3,2] + dNr[4] * coords[4,2]
        J21 = dNs[1] * coords[1,1] + dNs[2] * coords[2,1] + dNs[3] * coords[3,1] + dNs[4] * coords[4,1]
        J22 = dNs[1] * coords[1,2] + dNs[2] * coords[2,2] + dNs[3] * coords[3,2] + dNs[4] * coords[4,2]
        N_tp = (
            0.25 * (1.0 - xi_tp) * (1.0 - eta_tp),
            0.25 * (1.0 + xi_tp) * (1.0 - eta_tp),
            0.25 * (1.0 + xi_tp) * (1.0 + eta_tp),
            0.25 * (1.0 - xi_tp) * (1.0 + eta_tp),
        )
        if tp_idx <= 2
            for k in 1:4
                idx = (k - 1) * 6
                Bs_tp[tp_idx, idx + 3] = dNr[k]
                Bs_tp[tp_idx, idx + 4] = -J12 * N_tp[k]
                Bs_tp[tp_idx, idx + 5] = J11 * N_tp[k]
            end
        else
            for k in 1:4
                idx = (k - 1) * 6
                Bs_tp[tp_idx, idx + 3] = dNs[k]
                Bs_tp[tp_idx, idx + 4] = -J22 * N_tp[k]
                Bs_tp[tp_idx, idx + 5] = J21 * N_tp[k]
            end
        end
    end

    dNr_c = SVector(-0.25, 0.25, 0.25, -0.25)
    dNs_c = SVector(-0.25, -0.25, 0.25, 0.25)
    J11c = dNr_c[1] * coords[1,1] + dNr_c[2] * coords[2,1] + dNr_c[3] * coords[3,1] + dNr_c[4] * coords[4,1]
    J12c = dNr_c[1] * coords[1,2] + dNr_c[2] * coords[2,2] + dNr_c[3] * coords[3,2] + dNr_c[4] * coords[4,2]
    J21c = dNs_c[1] * coords[1,1] + dNs_c[2] * coords[2,1] + dNs_c[3] * coords[3,1] + dNs_c[4] * coords[4,1]
    J22c = dNs_c[1] * coords[1,2] + dNs_c[2] * coords[2,2] + dNs_c[3] * coords[3,2] + dNs_c[4] * coords[4,2]
    detJc = J11c * J22c - J12c * J21c
    abs_detJc = abs(detJc)
    inv_detc = 1.0 / detJc
    iJ11c = J22c * inv_detc
    iJ12c = -J12c * inv_detc
    iJ21c = -J21c * inv_detc
    iJ22c = J11c * inv_detc

    phi2_alpha = PHI2_ALPHA[]
    L_char_sq = max(4.0 * abs_detJc, 1e-30)
    phi2_trial = T(phi2_alpha) * h^2 / T(L_char_sq)
    phi2_shear = phi2_alpha > 0.0 ? min(oneT, phi2_trial) : oneT

    alpha_drill = (T(k6rot) / T(1e5)) * G_ref * h

    pt = 1.0 / sqrt(3.0)
    gauss_pts = ((-pt, -pt), (pt, -pt), (pt, pt), (-pt, pt))

    for gp in gauss_pts
        r, s = gp
        dNr, dNs = shape_derivs_quad(r, s)

        J11 = dNr[1] * coords[1,1] + dNr[2] * coords[2,1] + dNr[3] * coords[3,1] + dNr[4] * coords[4,1]
        J12 = dNr[1] * coords[1,2] + dNr[2] * coords[2,2] + dNr[3] * coords[3,2] + dNr[4] * coords[4,2]
        J21 = dNs[1] * coords[1,1] + dNs[2] * coords[2,1] + dNs[3] * coords[3,1] + dNs[4] * coords[4,1]
        J22 = dNs[1] * coords[1,2] + dNs[2] * coords[2,2] + dNs[3] * coords[3,2] + dNs[4] * coords[4,2]
        detJ = J11 * J22 - J12 * J21
        abs_detJ = max(abs(detJ), 1e-12)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(Bm, 0.0)
        fill!(Bb, 0.0)
        fill!(Bd, 0.0)
        for k in 1:4
            dN_dx = iJ11 * dNr[k] + iJ12 * dNs[k]
            dN_dy = iJ21 * dNr[k] + iJ22 * dNs[k]
            idx = (k - 1) * 6
            N_k = 0.25 * (1 + (k == 2 || k == 3 ? r : -r)) * (1 + (k >= 3 ? s : -s))
            Bm[1, idx + 1] = dN_dx
            Bm[2, idx + 2] = dN_dy
            Bm[3, idx + 1] = dN_dy
            Bm[3, idx + 2] = dN_dx
            Bb[1, idx + 5] = dN_dx
            Bb[2, idx + 4] = -dN_dy
            Bb[3, idx + 5] = dN_dy
            Bb[3, idx + 4] = -dN_dx
            Bd[1, idx + 1] = 0.5 * dN_dy
            Bd[1, idx + 2] = -0.5 * dN_dx
            Bd[1, idx + 6] = N_k
        end

        fill_quad4_membrane_incompatible_B!(
            Bi,
            r,
            s,
            iJ11,
            iJ12,
            iJ21,
            iJ22,
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            false,
        )

        Ke .+= (Bm' * Cm * Bm) .* abs_detJ
        Ke .+= (Bb' * Cb * Bb) .* abs_detJ
        Ke .+= (Bd' * Bd) .* (abs_detJ * alpha_drill)

        w_eta_p = 0.5 * (1.0 + s)
        w_eta_m = 0.5 * (1.0 - s)
        w_xi_p = 0.5 * (1.0 + r)
        w_xi_m = 0.5 * (1.0 - r)
        fill!(Bs_cov, 0.0)
        for j in 1:24
            Bs_cov[1, j] = w_eta_m * Bs_tp[1, j] + w_eta_p * Bs_tp[2, j]
            Bs_cov[2, j] = w_xi_m * Bs_tp[3, j] + w_xi_p * Bs_tp[4, j]
        end
        invJ = Float64[iJ11 iJ12; iJ21 iJ22]
        Cs_cov = phi2_shear .* (invJ' * Cs * invJ)
        Ke .+= (Bs_cov' * Cs_cov * Bs_cov) .* abs_detJ

        K_ab .+= (Bm' * Cm * Bi) .* abs_detJ
        K_bb .+= (Bi' * Cm * Bi) .* abs_detJ
    end

    if maximum(abs, K_bb) > zeroT
        Ke .-= K_ab * (K_bb \ K_ab')
    end

    return Ke
end

@inline function project_material_membrane_shear!(
    Bm::AbstractMatrix,
    dNdx_c,
    dNdy_c,
    curvature_membrane,
    theta::Float64,
)
    c = cos(theta)
    s = sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s
    k11 = curvature_membrane === nothing ? 0.0 : curvature_membrane[1]
    k22 = curvature_membrane === nothing ? 0.0 : curvature_membrane[2]
    k12 = curvature_membrane === nothing ? 0.0 : curvature_membrane[3]

    @inbounds for k in 1:4
        idx = (k - 1) * 6

        r1 = Bm[1, idx+1]; r2 = Bm[2, idx+1]; r3 = Bm[3, idx+1]
        r1c = dNdx_c[k];   r2c = 0.0;         r3c = dNdy_c[k]
        m1 = c2 * r1 + s2 * r2 + cs * r3
        m2 = s2 * r1 + c2 * r2 - cs * r3
        m3c = -2.0 * cs * r1c + 2.0 * cs * r2c + (c2 - s2) * r3c
        Bm[1, idx+1] = c2 * m1 + s2 * m2 - cs * m3c
        Bm[2, idx+1] = s2 * m1 + c2 * m2 + cs * m3c
        Bm[3, idx+1] = 2.0 * cs * m1 - 2.0 * cs * m2 + (c2 - s2) * m3c

        r1 = Bm[1, idx+2]; r2 = Bm[2, idx+2]; r3 = Bm[3, idx+2]
        r1c = 0.0;         r2c = dNdy_c[k];   r3c = dNdx_c[k]
        m1 = c2 * r1 + s2 * r2 + cs * r3
        m2 = s2 * r1 + c2 * r2 - cs * r3
        m3c = -2.0 * cs * r1c + 2.0 * cs * r2c + (c2 - s2) * r3c
        Bm[1, idx+2] = c2 * m1 + s2 * m2 - cs * m3c
        Bm[2, idx+2] = s2 * m1 + c2 * m2 + cs * m3c
        Bm[3, idx+2] = 2.0 * cs * m1 - 2.0 * cs * m2 + (c2 - s2) * m3c

        r1 = Bm[1, idx+3]; r2 = Bm[2, idx+3]; r3 = Bm[3, idx+3]
        r1c = -0.25 * k11
        r2c = -0.25 * k22
        r3c = -0.5 * k12
        m1 = c2 * r1 + s2 * r2 + cs * r3
        m2 = s2 * r1 + c2 * r2 - cs * r3
        m3c = -2.0 * cs * r1c + 2.0 * cs * r2c + (c2 - s2) * r3c
        Bm[1, idx+3] = c2 * m1 + s2 * m2 - cs * m3c
        Bm[2, idx+3] = s2 * m1 + c2 * m2 + cs * m3c
        Bm[3, idx+3] = 2.0 * cs * m1 - 2.0 * cs * m2 + (c2 - s2) * m3c
    end

    return Bm
end

@inline function apply_membrane_ans_mitc4plus!(Bm::AbstractMatrix, coords::AbstractMatrix, xi::Float64, eta::Float64)
    x1 = coords[1,1]; y1 = coords[1,2]
    x2 = coords[2,1]; y2 = coords[2,2]
    x3 = coords[3,1]; y3 = coords[3,2]
    x4 = coords[4,1]; y4 = coords[4,2]

    xr1 = 0.25 * (-x1 + x2 + x3 - x4)
    xr2 = 0.25 * (-y1 + y2 + y3 - y4)
    xs1 = 0.25 * (-x1 - x2 + x3 + x4)
    xs2 = 0.25 * (-y1 - y2 + y3 + y4)
    xd1 = 0.25 * (x1 - x2 + x3 - x4)
    xd2 = 0.25 * (y1 - y2 + y3 - y4)

    det0 = xr1 * xs2 - xr2 * xs1
    abs(det0) < 1e-12 && return Bm

    mr1 = xs2 / det0
    mr2 = -xs1 / det0
    ms1 = -xr2 / det0
    ms2 = xr1 / det0

    c_r = mr1 * xd1 + mr2 * xd2
    c_s = ms1 * xd1 + ms2 * xd2
    d = c_r * c_r + c_s * c_s - 1.0
    abs(d) < 1e-10 && return Bm

    coef_r = (-0.25, 0.25, 0.25, -0.25)
    coef_s = (-0.25, -0.25, 0.25, 0.25)
    coef_d = (0.25, -0.25, 0.25, -0.25)

    xi_eta = xi * eta
    xi2_m1 = xi * xi - 1.0
    eta2_m1 = eta * eta - 1.0
    inv_d = 1.0 / d

    # Zero only the u/v columns we are about to overwrite. Do NOT fill! the
    # entire Bm: the caller may have already filled idx+3 (w-DOF) columns with
    # curvature coupling terms (-N_k * curvature_membrane[i]) that must be
    # preserved for curved-shell formulations (Ko-Lee-Bathe 2016 §2.3).
    @inbounds for k in 1:4
        idx = (k - 1) * 6
        Bm[1, idx+1] = 0.0; Bm[1, idx+2] = 0.0
        Bm[2, idx+1] = 0.0; Bm[2, idx+2] = 0.0
        Bm[3, idx+1] = 0.0; Bm[3, idx+2] = 0.0
    end
    @inbounds for k in 1:4
        idx = (k - 1) * 6
        rk = coef_r[k]
        sk = coef_s[k]
        dk = coef_d[k]

        rr_con_x = xr1 * rk
        rr_con_y = xr2 * rk
        rr_lin_x = xr1 * dk + xd1 * rk
        rr_lin_y = xr2 * dk + xd2 * rk

        ss_con_x = xs1 * sk
        ss_con_y = xs2 * sk
        ss_lin_x = xs1 * dk + xd1 * sk
        ss_lin_y = xs2 * dk + xd2 * sk

        rs_con_x = 0.5 * (xr1 * sk + xs1 * rk)
        rs_con_y = 0.5 * (xr2 * sk + xs2 * rk)
        rs_bil_x = xd1 * dk
        rs_bil_y = xd2 * dk

        rs_bil_tilde_x =
            (c_r * (c_r * (rr_con_x + rs_bil_x) - rr_lin_x) +
             c_s * (c_s * (ss_con_x + rs_bil_x) - ss_lin_x) +
             2.0 * c_r * c_s * rs_con_x) * inv_d
        rs_bil_tilde_y =
            (c_r * (c_r * (rr_con_y + rs_bil_y) - rr_lin_y) +
             c_s * (c_s * (ss_con_y + rs_bil_y) - ss_lin_y) +
             2.0 * c_r * c_s * rs_con_y) * inv_d

        cov_rr_x = rr_con_x + rs_bil_x + eta * rr_lin_x + eta2_m1 * rs_bil_tilde_x
        cov_rr_y = rr_con_y + rs_bil_y + eta * rr_lin_y + eta2_m1 * rs_bil_tilde_y
        cov_ss_x = ss_con_x + rs_bil_x + xi * ss_lin_x + xi2_m1 * rs_bil_tilde_x
        cov_ss_y = ss_con_y + rs_bil_y + xi * ss_lin_y + xi2_m1 * rs_bil_tilde_y
        cov_rs_x = rs_con_x + 0.5 * xi * rr_lin_x + 0.5 * eta * ss_lin_x + xi_eta * rs_bil_tilde_x
        cov_rs_y = rs_con_y + 0.5 * xi * rr_lin_y + 0.5 * eta * ss_lin_y + xi_eta * rs_bil_tilde_y

        Bm[1, idx+1] = mr1 * mr1 * cov_rr_x + ms1 * ms1 * cov_ss_x + 2.0 * mr1 * ms1 * cov_rs_x
        Bm[1, idx+2] = mr1 * mr1 * cov_rr_y + ms1 * ms1 * cov_ss_y + 2.0 * mr1 * ms1 * cov_rs_y

        Bm[2, idx+1] = mr2 * mr2 * cov_rr_x + ms2 * ms2 * cov_ss_x + 2.0 * mr2 * ms2 * cov_rs_x
        Bm[2, idx+2] = mr2 * mr2 * cov_rr_y + ms2 * ms2 * cov_ss_y + 2.0 * mr2 * ms2 * cov_rs_y

        Bm[3, idx+1] = 2.0 * mr1 * mr2 * cov_rr_x + 2.0 * ms1 * ms2 * cov_ss_x +
                        2.0 * (mr1 * ms2 + mr2 * ms1) * cov_rs_x
        Bm[3, idx+2] = 2.0 * mr1 * mr2 * cov_rr_y + 2.0 * ms1 * ms2 * cov_ss_y +
                        2.0 * (mr1 * ms2 + mr2 * ms1) * cov_rs_y
    end

    return Bm
end

@inline function use_membrane_ans_mitc4plus(mode::Symbol, coords::AbstractMatrix, curvature_membrane)
    if curvature_membrane !== nothing && !MITC4PLUS_ALLOW_CURVED[]
        return false
    end
    if mode === :mitc4plus_all
        return true
    elseif mode === :mitc4plus
        return !quad4_is_axis_aligned_rectangle(coords)
    end
    return false
end

@inline function quad4_membrane_incompatible_jacobian_components(
    membrane_incomp_center_jacobian::Bool,
    iJ11::Float64, iJ12::Float64, iJ21::Float64, iJ22::Float64,
    iJ11c::Float64, iJ12c::Float64, iJ21c::Float64, iJ22c::Float64,
)
    if membrane_incomp_center_jacobian
        return iJ11c, iJ12c, iJ21c, iJ22c
    end
    return iJ11, iJ12, iJ21, iJ22
end

@inline function fill_quad4_membrane_incompatible_B!(
    Bi::AbstractMatrix,
    r::Float64,
    s::Float64,
    iJ11::Float64, iJ12::Float64, iJ21::Float64, iJ22::Float64,
    iJ11c::Float64, iJ12c::Float64, iJ21c::Float64, iJ22c::Float64,
    membrane_incomp_center_jacobian::Bool,
)
    miJ11, miJ12, miJ21, miJ22 = quad4_membrane_incompatible_jacobian_components(
        membrane_incomp_center_jacobian,
        iJ11, iJ12, iJ21, iJ22,
        iJ11c, iJ12c, iJ21c, iJ22c,
    )
    dphi1_dx = miJ11 * (-2.0 * r)
    dphi1_dy = miJ21 * (-2.0 * r)
    dphi2_dx = miJ12 * (-2.0 * s)
    dphi2_dy = miJ22 * (-2.0 * s)

    fill!(Bi, 0.0)
    Bi[1,1] = dphi1_dx
    Bi[3,1] = dphi1_dy
    Bi[2,2] = dphi1_dy
    Bi[3,2] = dphi1_dx
    Bi[1,3] = dphi2_dx
    Bi[3,3] = dphi2_dy
    Bi[2,4] = dphi2_dy
    Bi[3,4] = dphi2_dx
    return Bi
end

@inline function fill_quad4_membrane_enhanced_B!(
    Bi::AbstractMatrix,
    r::Float64,
    s::Float64,
    iJ11::Float64, iJ12::Float64, iJ21::Float64, iJ22::Float64,
    iJ11c::Float64, iJ12c::Float64, iJ21c::Float64, iJ22c::Float64,
    membrane_incomp_center_jacobian::Bool,
)
    miJ11, miJ12, miJ21, miJ22 = quad4_membrane_incompatible_jacobian_components(
        membrane_incomp_center_jacobian,
        iJ11, iJ12, iJ21, iJ22,
        iJ11c, iJ12c, iJ21c, iJ22c,
    )
    dphi1_dx = miJ11 * (-2.0 * r)
    dphi1_dy = miJ21 * (-2.0 * r)
    dphi2_dx = miJ12 * (-2.0 * s)
    dphi2_dy = miJ22 * (-2.0 * s)
    dpsi_dx = miJ11 * s + miJ12 * r
    dpsi_dy = miJ21 * s + miJ22 * r

    fill!(Bi, 0.0)
    Bi[1,1] = dphi1_dx
    Bi[3,1] = dphi1_dy
    Bi[2,2] = dphi1_dy
    Bi[3,2] = dphi1_dx
    Bi[1,3] = dphi2_dx
    Bi[3,3] = dphi2_dy
    Bi[2,4] = dphi2_dy
    Bi[3,4] = dphi2_dx
    Bi[1,5] = dpsi_dx
    Bi[3,5] = dpsi_dy
    Bi[2,6] = dpsi_dy
    Bi[3,6] = dpsi_dx
    return Bi
end

function stiffness_quad4_membrane_enhanced_matrices(
    coords,
    Cm,
    h,
    E_ref;
    enhanced_modes::Bool=true,
    k6rot=100.0,
    drill_scale::Float64=1.0,
    curvature_membrane=nothing,
    membrane_shear_center_row::Bool=false,
    material_shear_rotation::Float64=0.0,
    membrane_assumed_mode::Symbol=:none,
    membrane_incomp_center_jacobian::Bool=false,
)
    Ke = zeros(24, 24)
    K_ab = zeros(24, 6)
    K_bb = zeros(6, 6)
    Bm = zeros(3, 24)
    Bd = zeros(1, 24)
    Bi = zeros(3, 6)
    tmp3x24 = zeros(3, 24)
    tmp3x6 = zeros(3, 6)

    dNr_c = SVector(-0.25, 0.25, 0.25, -0.25)
    dNs_c = SVector(-0.25, -0.25, 0.25, 0.25)
    J11c = dNr_c[1]*coords[1,1]+dNr_c[2]*coords[2,1]+dNr_c[3]*coords[3,1]+dNr_c[4]*coords[4,1]
    J12c = dNr_c[1]*coords[1,2]+dNr_c[2]*coords[2,2]+dNr_c[3]*coords[3,2]+dNr_c[4]*coords[4,2]
    J21c = dNs_c[1]*coords[1,1]+dNs_c[2]*coords[2,1]+dNs_c[3]*coords[3,1]+dNs_c[4]*coords[4,1]
    J22c = dNs_c[1]*coords[1,2]+dNs_c[2]*coords[2,2]+dNs_c[3]*coords[3,2]+dNs_c[4]*coords[4,2]
    detJc = J11c*J22c - J12c*J21c
    abs(detJc) < 1e-12 && (detJc = detJc < 0.0 ? -1e-12 : 1e-12)
    inv_detc = 1.0 / detJc
    iJ11c =  J22c*inv_detc
    iJ12c = -J12c*inv_detc
    iJ21c = -J21c*inv_detc
    iJ22c =  J11c*inv_detc
    dNdx_c = ntuple(k -> iJ11c*dNr_c[k] + iJ12c*dNs_c[k], 4)
    dNdy_c = ntuple(k -> iJ21c*dNr_c[k] + iJ22c*dNs_c[k], 4)

    G_drill = Cm[3,3] / h
    if G_drill < 1e-6
        G_drill = E_ref / (2 * 3.0)
    end
    alpha_drill = drill_scale * (k6rot / 1e5) * G_drill * h

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt,-pt), SVector(pt,-pt), SVector(pt,pt), SVector(-pt,pt))

    @inbounds @fastmath for gp in 1:4
        r, s = gauss_pts[gp][1], gauss_pts[gp][2]
        dNr, dNs = shape_derivs_quad(r, s)

        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        abs_detJ = abs(detJ)
        if abs_detJ < 1e-12
            abs_detJ = 1e-12
        end
        inv_det = 1.0 / detJ
        iJ11 = J22*inv_det
        iJ12 = -J12*inv_det
        iJ21 = -J21*inv_det
        iJ22 = J11*inv_det

        fill!(Bm, 0.0)
        fill!(Bd, 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            idx = (k-1)*6
            N_k = 0.25*(1 + (k==2||k==3 ? r : -r))*(1 + (k>=3 ? s : -s))
            Bm[1, idx+1] = dN_dx
            Bm[2, idx+2] = dN_dy
            Bm[3, idx+1] = dN_dy
            Bm[3, idx+2] = dN_dx
            if curvature_membrane !== nothing
                Bm[1, idx+3] = -N_k * curvature_membrane[1]
                Bm[2, idx+3] = -N_k * curvature_membrane[2]
                Bm[3, idx+3] = -2.0 * N_k * curvature_membrane[3]
            end
            Bd[1, idx+1] = 0.5*dN_dy
            Bd[1, idx+2] = -0.5*dN_dx
            Bd[1, idx+6] = N_k
        end

        if membrane_shear_center_row
            project_material_membrane_shear!(
                Bm,
                dNdx_c,
                dNdy_c,
                curvature_membrane,
                material_shear_rotation,
            )
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
            apply_membrane_ans_mitc4plus!(Bm, coords, r, s)
        end

        ts_mul!(tmp3x24, Cm, Bm)
        ts_mul_At_add!(Ke, Bm, tmp3x24, abs_detJ)
        ts_mul_At_add!(Ke, Bd, Bd, abs_detJ * alpha_drill)

        if enhanced_modes
            fill_quad4_membrane_enhanced_B!(
                Bi,
                r,
                s,
                iJ11,
                iJ12,
                iJ21,
                iJ22,
                iJ11c,
                iJ12c,
                iJ21c,
                iJ22c,
                membrane_incomp_center_jacobian,
            )
            ts_mul!(tmp3x6, Cm, Bi)
            ts_mul_At_add!(K_ab, Bm, tmp3x6, abs_detJ)
            ts_mul_At_add!(K_bb, Bi, tmp3x6, abs_detJ)
        end
    end

    if enhanced_modes && maximum(abs, K_bb) > 1e-30
        Kcorr = K_bb \ K_ab'
        @inbounds @fastmath for j in 1:24, i in 1:24
            s = 0.0
            for l in 1:6
                s += K_ab[i, l] * Kcorr[l, j]
            end
            Ke[i, j] -= s
        end
    end

    return Ke
end

@inline function quad4_hierarchical_edge_shapes(r::Float64, s::Float64)
    N5 = 0.5 * (1.0 - r * r) * (1.0 - s)
    N6 = 0.5 * (1.0 + r) * (1.0 - s * s)
    N7 = 0.5 * (1.0 - r * r) * (1.0 + s)
    N8 = 0.5 * (1.0 - r) * (1.0 - s * s)
    return SVector(N5, N6, N7, N8)
end

@inline function quad4_hierarchical_edge_shape_derivs(r::Float64, s::Float64)
    dNr = SVector(
        -r * (1.0 - s),
        0.5 * (1.0 - s * s),
        -r * (1.0 + s),
        -0.5 * (1.0 - s * s),
    )
    dNs = SVector(
        -0.5 * (1.0 - r * r),
        -(1.0 + r) * s,
        0.5 * (1.0 - r * r),
        -(1.0 - r) * s,
    )
    return dNr, dNs
end

function stiffness_quad4_membrane_normal_rot_matrices(
    coords,
    Cm,
    h;
    curvature_membrane=nothing,
    include_drill_penalty::Bool=true,
)
    # Exact flat-membrane DKMQ24_2+ operator from the published formulation:
    # Allman drilling enrichment + 2-DOF bubble condensation + Hughes-Brezzi penalty.
    Ke = zeros(24, 24)
    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))

    edge_pairs = ((1, 2), (2, 3), (3, 4), (4, 1))
    edge_coeff_x = zeros(4)
    edge_coeff_y = zeros(4)
    @inbounds for e in 1:4
        i, j = edge_pairs[e]
        dx = coords[j, 1] - coords[i, 1]
        dy = coords[j, 2] - coords[i, 2]
        edge_coeff_x[e] = -dy / 8.0
        edge_coeff_y[e] =  dx / 8.0
    end

    gp_detJ = zeros(4)
    Bm_store = zeros(4, 3, 24)
    Bn_store = zeros(4, 3, 2)
    Kmn = zeros(24, 2)
    Knn = zeros(2, 2)
    tmp3x2 = zeros(3, 2)
    tmp3x24 = zeros(3, 24)
    area = 0.0

    @inbounds for gp in 1:4
        r, s = gauss_pts[gp][1], gauss_pts[gp][2]
        dNr, dNs = shape_derivs_quad(r, s)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11 * J22 - J12 * J21
        abs_detJ = abs(detJ)
        abs_detJ < 1e-12 && (abs_detJ = 1e-12)
        gp_detJ[gp] = abs_detJ
        area += abs_detJ

        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(view(Bm_store, gp, :, :), 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            idx = (k - 1) * 6
            N_k = 0.25*(1 + (k==2||k==3 ? r : -r))*(1 + (k>=3 ? s : -s))
            Bm_store[gp, 1, idx + 1] = dN_dx
            Bm_store[gp, 2, idx + 2] = dN_dy
            Bm_store[gp, 3, idx + 1] = dN_dy
            Bm_store[gp, 3, idx + 2] = dN_dx
            if curvature_membrane !== nothing
                Bm_store[gp, 1, idx + 3] = -N_k * curvature_membrane[1]
                Bm_store[gp, 2, idx + 3] = -N_k * curvature_membrane[2]
                Bm_store[gp, 3, idx + 3] = -2.0 * N_k * curvature_membrane[3]
            end
        end

        dH_dr, dH_ds = quad4_hierarchical_edge_shape_derivs(r, s)
        for e in 1:4
            dHx = iJ11 * dH_dr[e] + iJ12 * dH_ds[e]
            dHy = iJ21 * dH_dr[e] + iJ22 * dH_ds[e]
            ni, nj = edge_pairs[e]
            col_i = (ni - 1) * 6 + 6
            col_j = (nj - 1) * 6 + 6
            coeff_x = edge_coeff_x[e]
            coeff_y = edge_coeff_y[e]
            Bm_store[gp, 1, col_i] += coeff_x * dHx
            Bm_store[gp, 1, col_j] -= coeff_x * dHx
            Bm_store[gp, 2, col_i] += coeff_y * dHy
            Bm_store[gp, 2, col_j] -= coeff_y * dHy
            Bm_store[gp, 3, col_i] += coeff_x * dHy + coeff_y * dHx
            Bm_store[gp, 3, col_j] -= coeff_x * dHy + coeff_y * dHx
        end

        dN9_dr = -2.0 * r * (1.0 - s * s)
        dN9_ds = -2.0 * s * (1.0 - r * r)
        dN9_dx = iJ11 * dN9_dr + iJ12 * dN9_ds
        dN9_dy = iJ21 * dN9_dr + iJ22 * dN9_ds
        Bn_store[gp, 1, 1] = dN9_dx
        Bn_store[gp, 1, 2] = 0.0
        Bn_store[gp, 2, 1] = 0.0
        Bn_store[gp, 2, 2] = dN9_dy
        Bn_store[gp, 3, 1] = dN9_dy
        Bn_store[gp, 3, 2] = dN9_dx

        ts_mul!(tmp3x2, Cm, view(Bn_store, gp, :, :))
        ts_mul_At_add!(Kmn, view(Bm_store, gp, :, :), tmp3x2, abs_detJ)
        ts_mul_At_add!(Knn, view(Bn_store, gp, :, :), tmp3x2, abs_detJ)
    end

    T = Knn \ transpose(Kmn)
    Beff = zeros(3, 24)
    @inbounds for gp in 1:4
        for j in 1:24, i in 1:3
            Beff[i, j] = Bm_store[gp, i, j] -
                         (Bn_store[gp, i, 1] * T[1, j] + Bn_store[gp, i, 2] * T[2, j])
        end
        ts_mul!(tmp3x24, Cm, Beff)
        ts_mul_At_add!(Ke, Beff, tmp3x24, gp_detJ[gp])
    end

    # Published DKMQ24_2+ Hughes-Brezzi drilling penalty with 1x1 quadrature.
    dNr0, dNs0 = shape_derivs_quad(0.0, 0.0)
    J11c = dNr0[1]*coords[1,1] + dNr0[2]*coords[2,1] + dNr0[3]*coords[3,1] + dNr0[4]*coords[4,1]
    J12c = dNr0[1]*coords[1,2] + dNr0[2]*coords[2,2] + dNr0[3]*coords[3,2] + dNr0[4]*coords[4,2]
    J21c = dNs0[1]*coords[1,1] + dNs0[2]*coords[2,1] + dNs0[3]*coords[3,1] + dNs0[4]*coords[4,1]
    J22c = dNs0[1]*coords[1,2] + dNs0[2]*coords[2,2] + dNs0[3]*coords[3,2] + dNs0[4]*coords[4,2]
    detJc = J11c * J22c - J12c * J21c
    abs_detJc = abs(detJc)
    abs_detJc < 1e-12 && (abs_detJc = 1e-12)
    inv_detc = 1.0 / detJc
    iJ11c = J22c * inv_detc
    iJ12c = -J12c * inv_detc
    iJ21c = -J21c * inv_detc
    iJ22c = J11c * inv_detc
    g = zeros(24)
    for k in 1:4
        dN_dx = iJ11c*dNr0[k] + iJ12c*dNs0[k]
        dN_dy = iJ21c*dNr0[k] + iJ22c*dNs0[k]
        idx = (k - 1) * 6
        g[idx + 1] = -0.5 * dN_dy
        g[idx + 2] =  0.5 * dN_dx
        g[idx + 6] = -0.25
    end
    if include_drill_penalty
        c2 = 0.1
        c1 = c2 * h / sqrt(max(area, 1e-12))
        kstab = c1 * Cm[3, 3]
        stab_weight = 4.0 * abs_detJc
        @inbounds @fastmath for j in 1:24, i in 1:24
            Ke[i, j] += kstab * stab_weight * g[i] * g[j]
        end
    end

    return Ke
end

function stiffness_quad4_membrane_hybrid_stress_matrices(
    coords,
    Cm,
    h;
    curvature_membrane=nothing,
    curvature_w_coupling::Bool=false,
    include_drill_penalty::Bool=true,
)
    Ke = zeros(24, 24)

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))

    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J0 = @SMatrix [
        dNr_c[1]*coords[1,1] + dNr_c[2]*coords[2,1] + dNr_c[3]*coords[3,1] + dNr_c[4]*coords[4,1]  dNr_c[1]*coords[1,2] + dNr_c[2]*coords[2,2] + dNr_c[3]*coords[3,2] + dNr_c[4]*coords[4,2];
        dNs_c[1]*coords[1,1] + dNs_c[2]*coords[2,1] + dNs_c[3]*coords[3,1] + dNs_c[4]*coords[4,1]  dNs_c[1]*coords[1,2] + dNs_c[2]*coords[2,2] + dNs_c[3]*coords[3,2] + dNs_c[4]*coords[4,2]
    ]

    S_inv = inv(Cm)
    H = zeros(5, 5)
    G = zeros(5, 24)
    area = 0.0

    function fill_stress_mode!(P::AbstractMatrix, xi::Float64, eta::Float64)
        fill!(P, 0.0)
        skew_modes = Vector{SMatrix{2, 2, Float64, 4}}(undef, 5)
        skew_modes[1] = @SMatrix [1.0 0.0; 0.0 0.0]
        skew_modes[2] = @SMatrix [0.0 0.0; 0.0 1.0]
        skew_modes[3] = @SMatrix [0.0 1.0; 1.0 0.0]
        skew_modes[4] = @SMatrix [eta 0.0; 0.0 0.0]
        skew_modes[5] = @SMatrix [0.0 0.0; 0.0 xi]
        @inbounds for a in 1:5
            sigma_mat = J0 * skew_modes[a] * transpose(J0)
            P[1, a] = sigma_mat[1, 1]
            P[2, a] = sigma_mat[2, 2]
            P[3, a] = sigma_mat[1, 2]
        end
        return P
    end

    Bm = zeros(3, 24)
    P = zeros(3, 5)
    tmp3x24 = zeros(3, 24)
    tmp3x5 = zeros(3, 5)

    @inbounds for gp in gauss_pts
        xi, eta = gp[1], gp[2]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        abs_detJ = abs(detJ)
        if abs_detJ < 1e-12
            abs_detJ = 1e-12
        end
        area += abs_detJ

        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(Bm, 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            N_k = 0.25*(1 + (k==2||k==3 ? xi : -xi))*(1 + (k>=3 ? eta : -eta))
            idx = (k - 1) * 6
            Bm[1, idx + 1] = dN_dx
            Bm[2, idx + 2] = dN_dy
            Bm[3, idx + 1] = dN_dy
            Bm[3, idx + 2] = dN_dx
            if curvature_w_coupling && curvature_membrane !== nothing
                Bm[1, idx + 3] = -N_k * curvature_membrane[1]
                Bm[2, idx + 3] = -N_k * curvature_membrane[2]
                Bm[3, idx + 3] = -2.0 * N_k * curvature_membrane[3]
            end
        end

        fill_stress_mode!(P, xi, eta)
        ts_mul!(tmp3x5, S_inv, P)
        ts_mul_At_add!(H, P, tmp3x5, abs_detJ)
        ts_mul_At_add!(G, P, Bm, abs_detJ)
    end

    Kmem = transpose(G) * (H \ G)
    Ke .+= Kmem

    # Minimal drilling regularization tied to the membrane center field to keep
    # the shell operator well-posed without letting drilling dominate.
    dNr0, dNs0 = dNr_c, dNs_c
    detJ0 = J0[1,1] * J0[2,2] - J0[1,2] * J0[2,1]
    inv_det0 = 1.0 / detJ0
    iJ11 = J0[2,2] * inv_det0
    iJ12 = -J0[1,2] * inv_det0
    iJ21 = -J0[2,1] * inv_det0
    iJ22 = J0[1,1] * inv_det0
    b0 = zeros(24)
    for k in 1:4
        dN_dx = iJ11*dNr0[k] + iJ12*dNs0[k]
        dN_dy = iJ21*dNr0[k] + iJ22*dNs0[k]
        idx = (k - 1) * 6
        b0[idx + 1] = -0.5 * dN_dy
        b0[idx + 2] =  0.5 * dN_dx
        b0[idx + 6] = -0.25
    end
    if include_drill_penalty
        k0 = 0.025 * Cm[3, 3]
        @inbounds @fastmath for j in 1:24, i in 1:24
            Ke[i, j] += k0 * area * b0[i] * b0[j]
        end
    end

    return Ke
end

function quad4_hybrid_stress_modes!(P::AbstractMatrix, J0, xi::Float64, eta::Float64)
    fill!(P, 0.0)
    skew_modes = (
        (@SMatrix [1.0 0.0; 0.0 0.0]),
        (@SMatrix [0.0 0.0; 0.0 1.0]),
        (@SMatrix [0.0 1.0; 1.0 0.0]),
        (@SMatrix [eta 0.0; 0.0 0.0]),
        (@SMatrix [0.0 0.0; 0.0 xi]),
    )
    @inbounds for a in 1:5
        sigma_mat = J0 * skew_modes[a] * transpose(J0)
        P[1, a] = sigma_mat[1, 1]
        P[2, a] = sigma_mat[2, 2]
        P[3, a] = sigma_mat[1, 2]
    end
    return P
end

function stiffness_quad4_bending_hybrid_stress_matrices(coords, Cb)
    Ke = zeros(24, 24)
    maximum(abs, Cb) < 1e-30 && return Ke

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))

    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J0 = @SMatrix [
        dNr_c[1]*coords[1,1] + dNr_c[2]*coords[2,1] + dNr_c[3]*coords[3,1] + dNr_c[4]*coords[4,1]  dNr_c[1]*coords[1,2] + dNr_c[2]*coords[2,2] + dNr_c[3]*coords[3,2] + dNr_c[4]*coords[4,2];
        dNs_c[1]*coords[1,1] + dNs_c[2]*coords[2,1] + dNs_c[3]*coords[3,1] + dNs_c[4]*coords[4,1]  dNs_c[1]*coords[1,2] + dNs_c[2]*coords[2,2] + dNs_c[3]*coords[3,2] + dNs_c[4]*coords[4,2]
    ]

    S_inv = inv(Cb)
    H = zeros(5, 5)
    G = zeros(5, 24)
    Bb = zeros(3, 24)
    P = zeros(3, 5)
    tmp3x24 = zeros(3, 24)
    tmp3x5 = zeros(3, 5)

    @inbounds for gp in gauss_pts
        xi, eta = gp[1], gp[2]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        abs_detJ = max(abs(detJ), 1e-12)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(Bb, 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            idx = (k - 1) * 6
            Bb[1, idx + 5] = dN_dx
            Bb[2, idx + 4] = -dN_dy
            Bb[3, idx + 5] = dN_dy
            Bb[3, idx + 4] = -dN_dx
        end

        quad4_hybrid_stress_modes!(P, J0, xi, eta)
        ts_mul!(tmp3x5, S_inv, P)
        ts_mul_At_add!(H, P, tmp3x5, abs_detJ)
        ts_mul_At_add!(G, P, Bb, abs_detJ)
    end

    Ke .+= transpose(G) * (H \ G)
    return Ke
end

function stiffness_quad4_membrane_bending_hybrid_stress_matrices(coords, Cm, Cb, Bmb=nothing)
    C6 = zeros(6, 6)
    @inbounds for j in 1:3, i in 1:3
        C6[i, j] = Cm[i, j]
        C6[i + 3, j + 3] = Cb[i, j]
        if Bmb !== nothing
            C6[i, j + 3] = Bmb[i, j]
            C6[j + 3, i] = Bmb[i, j]
        end
    end
    maximum(abs, C6) < 1e-30 && return zeros(24, 24)

    S_inv = inv(C6)
    Ke = zeros(24, 24)
    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))

    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J0 = @SMatrix [
        dNr_c[1]*coords[1,1] + dNr_c[2]*coords[2,1] + dNr_c[3]*coords[3,1] + dNr_c[4]*coords[4,1]  dNr_c[1]*coords[1,2] + dNr_c[2]*coords[2,2] + dNr_c[3]*coords[3,2] + dNr_c[4]*coords[4,2];
        dNs_c[1]*coords[1,1] + dNs_c[2]*coords[2,1] + dNs_c[3]*coords[3,1] + dNs_c[4]*coords[4,1]  dNs_c[1]*coords[1,2] + dNs_c[2]*coords[2,2] + dNs_c[3]*coords[3,2] + dNs_c[4]*coords[4,2]
    ]

    H = zeros(10, 10)
    G = zeros(10, 24)
    B6 = zeros(6, 24)
    P3 = zeros(3, 5)
    P6 = zeros(6, 10)
    tmp6x10 = zeros(6, 10)

    @inbounds for gp in gauss_pts
        xi, eta = gp[1], gp[2]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        abs_detJ = max(abs(detJ), 1e-12)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(B6, 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            idx = (k - 1) * 6
            B6[1, idx + 1] = dN_dx
            B6[2, idx + 2] = dN_dy
            B6[3, idx + 1] = dN_dy
            B6[3, idx + 2] = dN_dx
            B6[4, idx + 5] = dN_dx
            B6[5, idx + 4] = -dN_dy
            B6[6, idx + 5] = dN_dy
            B6[6, idx + 4] = -dN_dx
        end

        fill!(P6, 0.0)
        quad4_hybrid_stress_modes!(P3, J0, xi, eta)
        for a in 1:5
            for r in 1:3
                P6[r, a] = P3[r, a]
                P6[r + 3, a + 5] = P3[r, a]
            end
        end
        ts_mul!(tmp6x10, S_inv, P6)
        ts_mul_At_add!(H, P6, tmp6x10, abs_detJ)
        ts_mul_At_add!(G, P6, B6, abs_detJ)
    end

    Ke .+= transpose(G) * (H \ G)
    return Ke
end

function stiffness_quad4_mitc_shear_drill_matrices(
    coords,
    Cm,
    Cb,
    Cs,
    h,
    E_ref;
    k6rot=100.0,
    drill_scale::Float64=1.0,
    shear_center_only::Bool=false,
    no_phi2::Bool=true,
)
    Ke = zeros(24, 24)
    maximum(abs, Cs) < 1e-30 && maximum(abs, Cm) < 1e-30 && return Ke

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
    tying_pts = (SVector(0.0, -1.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(1.0, 0.0))
    Bs_tp = zeros(4, 24)
    Bs_row = zeros(24)

    @inbounds for tp_idx in 1:4
        xi_tp, eta_tp = tying_pts[tp_idx][1], tying_pts[tp_idx][2]
        dNr, dNs = shape_derivs_quad(xi_tp, eta_tp)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        N_tp = SVector(
            0.25*(1.0 - xi_tp)*(1.0 - eta_tp),
            0.25*(1.0 + xi_tp)*(1.0 - eta_tp),
            0.25*(1.0 + xi_tp)*(1.0 + eta_tp),
            0.25*(1.0 - xi_tp)*(1.0 + eta_tp),
        )
        fill!(Bs_row, 0.0)
        if tp_idx <= 2
            for k in 1:4
                idx = (k - 1) * 6
                Bs_row[idx + 3] = dNr[k]
                Bs_row[idx + 4] = -J12 * N_tp[k]
                Bs_row[idx + 5] =  J11 * N_tp[k]
            end
        else
            for k in 1:4
                idx = (k - 1) * 6
                Bs_row[idx + 3] = dNs[k]
                Bs_row[idx + 4] = -J22 * N_tp[k]
                Bs_row[idx + 5] =  J21 * N_tp[k]
            end
        end
        @views copyto!(Bs_tp[tp_idx, :], Bs_row)
    end

    dNr_c = SVector(-0.25, 0.25, 0.25, -0.25)
    dNs_c = SVector(-0.25, -0.25, 0.25, 0.25)
    J11c = dNr_c[1]*coords[1,1]+dNr_c[2]*coords[2,1]+dNr_c[3]*coords[3,1]+dNr_c[4]*coords[4,1]
    J12c = dNr_c[1]*coords[1,2]+dNr_c[2]*coords[2,2]+dNr_c[3]*coords[3,2]+dNr_c[4]*coords[4,2]
    J21c = dNs_c[1]*coords[1,1]+dNs_c[2]*coords[2,1]+dNs_c[3]*coords[3,1]+dNs_c[4]*coords[4,1]
    J22c = dNs_c[1]*coords[1,2]+dNs_c[2]*coords[2,2]+dNs_c[3]*coords[3,2]+dNs_c[4]*coords[4,2]
    detJc = J11c*J22c - J12c*J21c
    abs_detJc = max(abs(detJc), 1e-12)

    phi2_shear = 1.0
    if !shear_center_only && !no_phi2 && PHI2_ALPHA[] > 0.0 && maximum(abs, Cb) > 1e-30
        L_char_sq = max(4.0 * abs_detJc, 1e-30)
        phi2_shear = min(1.0, PHI2_ALPHA[] * h^2 / L_char_sq)
    end

    G_drill = Cm[3,3] / h
    if G_drill < 1e-6
        G_drill = E_ref / (2 * 3.0)
    end
    alpha_drill = drill_scale * (k6rot / 1e5) * G_drill * h

    Bs_cov = zeros(2, 24)
    Cs_cov = zeros(2, 2)
    tmp2x24 = zeros(2, 24)
    Bd = zeros(1, 24)

    function add_shear_at!(r::Float64, s::Float64, weight::Float64)
        dNr, dNs = shape_derivs_quad(r, s)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        abs_detJ = max(abs(detJ), 1e-12)
        inv_det = 1.0 / detJ
        iJ11 = J22*inv_det
        iJ12 = -J12*inv_det
        iJ21 = -J21*inv_det
        iJ22 = J11*inv_det

        w_eta_p = 0.5*(1.0+s)
        w_eta_m = 0.5*(1.0-s)
        w_xi_p = 0.5*(1.0+r)
        w_xi_m = 0.5*(1.0-r)
        fill!(Bs_cov, 0.0)
        for j in 1:24
            Bs_cov[1,j] = w_eta_m*Bs_tp[1,j] + w_eta_p*Bs_tp[2,j]
            Bs_cov[2,j] = w_xi_m*Bs_tp[3,j] + w_xi_p*Bs_tp[4,j]
        end

        t11 = Cs[1,1]*iJ11 + Cs[1,2]*iJ21
        t12 = Cs[1,1]*iJ12 + Cs[1,2]*iJ22
        t21 = Cs[2,1]*iJ11 + Cs[2,2]*iJ21
        t22 = Cs[2,1]*iJ12 + Cs[2,2]*iJ22
        Cs_cov[1,1] = phi2_shear*(iJ11*t11 + iJ21*t21)
        Cs_cov[1,2] = phi2_shear*(iJ11*t12 + iJ21*t22)
        Cs_cov[2,1] = phi2_shear*(iJ12*t11 + iJ22*t21)
        Cs_cov[2,2] = phi2_shear*(iJ12*t12 + iJ22*t22)
        ts_mul!(tmp2x24, Cs_cov, Bs_cov)
        ts_mul_At_add!(Ke, Bs_cov, tmp2x24, weight * abs_detJ)
        return nothing
    end

    if maximum(abs, Cs) >= 1e-30
        if shear_center_only
            add_shear_at!(0.0, 0.0, 4.0)
        else
            for gp in gauss_pts
                add_shear_at!(gp[1], gp[2], 1.0)
            end
        end
    end

    if alpha_drill != 0.0
        @inbounds for gp in gauss_pts
            r, s = gp[1], gp[2]
            dNr, dNs = shape_derivs_quad(r, s)
            J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
            J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
            J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
            J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
            detJ = J11*J22 - J12*J21
            abs_detJ = max(abs(detJ), 1e-12)
            inv_det = 1.0 / detJ
            iJ11 = J22*inv_det
            iJ12 = -J12*inv_det
            iJ21 = -J21*inv_det
            iJ22 = J11*inv_det
            fill!(Bd, 0.0)
            for k in 1:4
                dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
                dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
                N_k = 0.25*(1 + (k==2||k==3 ? r : -r))*(1 + (k>=3 ? s : -s))
                idx = (k - 1) * 6
                Bd[1, idx + 1] = 0.5*dN_dy
                Bd[1, idx + 2] = -0.5*dN_dx
                Bd[1, idx + 6] = N_k
            end
            ts_mul_At_add!(Ke, Bd, Bd, abs_detJ * alpha_drill)
        end
    end

    return Ke
end

function stiffness_quad4_shear_hybrid_stress_matrices(coords, Cs)
    Ke = zeros(24, 24)
    maximum(abs, Cs) < 1e-30 && return Ke

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))

    area = 0.0
    xi_moment = 0.0
    eta_moment = 0.0
    @inbounds for gp in gauss_pts
        xi, eta = gp[1], gp[2]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        dA = max(abs(J11*J22 - J12*J21), 1e-12)
        area += dA
        xi_moment += xi * dA
        eta_moment += eta * dA
    end
    xi_bar = xi_moment / max(area, 1e-30)
    eta_bar = eta_moment / max(area, 1e-30)

    S_inv = inv(Cs)
    H = zeros(4, 4)
    G = zeros(4, 24)
    Bs = zeros(2, 24)
    P = zeros(2, 4)
    tmp2x4 = zeros(2, 4)

    @inbounds for gp in gauss_pts
        xi, eta = gp[1], gp[2]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        abs_detJ = max(abs(detJ), 1e-12)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(Bs, 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            N_k = 0.25*(1 + (k==2||k==3 ? xi : -xi))*(1 + (k>=3 ? eta : -eta))
            idx = (k - 1) * 6
            Bs[1, idx + 3] = dN_dx
            Bs[1, idx + 5] = N_k
            Bs[2, idx + 3] = dN_dy
            Bs[2, idx + 4] = -N_k
        end

        fill!(P, 0.0)
        P[1, 1] = 1.0
        P[2, 2] = 1.0
        P[1, 3] = eta - eta_bar
        P[2, 4] = xi - xi_bar
        ts_mul!(tmp2x4, S_inv, P)
        ts_mul_At_add!(H, P, tmp2x4, abs_detJ)
        ts_mul_At_add!(G, P, Bs, abs_detJ)
    end

    Ke .+= transpose(G) * (H \ G)
    return Ke
end

@inline function quad4_hw_shape_factor(coords)
    J11, J12, J21, J22 = quad4_center_jacobian_entries(coords)
    g11 = J11*J11 + J12*J12
    g12 = J11*J21 + J12*J22
    g22 = J21*J21 + J22*J22
    tr = g11 + g22
    disc = sqrt(max((g11 - g22)^2 + 4.0*g12^2, 0.0))
    lam_min = max(0.5 * (tr - disc), 1e-30)
    lam_max = max(0.5 * (tr + disc), lam_min)
    return sqrt(lam_max / lam_min)
end

@inline function quad4_center_jacobian_entries(coords)
    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J11 = dNr_c[1]*coords[1,1] + dNr_c[2]*coords[2,1] + dNr_c[3]*coords[3,1] + dNr_c[4]*coords[4,1]
    J12 = dNr_c[1]*coords[1,2] + dNr_c[2]*coords[2,2] + dNr_c[3]*coords[3,2] + dNr_c[4]*coords[4,2]
    J21 = dNs_c[1]*coords[1,1] + dNs_c[2]*coords[2,1] + dNs_c[3]*coords[3,1] + dNs_c[4]*coords[4,1]
    J22 = dNs_c[1]*coords[1,2] + dNs_c[2]*coords[2,2] + dNs_c[3]*coords[3,2] + dNs_c[4]*coords[4,2]
    return J11, J12, J21, J22
end

@inline function quad4_hw_T0(J11::Float64, J12::Float64, J21::Float64, J22::Float64, a::Float64, b::Float64)
    return @SMatrix [
        J11*J11       J21*J21       a*J11*J21;
        J12*J12       J22*J22       a*J12*J22;
        b*J11*J12     b*J21*J22     J11*J22 + J12*J21
    ]
end

@inline function quad4_hw_Ttilde(J11::Float64, J12::Float64, J21::Float64, J22::Float64)
    return @SMatrix [
        J11 J21;
        J12 J22
    ]
end

@inline function quad4_constitutive_looks_isotropic(Cm, Cb, Cs, Bmb)
    Bmb !== nothing && maximum(abs, Bmb) > 1e-10 && return false
    cm_scale = max(maximum(abs, Cm), 1e-30)
    cb_scale = max(maximum(abs, Cb), 1e-30)
    cs_scale = max(maximum(abs, Cs), 1e-30)
    tol = 1e-6
    cm_iso =
        abs(Cm[1,1] - Cm[2,2]) <= tol * cm_scale &&
        abs(Cm[1,3]) <= tol * cm_scale &&
        abs(Cm[2,3]) <= tol * cm_scale &&
        abs(Cm[3,3] - 0.5*(Cm[1,1] - Cm[1,2])) <= 1e-5 * cm_scale
    cb_iso =
        cb_scale <= 1e-20 ||
        (
            abs(Cb[1,1] - Cb[2,2]) <= tol * cb_scale &&
            abs(Cb[1,3]) <= tol * cb_scale &&
            abs(Cb[2,3]) <= tol * cb_scale &&
            abs(Cb[3,3] - 0.5*(Cb[1,1] - Cb[1,2])) <= 1e-5 * cb_scale
        )
    cs_iso =
        abs(Cs[1,1] - Cs[2,2]) <= tol * cs_scale &&
        abs(Cs[1,2]) <= tol * cs_scale &&
        abs(Cs[2,1]) <= tol * cs_scale
    return cm_iso && cb_iso && cs_iso
end

@inline function quad4_coords3d_is_planar(coords_3d::AbstractMatrix)
    p1 = SVector{3,Float64}(coords_3d[1,1], coords_3d[1,2], coords_3d[1,3])
    p2 = SVector{3,Float64}(coords_3d[2,1], coords_3d[2,2], coords_3d[2,3])
    p3 = SVector{3,Float64}(coords_3d[3,1], coords_3d[3,2], coords_3d[3,3])
    p4 = SVector{3,Float64}(coords_3d[4,1], coords_3d[4,2], coords_3d[4,3])
    d21 = p2 - p1
    d31 = p3 - p1
    d41 = p4 - p1
    n = cross(d21, d31)
    nrm = norm(n)
    if nrm < 1e-24
        n = cross(d21, d41)
        nrm = norm(n)
    end
    nrm < 1e-24 && return true
    max_dev = max(abs(dot(d21, n)), abs(dot(d31, n)), abs(dot(d41, n))) / nrm
    L = max(norm(d21), norm(d31), norm(d41), 1e-12)
    return max_dev <= 1e-6 * L
end

function stiffness_quad4_huwashizu_full_matrices(
    coords,
    Cm,
    Cb,
    Cs,
    Bmb=nothing;
    n_extra::Int=11,
    k_extra::Int=0,
)
    C8 = zeros(8, 8)
    @inbounds for j in 1:3, i in 1:3
        C8[i, j] = Cm[i, j]
        C8[i + 3, j + 3] = Cb[i, j]
        if Bmb !== nothing
            C8[i, j + 3] = Bmb[i, j]
            C8[j + 3, i] = Bmb[i, j]
        end
    end
    @inbounds for j in 1:2, i in 1:2
        C8[i + 6, j + 6] = Cs[i, j]
    end
    maximum(abs, C8) < 1e-30 && return zeros(24, 24)

    n_use = clamp(n_extra, 0, 11)
    k_use = clamp(k_extra, 0, 6)
    nstrain = 14 + n_use + k_use
    hw_b_shear_mode = lowercase(strip(get(ENV, "JFEM_HW_B_SHEAR", "direct")))
    use_mitc_cov_shear = hw_b_shear_mode in ("mitc", "mitc_cov", "ans", "covariant")

    pt3 = sqrt(3.0 / 5.0)
    gauss_pts = (
        SVector(-pt3, -pt3, 25.0/81.0), SVector(0.0, -pt3, 40.0/81.0), SVector(pt3, -pt3, 25.0/81.0),
        SVector(-pt3,  0.0, 40.0/81.0), SVector(0.0,  0.0, 64.0/81.0), SVector(pt3,  0.0, 40.0/81.0),
        SVector(-pt3,  pt3, 25.0/81.0), SVector(0.0,  pt3, 40.0/81.0), SVector(pt3,  pt3, 25.0/81.0),
    )

    area = 0.0
    xi_moment = 0.0
    eta_moment = 0.0
    @inbounds for gp in gauss_pts
        xi, eta, wgt = gp[1], gp[2], gp[3]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        dA = wgt * max(abs(J11*J22 - J12*J21), 1e-12)
        area += dA
        xi_moment += xi * dA
        eta_moment += eta * dA
    end
    xi_bar = xi_moment / max(area, 1e-30)
    eta_bar = eta_moment / max(area, 1e-30)
    cshape = quad4_hw_shape_factor(coords)
    J11_0, J12_0, J21_0, J22_0 = quad4_center_jacobian_entries(coords)
    j0 = max(abs(J11_0*J22_0 - J12_0*J21_0), 1e-12)
    T_sigma = quad4_hw_T0(Float64(J11_0), Float64(J12_0), Float64(J21_0), Float64(J22_0), 2.0, 1.0)
    T_epsilon = quad4_hw_T0(Float64(J11_0), Float64(J12_0), Float64(J21_0), Float64(J22_0), 1.0, 2.0)
    T_tilde = quad4_hw_Ttilde(Float64(J11_0), Float64(J12_0), Float64(J21_0), Float64(J22_0))

    H = zeros(nstrain, nstrain)
    F = zeros(nstrain, 14)
    G = zeros(14, 24)
    B8 = zeros(8, 24)
    Nsig = zeros(8, 14)
    Neps = zeros(8, nstrain)
    tmp8xn = zeros(8, nstrain)
    Bs_tp_hw = zeros(4, 24)
    if use_mitc_cov_shear
        tying_pts = (SVector(0.0, -1.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(1.0, 0.0))
        Bs_row_hw = zeros(24)
        @inbounds for tp_idx in 1:4
            xi_tp, eta_tp = tying_pts[tp_idx][1], tying_pts[tp_idx][2]
            dNr_tp, dNs_tp = shape_derivs_quad(xi_tp, eta_tp)
            J11_tp = dNr_tp[1]*coords[1,1] + dNr_tp[2]*coords[2,1] + dNr_tp[3]*coords[3,1] + dNr_tp[4]*coords[4,1]
            J12_tp = dNr_tp[1]*coords[1,2] + dNr_tp[2]*coords[2,2] + dNr_tp[3]*coords[3,2] + dNr_tp[4]*coords[4,2]
            J21_tp = dNs_tp[1]*coords[1,1] + dNs_tp[2]*coords[2,1] + dNs_tp[3]*coords[3,1] + dNs_tp[4]*coords[4,1]
            J22_tp = dNs_tp[1]*coords[1,2] + dNs_tp[2]*coords[2,2] + dNs_tp[3]*coords[3,2] + dNs_tp[4]*coords[4,2]
            N_tp = SVector(
                0.25*(1.0 - xi_tp)*(1.0 - eta_tp),
                0.25*(1.0 + xi_tp)*(1.0 - eta_tp),
                0.25*(1.0 + xi_tp)*(1.0 + eta_tp),
                0.25*(1.0 - xi_tp)*(1.0 + eta_tp),
            )
            fill!(Bs_row_hw, 0.0)
            if tp_idx <= 2
                for k in 1:4
                    idx = (k - 1) * 6
                    Bs_row_hw[idx + 3] = dNr_tp[k]
                    Bs_row_hw[idx + 4] = -J12_tp * N_tp[k]
                    Bs_row_hw[idx + 5] =  J11_tp * N_tp[k]
                end
            else
                for k in 1:4
                    idx = (k - 1) * 6
                    Bs_row_hw[idx + 3] = dNs_tp[k]
                    Bs_row_hw[idx + 4] = -J22_tp * N_tp[k]
                    Bs_row_hw[idx + 5] =  J21_tp * N_tp[k]
                end
            end
            @views copyto!(Bs_tp_hw[tp_idx, :], Bs_row_hw)
        end
    end

    @inbounds for gp in gauss_pts
        xi, eta, wgt = gp[1], gp[2], gp[3]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        dA = wgt * max(abs(detJ), 1e-12)
        jscale = j0 / max(abs(detJ), 1e-12)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(B8, 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            N_k = 0.25*(1 + (k==2||k==3 ? xi : -xi))*(1 + (k>=3 ? eta : -eta))
            idx = (k - 1) * 6
            B8[1, idx + 1] = dN_dx
            B8[2, idx + 2] = dN_dy
            B8[3, idx + 1] = dN_dy
            B8[3, idx + 2] = dN_dx
            B8[4, idx + 5] = dN_dx
            B8[5, idx + 4] = -dN_dy
            B8[6, idx + 5] = dN_dy
            B8[6, idx + 4] = -dN_dx
            B8[7, idx + 3] = dN_dx
            B8[7, idx + 5] = N_k
            B8[8, idx + 3] = dN_dy
            B8[8, idx + 4] = -N_k
        end
        if use_mitc_cov_shear
            w_eta_p = 0.5*(1.0 + eta)
            w_eta_m = 0.5*(1.0 - eta)
            w_xi_p = 0.5*(1.0 + xi)
            w_xi_m = 0.5*(1.0 - xi)
            for j in 1:24
                B8[7, j] = w_eta_m*Bs_tp_hw[1, j] + w_eta_p*Bs_tp_hw[2, j]
                B8[8, j] = w_xi_m*Bs_tp_hw[3, j] + w_xi_p*Bs_tp_hw[4, j]
            end
        end

        fill!(Nsig, 0.0)
        for i in 1:8
            Nsig[i, i] = 1.0
        end
        Nmb_sigma_in = @SMatrix [
            eta - eta_bar  0.0;
            0.0            xi - xi_bar;
            0.0            0.0
        ]
        Ns_sigma_in = @SMatrix [
            eta - eta_bar  0.0;
            0.0            xi - xi_bar
        ]
        Nmb_sigma = T_sigma * Nmb_sigma_in
        Ns_sigma = T_tilde * Ns_sigma_in
        for a in 1:2
            for r0 in 1:3
                Nsig[r0, 8 + a] = Nmb_sigma[r0, a]
                Nsig[r0 + 3, 10 + a] = Nmb_sigma[r0, a]
            end
            for r0 in 1:2
                Nsig[r0 + 6, 12 + a] = Ns_sigma[r0, a]
            end
        end

        fill!(Neps, 0.0)
        for i in 1:8
            Neps[i, i] = 1.0
        end
        Nmb_epsilon = T_epsilon * Nmb_sigma_in
        Ns_epsilon = T_tilde * Ns_sigma_in
        for a in 1:2
            for r0 in 1:3
                Neps[r0, 8 + a] = Nmb_epsilon[r0, a]
                Neps[r0 + 3, 10 + a] = Nmb_epsilon[r0, a]
            end
            for r0 in 1:2
                Neps[r0 + 6, 12 + a] = Ns_epsilon[r0, a]
            end
        end
        col = 15
        if n_use > 0
            Mm_n = @SMatrix [
                xi  0.0 0.0 0.0 xi*eta 0.0    0.0    (xi^2 - cshape)*eta 0.0                       eta^2*xi 0.0;
                0.0 eta 0.0 0.0 0.0    xi*eta 0.0    0.0                    (eta^2 - cshape)*xi 0.0      xi^2*eta;
                0.0 0.0 xi  eta 0.0    0.0    xi*eta 0.0                    0.0                       0.0      0.0
            ]
            Mm_enriched = jscale .* (T_epsilon * Mm_n)
            for a in 1:n_use
                Neps[1, col] = Mm_enriched[1, a]
                Neps[2, col] = Mm_enriched[2, a]
                Neps[3, col] = Mm_enriched[3, a]
                col += 1
            end
        end
        if k_use > 0
            Mb_k = @SMatrix [
                xi  0.0 xi*eta 0.0    xi^2*eta 0.0;
                0.0 eta 0.0    xi*eta 0.0      eta^2*xi;
                0.0 0.0 0.0    0.0    0.0      0.0
            ]
            Mb_enriched = jscale .* (T_epsilon * Mb_k)
            for a in 1:k_use
                Neps[4, col] = Mb_enriched[1, a]
                Neps[5, col] = Mb_enriched[2, a]
                Neps[6, col] = Mb_enriched[3, a]
                col += 1
            end
        end

        ts_mul!(tmp8xn, C8, Neps)
        ts_mul_At_add!(H, Neps, tmp8xn, dA)
        ts_mul_At_add!(F, Neps, Nsig, -dA)
        ts_mul_At_add!(G, Nsig, B8, dA)
    end

    S_eff = transpose(F) * (H \ F)
    return transpose(G) * (S_eff \ G)
end

function stiffness_quad4_huwashizu_matrices(
    coords,
    Cm,
    Cb,
    Cs,
    h,
    E_ref;
    k6rot=100.0,
    drill_scale::Float64=1.0,
    shear_center_only::Bool=false,
    Bmb=nothing,
)
    hw_form = lowercase(strip(get(ENV, "JFEM_HW_FORM", "full")))
    hw_shear_mode = lowercase(strip(get(ENV, "JFEM_HW_SHEAR", "hybrid")))
    if hw_form in ("full", "wg", "wagner")
        n_raw = tryparse(Int, strip(get(ENV, "JFEM_HW_N", "11")))
        k_raw = tryparse(Int, strip(get(ENV, "JFEM_HW_K", "0")))
        Ke = stiffness_quad4_huwashizu_full_matrices(
            coords,
            Cm,
            Cb,
            Cs,
            Bmb;
            n_extra=n_raw === nothing ? 11 : n_raw,
            k_extra=k_raw === nothing ? 0 : k_raw,
        )
        Ke .+= stiffness_quad4_mitc_shear_drill_matrices(
            coords,
            Cm,
            Cb,
            zeros(eltype(Cs), 2, 2),
            h,
            E_ref;
            k6rot=k6rot,
            drill_scale=drill_scale,
            shear_center_only=shear_center_only,
            no_phi2=true,
        )
    else
        Ke = stiffness_quad4_membrane_bending_hybrid_stress_matrices(coords, Cm, Cb, Bmb)
        if hw_shear_mode in ("hybrid", "stress", "huwashizu")
            Ke .+= stiffness_quad4_shear_hybrid_stress_matrices(coords, Cs)
            Ke .+= stiffness_quad4_mitc_shear_drill_matrices(
                coords,
                Cm,
                Cb,
                zeros(eltype(Cs), 2, 2),
                h,
                E_ref;
                k6rot=k6rot,
                drill_scale=drill_scale,
                shear_center_only=shear_center_only,
                no_phi2=true,
            )
        else
            Ke .+= stiffness_quad4_mitc_shear_drill_matrices(
                coords,
                Cm,
                Cb,
                Cs,
                h,
                E_ref;
                k6rot=k6rot,
                drill_scale=drill_scale,
                shear_center_only=shear_center_only,
                no_phi2=true,
            )
        end
    end
    return Ke
end

# Performance-optimized QUAD4 stiffness with optional pre-allocated workspace.
# When ws is provided, eliminates ALL heap allocations in the hot loop (~5M saved across model).

function stiffness_quad4_matrices(coords, Cm, Cb, Cs, h, E_ref; bend_ratio=1.0, k6rot=100.0, drill_scale::Float64=1.0, Bmb=nothing, ws::Union{Nothing,Quad4Workspace}=nothing, bending_incomp::Bool=false, shear_center_only::Bool=false, no_phi2::Bool=false, membrane_incomp::Bool=true, curvature_membrane=nothing, membrane_shear_center_row::Bool=false, material_shear_rotation::Float64=0.0, membrane_assumed_mode::Symbol=:none, membrane_incomp_center_jacobian::Bool=false, selective_shear::Bool=false, selective_shear_mode::Symbol=:all, exact_side_shear::Bool=false, exact_side_rotcorr::Bool=false, exact_membrane_operator::Bool=false, exact_membrane_curvature_w_coupling::Bool=false, slope_membrane=nothing, coords_3d::Union{Nothing,AbstractMatrix}=nothing, kernel_planar::Bool=true, macneal_rigid_shear::Bool=false, marguerre_warp_to_uz::Bool=false, min4_disable::Bool=false)
    # Allow env-var override for marguerre_warp_to_uz so it can be enabled
    # globally without plumbing through every caller. Currently the assembly
    # loop doesn't pass this kwarg, so default is false. Env override:
    # JFEM_Q4_MARGUERRE_WARP_TO_UZ=true forces it on for ALL elements where
    # coords_3d is supplied (i.e., wherever the curved-shell path runs).
    # The added term is the Marguerre membrane–uz coupling:
    #   εxx ⊃ z_x · ∂w/∂x, εyy ⊃ z_y · ∂w/∂y, εxy ⊃ z_x·∂w/∂y + z_y·∂w/∂x
    # where z_x, z_y are the element-local-frame slopes of the corner
    # z-coords. Activates only on genuinely warped (non-coplanar) elements.
    if !marguerre_warp_to_uz
        env_raw = strip(get(ENV, "JFEM_Q4_MARGUERRE_WARP_TO_UZ", ""))
        if !isempty(env_raw) && lowercase(env_raw) in ("1", "true", "yes", "on")
            marguerre_warp_to_uz = true
        end
    end
    # coords_3d (optional 4×3 matrix of 3D corner coordinates) activates the
    # experimental curved-shell GP-local frame path used by
    # JFEM_SOL105_EIG_CURVED_JACOBIAN. The path is intentionally narrow:
    # it engages only on the pure formulation branch (no curvature heuristics,
    # Marguerre slopes, membrane-center projection, or exact side shear).
    # That keeps the parity study interpretable while we measure the effect of
    # the 3D tangent mapping plus the geometry-driven membrane -b·w coupling.
    if exact_membrane_operator
        zero_Cb = zeros(eltype(Cb), 3, 3)
        zero_Cs = zeros(eltype(Cs), 2, 2)
        # Replace only the flat shell membrane/drilling block with the
        # paper-based DKMQ24_2+ membrane operator while preserving the
        # validated shell bending/shear field.
        membrane_curvature_default =
            exact_membrane_curvature_w_coupling ? nothing : curvature_membrane
        Ke_shell = stiffness_quad4_matrices(
            coords, Cm, Cb, Cs, h, E_ref;
            bend_ratio=bend_ratio,
            k6rot=k6rot,
            drill_scale=drill_scale,
            Bmb=nothing,
            ws=nothing,
            bending_incomp=bending_incomp,
            shear_center_only=shear_center_only,
            no_phi2=no_phi2,
            membrane_incomp=false,
            curvature_membrane=curvature_membrane,
            membrane_shear_center_row=membrane_shear_center_row,
            material_shear_rotation=material_shear_rotation,
            membrane_assumed_mode=membrane_assumed_mode,
            membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
            selective_shear=selective_shear,
            selective_shear_mode=selective_shear_mode,
            exact_side_shear=exact_side_shear,
            exact_side_rotcorr=exact_side_rotcorr,
            exact_membrane_operator=false,
            exact_membrane_curvature_w_coupling=false,
            coords_3d=coords_3d,
            kernel_planar=kernel_planar,
            macneal_rigid_shear=macneal_rigid_shear,
        )
        Ke_mem_default = stiffness_quad4_matrices(
            coords, Cm, zero_Cb, zero_Cs, h, E_ref;
            bend_ratio=bend_ratio,
            k6rot=0.0,
            drill_scale=0.0,
            Bmb=nothing,
            ws=nothing,
            bending_incomp=false,
            shear_center_only=true,
            no_phi2=true,
            membrane_incomp=false,
            curvature_membrane=membrane_curvature_default,
            membrane_shear_center_row=membrane_shear_center_row,
            material_shear_rotation=material_shear_rotation,
            membrane_assumed_mode=membrane_assumed_mode,
            membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
            selective_shear=false,
            selective_shear_mode=:all,
            exact_side_shear=false,
            exact_side_rotcorr=false,
            exact_membrane_operator=false,
            exact_membrane_curvature_w_coupling=false,
            coords_3d=coords_3d,
            kernel_planar=kernel_planar,
            macneal_rigid_shear=false,
        )
        Ke_mem_exact = stiffness_quad4_membrane_hybrid_stress_matrices(coords, Cm, h)
        return Ke_shell .+ Ke_mem_exact .- Ke_mem_default
    end
    q4_kernel = lowercase(strip(get(ENV, "JFEM_Q4_KERNEL", "")))
    huwashizu_kernel = q4_kernel in ("huwashizu", "hu-washizu", "hw")
    if huwashizu_kernel &&
       curvature_membrane === nothing &&
       slope_membrane === nothing &&
       coords_3d === nothing &&
        (!membrane_shear_center_row || material_shear_rotation == 0.0) &&
       membrane_assumed_mode === :none &&
       !membrane_shear_center_row &&
       !selective_shear &&
       !exact_side_shear &&
       !exact_side_rotcorr
        return stiffness_quad4_huwashizu_matrices(
            coords,
            Cm,
            Cb,
            Cs,
            h,
            E_ref;
            k6rot=k6rot,
            drill_scale=drill_scale,
            shear_center_only=shear_center_only,
            Bmb=Bmb,
        )
    end
    # Tessler-Hughes 1983 MIN4 kernel branch (2026-05-14 evening). Replaces
    # the MacNeal/MITC bending+shear blocks with the anisoparametric MIN4
    # formulation (interpolation: biquadratic w + bilinear θ, continuous
    # edge shear constraint) plus the residual-bending shear correction
    # φ² = C_b·ψ̂/(1+C_b·ψ̂) (eq 4.21). The membrane/drilling blocks come
    # from a recursive call with Cb=Cs=0 (going through the standard
    # MacNeal/MITC path; min4_disable=true breaks recursion).
    # Env vars:
    #   JFEM_Q4_KERNEL=min4 (or tessler_hughes, or tessler-hughes)
    #   JFEM_MIN4_CBMIN4=3.6 (default; from MYSTRAN MIN4 calibration)
    min4_kernel = !min4_disable && q4_kernel in ("min4", "tessler_hughes", "tessler-hughes")
    if min4_kernel
        zero_Cb = zeros(eltype(Cb), 3, 3)
        zero_Cs = zeros(eltype(Cs), 2, 2)
        # Membrane + drilling from the standard kernel (Cb=Cs=0 → no bending/shear)
        Ke_membrane_drill = stiffness_quad4_matrices(
            coords, Cm, zero_Cb, zero_Cs, h, E_ref;
            bend_ratio=bend_ratio,
            k6rot=k6rot,
            drill_scale=drill_scale,
            Bmb=nothing,
            ws=nothing,
            bending_incomp=false,
            shear_center_only=shear_center_only,
            no_phi2=true,
            membrane_incomp=membrane_incomp,
            curvature_membrane=curvature_membrane,
            membrane_shear_center_row=membrane_shear_center_row,
            material_shear_rotation=material_shear_rotation,
            membrane_assumed_mode=membrane_assumed_mode,
            membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
            selective_shear=false,
            selective_shear_mode=:all,
            exact_side_shear=false,
            exact_side_rotcorr=false,
            exact_membrane_operator=false,
            exact_membrane_curvature_w_coupling=false,
            slope_membrane=slope_membrane,
            coords_3d=coords_3d,
            kernel_planar=kernel_planar,
            macneal_rigid_shear=false,
            marguerre_warp_to_uz=false,
            min4_disable=true,
        )
        # MIN4 bending + φ²·shear
        cbmin4_env = strip(get(ENV, "JFEM_MIN4_CBMIN4", ""))
        cbmin4_val = isempty(cbmin4_env) ? 3.6 :
            (something(tryparse(Float64, cbmin4_env), 3.6))
        Ke_bs, _, _, _ = stiffness_quad4_min4_bending_shear(coords, Cb, Cs;
                                                           cbmin4=cbmin4_val)
        return Ke_membrane_drill .+ Ke_bs
    end
    if ws === nothing; ws = create_quad4_workspace(); end

    # Clear accumulated matrices
    fill!(ws.Ke, 0.0)
    fill!(ws.K_ab, 0.0); fill!(ws.K_bb, 0.0)
    fill!(ws.K_ab_bend, 0.0); fill!(ws.K_bb_bend, 0.0)

    # B coupling accumulators (cleared even if Bmb is nothing — branch-free)
    fill!(ws.K_ab_cross, 0.0); fill!(ws.K_ab_bend_cross, 0.0); fill!(ws.K_mb_incomp, 0.0)

    curved_frame_supported =
        coords_3d !== nothing &&
        curvature_membrane === nothing &&
        slope_membrane === nothing &&
        (!membrane_shear_center_row || material_shear_rotation == 0.0) &&
        membrane_assumed_mode === :none &&
        !membrane_shear_center_row &&
        !selective_shear &&
        !exact_side_shear &&
        !exact_side_rotcorr

    elem_v1 = SVector(1.0, 0.0, 0.0)
    elem_v2 = SVector(0.0, 1.0, 0.0)
    elem_v3 = SVector(0.0, 0.0, 1.0)
    center_beta_gp = 0.0
    center_Cs = Cs
    if curved_frame_supported
        elem_v1, elem_v2, elem_v3 = quad4_center_frame_from_coords3d(coords_3d)
    end

    # --- MITC4 transverse shear (Bathe-Dvorkin) tying points ---
    skip_all_shear = shear_center_only && maximum(abs, Cb) < 1e-30
    A_beta_rotcorr = nothing
    edge_L_rotcorr = nothing
    Bs_rotcorr = nothing
    shear_rotation_scale = fem_env_float("JFEM_Q4_SHEAR_ROTATION_SCALE", 1.0)
    if exact_side_rotcorr && !shear_center_only && !skip_all_shear
        A_beta_rotcorr, _, _, edge_L_rotcorr = dkmq_flat_plate_edge_relation(coords, Cb, Cs)
        Bs_rotcorr = zeros(2, 12)
    end
    fill!(ws.Bs_tp, 0.0)
    tying_pts = (SVector(0.0, -1.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(1.0, 0.0))
    for tp_idx in 1:4
        xi_tp, eta_tp = tying_pts[tp_idx][1], tying_pts[tp_idx][2]
        dNr, dNs = shape_derivs_quad(xi_tp, eta_tp)
        n_tp = SVector(0.0, 0.0, 1.0)
        t1_tp = elem_v1
        t2_tp = elem_v2
        curvature_tp = nothing
        if curved_frame_supported
            n_tp, _, t1_tp, t2_tp, J11, J12, J21, J22 =
                quad4_gp_local_frame_from_coords3d(coords_3d, xi_tp, eta_tp)
            curvature_tp = quad4_gp_curvature_membrane_from_coords3d(coords_3d, xi_tp, eta_tp)
        else
            J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
            J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
            J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
            J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        end
        N1 = 0.25*(1-xi_tp)*(1-eta_tp); N2 = 0.25*(1+xi_tp)*(1-eta_tp)
        N3 = 0.25*(1+xi_tp)*(1+eta_tp); N4 = 0.25*(1-xi_tp)*(1+eta_tp)
        N_tp = SVector(N1, N2, N3, N4)
        fill!(ws.Bs_row, 0.0)
        if tp_idx <= 2  # A,B: e_ξz
            for k in 1:4
                idx = (k-1)*6
                if curvature_tp !== nothing
                    ws.Bs_row[idx+1] = N_tp[k] * (J11*curvature_tp[1] + J12*curvature_tp[3])
                    ws.Bs_row[idx+2] = N_tp[k] * (J11*curvature_tp[3] + J12*curvature_tp[2])
                end
                ws.Bs_row[idx+3] = dNr[k]
                ws.Bs_row[idx+4] = -J12*N_tp[k]
                ws.Bs_row[idx+5] =  J11*N_tp[k]
            end
        else  # C,D: e_ηz
            for k in 1:4
                idx = (k-1)*6
                if curvature_tp !== nothing
                    ws.Bs_row[idx+1] = N_tp[k] * (J21*curvature_tp[1] + J22*curvature_tp[3])
                    ws.Bs_row[idx+2] = N_tp[k] * (J21*curvature_tp[3] + J22*curvature_tp[2])
                end
                ws.Bs_row[idx+3] = dNs[k]
                ws.Bs_row[idx+4] = -J22*N_tp[k]
                ws.Bs_row[idx+5] =  J21*N_tp[k]
            end
        end
        if Bs_rotcorr !== nothing
            dkmq_plate_side_shear_operator!(Bs_rotcorr, coords, A_beta_rotcorr, edge_L_rotcorr, xi_tp, eta_tp)
            row_idx = tp_idx <= 2 ? 1 : 2
            for a in 1:4
                col24 = (a - 1) * 6
                col12 = (a - 1) * 3
                ws.Bs_row[col24 + 4] = Bs_rotcorr[row_idx, col12 + 2]
                ws.Bs_row[col24 + 5] = Bs_rotcorr[row_idx, col12 + 3]
            end
        end
        if shear_rotation_scale != 1.0
            for a in 1:4
                col24 = (a - 1) * 6
                ws.Bs_row[col24 + 4] *= shear_rotation_scale
                ws.Bs_row[col24 + 5] *= shear_rotation_scale
            end
        end
        if curved_frame_supported
            quad4_gp_rotation_from_element!(ws.Rel_t, elem_v1, elem_v2, elem_v3, t1_tp, t2_tp, n_tp)
            rotate_quad4_dof_blocks!(ws.Bs_row, ws.Rel_t)
        end
        @views copyto!(ws.Bs_tp[tp_idx, :], ws.Bs_row)
    end

    # phi2 shear anti-locking: phi2 = min(1, alpha*(h/L_char)^2)
    # MITC4 at 2×2 Gauss points still locks for thin plates on coarse meshes (h/L<0.05).
    # phi2 matches Nastran CQUAD4's Selective Reduced Integration behavior.
    # PHI2_ALPHA=10.0 is the globally optimal coefficient across all test cases.
    _alpha = PHI2_ALPHA[]
    phi2_shear = 1.0
    # MacNeal 1978 eq (12): 1/GA* = 1/GA + L²/(12 EI). Series-flexibility
    # correction that makes the element match Nastran CQUAD4 for both
    # long-wavelength (launch) and short-wavelength (3wp) buckling modes.
    # When enabled, this replaces phi2 as the shear softening mechanism.
    macneal_rbf = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_RBF", "false"))) in ("1","true","yes","on")
    # Twist-compatibility correction (MacNeal eq 17): χ̃xy = 2·χxy(gp) − χxy(0).
    # Replaces the twist row of Bb at each Gauss point by 2·row(gp) − row(center).
    # Only engages on the flat default path; disabled with curved_frame_supported.
    macneal_twist_env_set = haskey(ENV, "JFEM_Q4_MACNEAL_TWIST")
    macneal_twist = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_TWIST", "false"))) in ("1","true","yes","on")
    # Full MacNeal 1978 CQUAD4 kernel: replaces MITC4+phi2 shear block with
    # MacNeal's [D]ᵀ·([Z_s]+[Z_b])⁻¹·[D] formulation + twist correction.
    # MSC/Nastran's CQUAD4 lineage uses this operator for the released QUAD4.
    # This implementation is a flat/projected CQUAD4 kernel, so the default
    # applies it to planar elements for all material types and leaves nonflat
    # facets on the covariant/MITC path unless JFEM_Q4_KERNEL=macneal_all is
    # explicitly requested. The older anisotropic-only split remains available
    # as JFEM_Q4_KERNEL=macneal_pcomp; use JFEM_Q4_KERNEL=default (or any
    # unrecognized value) to force the legacy non-MacNeal path.
    q4_kernel_mode = lowercase(strip(get(ENV, "JFEM_Q4_KERNEL", "macneal")))
    macneal_default_kernel = q4_kernel_mode in (
        "macneal", "mitc4_3d_aspect", "mitc4-3d-aspect", "mitc3d_aspect", "mitc3d-aspect",
    )
    macneal_pcomp_kernel = q4_kernel_mode in ("macneal_pcomp", "macneal-pcomp", "macneal_aniso")
    macneal_pcomp_flat_ok = kernel_planar && (coords_3d === nothing || quad4_coords3d_is_planar(coords_3d))
    macneal_kernel =
        q4_kernel_mode in ("macneal_all", "macneal-force", "macneal_force") ||
        (macneal_default_kernel && kernel_planar) ||
        (
            macneal_pcomp_kernel &&
            macneal_pcomp_flat_ok &&
            !quad4_constitutive_looks_isotropic(Cm, Cb, Cs, Bmb)
        )
    if macneal_kernel && !macneal_twist_env_set
        macneal_twist = true  # twist correction is part of full MacNeal kernel
    end
    macneal_rbf_eps = begin
        raw = get(ENV, "JFEM_Q4_MACNEAL_EPSILON", "0.04")
        v = tryparse(Float64, strip(raw))
        (v === nothing || v < 0.0) ? 0.04 : v
    end
    # Center Jacobian — needed for phi2 and/or shear_center_only 1-point integration
    dNr_c = SVector(-0.25, 0.25, 0.25, -0.25)
    dNs_c = SVector(-0.25, -0.25, 0.25, 0.25)
    if curved_frame_supported
        n_c, area_c, t1_c, t2_c, J11c, J12c, J21c, J22c =
            quad4_gp_local_frame_from_coords3d(coords_3d, 0.0, 0.0)
        detJc = J11c*J22c - J12c*J21c
        abs_detJc = max(area_c, 1e-12)
        center_beta_gp = quad4_gp_rotation_from_element!(ws.Rel_t, elem_v1, elem_v2, elem_v3, t1_c, t2_c, n_c)
        if abs(center_beta_gp) > 1e-12
            copyto!(ws.Cs_buf, Cs)
            rotate_constitutive_2x2!(ws.Cs_buf, center_beta_gp)
            center_Cs = ws.Cs_buf
        end
    else
        J11c = dNr_c[1]*coords[1,1]+dNr_c[2]*coords[2,1]+dNr_c[3]*coords[3,1]+dNr_c[4]*coords[4,1]
        J12c = dNr_c[1]*coords[1,2]+dNr_c[2]*coords[2,2]+dNr_c[3]*coords[3,2]+dNr_c[4]*coords[4,2]
        J21c = dNs_c[1]*coords[1,1]+dNs_c[2]*coords[2,1]+dNs_c[3]*coords[3,1]+dNs_c[4]*coords[4,1]
        J22c = dNs_c[1]*coords[1,2]+dNs_c[2]*coords[2,2]+dNs_c[3]*coords[3,2]+dNs_c[4]*coords[4,2]
        detJc = J11c*J22c - J12c*J21c
        abs_detJc = abs(detJc)
    end
    inv_detc = 1.0 / detJc
    iJ11c =  J22c*inv_detc
    iJ12c = -J12c*inv_detc
    iJ21c = -J21c*inv_detc
    iJ22c =  J11c*inv_detc
    dNdx_c = ntuple(k -> iJ11c*dNr_c[k] + iJ12c*dNs_c[k], 4)
    dNdy_c = ntuple(k -> iJ21c*dNr_c[k] + iJ22c*dNs_c[k], 4)
    if !shear_center_only && !no_phi2 && _alpha > 0.0
        L_char_sq = max(4.0 * abs_detJc, 1e-30)  # ≈ element area
        if macneal_rbf
            # MacNeal 1978 eq (12): GA* = GA / (1 + GA·L²/(12·D)). For the
            # isotropic plate bending modulus D = Cb[1,1] and the transverse-
            # shear modulus per unit width GA = Cs[1,1]. The ratio α_rbf
            # replaces φ₂ as the shear-direction softening factor, with the
            # advantage that it is derived from the element's actual bending
            # compliance rather than an empirical h/L scaling.
            D_bend = max(Cb[1,1], 1e-30)
            GA_shear = max(Cs[1,1], 1e-30)
            phi2_shear = 1.0 / (1.0 + GA_shear * L_char_sq / (12.0 * D_bend))
        else
            phi2_shear = min(1.0, _alpha * h^2 / L_char_sq)
        end
    end
    # For membrane-only elements (Cb≈0, bend_ratio=0) assembled with shear_center_only=true:
    # skip all shear so DOF3/4/5 are truly zero → AUTOSPC in eigenvalue solve constrains them,
    # matching Nastran's behavior where AUTOSPC eliminates membrane-only plate out-of-plane DOFs.
    pt = 1.0/sqrt(3.0)
    gauss_pts = (SVector(-pt,-pt), SVector(pt,-pt), SVector(pt,pt), SVector(-pt,pt))

    # Hughes-Brezzi drilling
    G_drill = Cm[3,3] / h
    if G_drill < 1e-6; G_drill = E_ref / (2*3.0); end
    alpha_drill = drill_scale * (k6rot / 1e5) * G_drill * h

    @inbounds @fastmath for gp in 1:4
        r, s = gauss_pts[gp][1], gauss_pts[gp][2]
        dNr, dNs = shape_derivs_quad(r, s)

        n_gp = SVector(0.0, 0.0, 1.0)
        t1_gp = elem_v1
        t2_gp = elem_v2
        if curved_frame_supported
            n_gp, area_elem, t1_gp, t2_gp, J11, J12, J21, J22 =
                quad4_gp_local_frame_from_coords3d(coords_3d, r, s)
            detJ = J11*J22 - J12*J21
            abs_detJ = max(area_elem, 1e-12)
        else
            # Inline Jacobian computation
            J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
            J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
            J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
            J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
            detJ = J11*J22 - J12*J21
            abs_detJ = abs(detJ)
            if abs_detJ < 1e-12; abs_detJ = 1e-12; end
        end
        inv_det = 1.0 / detJ
        iJ11 = J22*inv_det; iJ12 = -J12*inv_det
        iJ21 = -J21*inv_det; iJ22 = J11*inv_det

        curvature_gp = curvature_membrane
        bending_connection_gp = SVector(0.0, 0.0)
        if curved_frame_supported
            curvature_gp = quad4_gp_curvature_membrane_from_coords3d(coords_3d, r, s)
            bending_connection_gp = quad4_gp_bending_connection_from_coords3d(coords_3d, r, s)
        end

        # Marguerre direct-slope-to-uz coupling (2026-05-12). For a non-planar
        # element, the linear-warp slope z₀,α at the GP couples the membrane
        # strain to the transverse displacement uz_k via
        #   ε_xx ⊃ z₀,x · ∂w/∂x = z₀,x · Σ_k dN_k/dx · uz_k
        # The bilinear-Q4 curvature term (curvature_gp) only captures the
        # twist component (∂²z/∂ξ∂η); the linear-slope term fills in the
        # cylindrical-warp coupling that Nastran's CQUAD4 generates and the
        # standard MacNeal kernel misses (see SOL105 parity TODO, 2026-05-12).
        marguerre_z_x_gp = 0.0
        marguerre_z_y_gp = 0.0
        if marguerre_warp_to_uz && coords_3d !== nothing
            z_corners = quad4_local_z_from_coords3d(coords_3d)
            z_xi  = dNr[1]*z_corners[1] + dNr[2]*z_corners[2] + dNr[3]*z_corners[3] + dNr[4]*z_corners[4]
            z_eta = dNs[1]*z_corners[1] + dNs[2]*z_corners[2] + dNs[3]*z_corners[3] + dNs[4]*z_corners[4]
            marguerre_z_x_gp = iJ11*z_xi + iJ12*z_eta
            marguerre_z_y_gp = iJ21*z_xi + iJ22*z_eta
        end

        # Fill Bm, Bb, Bd directly from inline dN/dx, dN/dy
        fill!(ws.Bm, 0.0); fill!(ws.Bb, 0.0); fill!(ws.Bd, 0.0)
        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            idx = (k-1)*6
            N_k = 0.25*(1 + (k==2||k==3 ? r : -r))*(1 + (k>=3 ? s : -s))
            ws.Bm[1, idx+1] = dN_dx;  ws.Bm[2, idx+2] = dN_dy
            ws.Bm[3, idx+1] = dN_dy;  ws.Bm[3, idx+2] = dN_dx
            if curvature_gp !== nothing
                ws.Bm[1, idx+3] = -N_k * curvature_gp[1]
                ws.Bm[2, idx+3] = -N_k * curvature_gp[2]
                ws.Bm[3, idx+3] = -2.0 * N_k * curvature_gp[3]
            end
            if marguerre_warp_to_uz && coords_3d !== nothing
                ws.Bm[1, idx+3] += marguerre_z_x_gp * dN_dx
                ws.Bm[2, idx+3] += marguerre_z_y_gp * dN_dy
                ws.Bm[3, idx+3] += marguerre_z_x_gp * dN_dy + marguerre_z_y_gp * dN_dx
            end
            if curved_frame_supported
                # Naghdi-RM covariant membrane connection:
                # u_{α|β} = ∂u_α/∂x^β − Γ^γ_{αβ} u_γ, with Γ^2_{1β}=η_β
                # and Γ^1_{2β}=−η_β (orthonormality). Mirror of the bending
                # connection already applied to Bb below.
                ws.Bm[1, idx+2] -= N_k * bending_connection_gp[1]
                ws.Bm[2, idx+1] += N_k * bending_connection_gp[2]
                ws.Bm[3, idx+1] += N_k * bending_connection_gp[1]
                ws.Bm[3, idx+2] -= N_k * bending_connection_gp[2]
            end
            if slope_membrane !== nothing
                # Marguerre shallow-shell coupling (Ibrahimbegović 1994 Eq. 6.14).
                # Two conventions selectable via JFEM_Q4_MARGUERRE_CONVENTION:
                #   jfem_kl (default) — ε_xx on θ_y (idx+5), ε_yy on θ_x (idx+4)
                #   handover          — ε_xx on θ_x (idx+4), ε_yy on θ_y (idx+5)
                fx = 0.0; fy = 0.0
                for j in 1:4
                    Nj = 0.25*(1 + (j==2||j==3 ? r : -r))*(1 + (j>=3 ? s : -s))
                    fx += Nj * slope_membrane[2*j-1]
                    fy += Nj * slope_membrane[2*j]
                end
                if length(slope_membrane) >= 9 && slope_membrane[9] != 0.0
                    # Handover convention (θ-axis labeled by tangent direction)
                    ws.Bm[1, idx+4] += N_k * fx
                    ws.Bm[2, idx+5] += N_k * fy
                    ws.Bm[3, idx+4] += N_k * fy
                    ws.Bm[3, idx+5] += N_k * fx
                else
                    # JFEM KL convention (θ_y = ∂w/∂x, θ_x = ∂w/∂y)
                    ws.Bm[1, idx+5] += N_k * fx
                    ws.Bm[2, idx+4] += N_k * fy
                    ws.Bm[3, idx+4] += N_k * fx
                    ws.Bm[3, idx+5] += N_k * fy
                end
            end
            ws.Bb[1, idx+5] = dN_dx;  ws.Bb[2, idx+4] = -dN_dy
            ws.Bb[3, idx+5] = dN_dy;  ws.Bb[3, idx+4] = -dN_dx
            # MacNeal twist compatibility (eq 17): χ̃_xy = 2·χ_xy(GP) − χ_xy(0).
            # Applied only on the flat default path (no curved_frame_supported).
            if macneal_twist && !curved_frame_supported
                ws.Bb[3, idx+5] = 2.0*dN_dy - dNdy_c[k]
                ws.Bb[3, idx+4] = 2.0*(-dN_dx) - (-dNdx_c[k])
            end
            if curved_frame_supported
                # Covariant change-of-curvature in the moving tangent frame:
                # β₁ = θ_y, β₂ = -θ_x, so β_{α|β} adds connection terms to the
                # flat rotation-derivative plate operator.
                ws.Bb[1, idx+4] += N_k * bending_connection_gp[1]
                ws.Bb[2, idx+5] += N_k * bending_connection_gp[2]
                ws.Bb[3, idx+4] += N_k * bending_connection_gp[2]
                ws.Bb[3, idx+5] += N_k * bending_connection_gp[1]
            end
            # Drilling: Bd = [0.5*dN/dy, -0.5*dN/dx, 0, 0, 0, N_k] per node
            ws.Bd[1, idx+1] = 0.5*dN_dy
            ws.Bd[1, idx+2] = -0.5*dN_dx
            ws.Bd[1, idx+6] = N_k
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(
                ws.Bm,
                dNdx_c,
                dNdy_c,
                curvature_membrane,
                material_shear_rotation,
            )
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
            apply_membrane_ans_mitc4plus!(ws.Bm, coords, r, s)
        end

        dphi1_dx = iJ11*(-2.0*r);  dphi1_dy = iJ21*(-2.0*r)
        dphi2_dx = iJ12*(-2.0*s);  dphi2_dy = iJ22*(-2.0*s)

        # Fill Bi (membrane incompatible)
        fill_quad4_membrane_incompatible_B!(
            ws.Bi,
            r,
            s,
            iJ11,
            iJ12,
            iJ21,
            iJ22,
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            membrane_incomp_center_jacobian,
        )

        # Fill Bi_bend (bending incompatible)
        fill!(ws.Bi_bend, 0.0)
        ws.Bi_bend[2,1]=-dphi1_dy; ws.Bi_bend[2,2]=-dphi2_dy
        ws.Bi_bend[1,3]=dphi1_dx;  ws.Bi_bend[1,4]=dphi2_dx
        ws.Bi_bend[3,1]=-dphi1_dx; ws.Bi_bend[3,2]=-dphi2_dx
        ws.Bi_bend[3,3]=dphi1_dy;  ws.Bi_bend[3,4]=dphi2_dy

        Cm_use = Cm
        Cb_use = Cb
        Cs_use = Cs
        Bmb_use = Bmb
        if curved_frame_supported
            beta_gp = quad4_gp_rotation_from_element!(ws.Rel_t, elem_v1, elem_v2, elem_v3, t1_gp, t2_gp, n_gp)
            rotate_quad4_dof_blocks!(ws.Bm, ws.Rel_t)
            rotate_quad4_dof_blocks!(ws.Bb, ws.Rel_t)
            rotate_quad4_dof_blocks!(ws.Bd, ws.Rel_t)
            if abs(beta_gp) > 1e-12
                copyto!(ws.Cm_buf, Cm)
                copyto!(ws.Cb_buf, Cb)
                copyto!(ws.Cs_buf, Cs)
                rotate_constitutive_3x3!(ws.Cm_buf, beta_gp)
                rotate_constitutive_3x3!(ws.Cb_buf, beta_gp)
                rotate_constitutive_2x2!(ws.Cs_buf, beta_gp)
                Cm_use = ws.Cm_buf
                Cb_use = ws.Cb_buf
                Cs_use = ws.Cs_buf
                if Bmb !== nothing
                    copyto!(ws.Bmb_buf, Bmb)
                    rotate_constitutive_3x3!(ws.Bmb_buf, beta_gp)
                    Bmb_use = ws.Bmb_buf
                end
            end
        end

        # === In-place stiffness accumulation (thread-safe, BLAS-free) ===
        # Ke += abs_detJ * Bm' * Cm * Bm (membrane: 2×2 full integration)
        ts_mul!(ws.tmp3x24, Cm_use, ws.Bm)
        ts_mul_At_add!(ws.Ke, ws.Bm, ws.tmp3x24, abs_detJ)
        # Ke += abs_detJ * Bb' * Cb * Bb (bending: 2×2 full integration)
        ts_mul!(ws.tmp3x24, Cb_use, ws.Bb)
        ts_mul_At_add!(ws.Ke, ws.Bb, ws.tmp3x24, abs_detJ)
        # Shear: MITC4 (Bathe-Dvorkin) 2×2 full integration — locking-free by construction.
        # Skipped when JFEM_Q4_KERNEL=macneal (MacNeal RBF shear is added after the GP loop).
        if !shear_center_only && !selective_shear && !exact_side_shear && !macneal_kernel
            w_eta_p = 0.5*(1.0+s); w_eta_m = 0.5*(1.0-s)
            w_xi_p  = 0.5*(1.0+r); w_xi_m  = 0.5*(1.0-r)
            fill!(ws.Bs_cov, 0.0)
            for j in 1:24
                ws.Bs_cov[1,j] = w_eta_m*ws.Bs_tp[1,j] + w_eta_p*ws.Bs_tp[2,j]  # e_ξz
                ws.Bs_cov[2,j] = w_xi_m*ws.Bs_tp[3,j]  + w_xi_p*ws.Bs_tp[4,j]   # e_ηz
            end
            # Cs_cov = phi2 * J^{-T} Cs J^{-1} (covariant basis + phi2 anti-locking)
            t11 = Cs_use[1,1]*iJ11 + Cs_use[1,2]*iJ21; t12 = Cs_use[1,1]*iJ12 + Cs_use[1,2]*iJ22
            t21 = Cs_use[2,1]*iJ11 + Cs_use[2,2]*iJ21; t22 = Cs_use[2,1]*iJ12 + Cs_use[2,2]*iJ22
            ws.tmp2x2[1,1] = phi2_shear*(iJ11*t11 + iJ21*t21); ws.tmp2x2[1,2] = phi2_shear*(iJ11*t12 + iJ21*t22)
            ws.tmp2x2[2,1] = phi2_shear*(iJ12*t11 + iJ22*t21); ws.tmp2x2[2,2] = phi2_shear*(iJ12*t12 + iJ22*t22)
            # Ke += abs_detJ * Bs_cov' * Cs_cov * Bs_cov
            ts_mul!(ws.tmp2x24, ws.tmp2x2, ws.Bs_cov)
            ts_mul_At_add!(ws.Ke, ws.Bs_cov, ws.tmp2x24, abs_detJ)
        end
        # Ke += abs_detJ * alpha_drill * Bd' * Bd
        ts_mul_At_add!(ws.Ke, ws.Bd, ws.Bd, abs_detJ * alpha_drill)

        # B matrix coupling: Ke += abs_detJ * (Bm'*B*Bb + Bb'*B*Bm)
        if Bmb_use !== nothing
            ts_mul!(ws.tmp3x24, Bmb_use, ws.Bb)
            ts_mul_At_add!(ws.Ke, ws.Bm, ws.tmp3x24, abs_detJ)
            ts_mul!(ws.tmp3x24, Bmb_use, ws.Bm)
            ts_mul_At_add!(ws.Ke, ws.Bb, ws.tmp3x24, abs_detJ)
        end

        # Incompatible mode coupling: K_ab += abs_detJ * Bm' * Cm * Bi
        if membrane_incomp
            ts_mul!(ws.tmp3x4, Cm_use, ws.Bi)
            ts_mul_At_add!(ws.K_ab, ws.Bm, ws.tmp3x4, abs_detJ)
            # K_bb += abs_detJ * Bi' * Cm * Bi (reuse tmp3x4 = Cm*Bi)
            ts_mul_At_add!(ws.K_bb, ws.Bi, ws.tmp3x4, abs_detJ)
        end
        # K_ab_bend += abs_detJ * Bb' * Cb * Bi_bend
        ts_mul!(ws.tmp3x4, Cb_use, ws.Bi_bend)
        ts_mul_At_add!(ws.K_ab_bend, ws.Bb, ws.tmp3x4, abs_detJ)
        # K_bb_bend += abs_detJ * Bi_bend' * Cb * Bi_bend (reuse tmp3x4)
        ts_mul_At_add!(ws.K_bb_bend, ws.Bi_bend, ws.tmp3x4, abs_detJ)

        # B coupling cross-terms for incompatible modes (accumulated during main loop)
        if Bmb_use !== nothing
            ts_mul!(ws.tmp3x4, Bmb_use, ws.Bi)
            ts_mul_At_add!(ws.K_ab_cross, ws.Bb, ws.tmp3x4, abs_detJ)
            ts_mul!(ws.tmp3x4, Bmb_use, ws.Bi_bend)
            ts_mul_At_add!(ws.K_ab_bend_cross, ws.Bm, ws.tmp3x4, abs_detJ)
            ts_mul!(ws.tmp3x4, Bmb_use, ws.Bi_bend)
            ts_mul_At_add!(ws.K_mb_incomp, ws.Bi, ws.tmp3x4, abs_detJ)
        end
    end

    # 1-point center shear (shear_center_only=true): locking-free reduced integration.
    # Equivalent to Nastran CQUAD4 selective reduced shear; avoids locking on thin plates.
    # Skipped when skip_all_shear=true (membrane-only element: DOF3 must be exactly zero).
    if shear_center_only && !skip_all_shear
        # Bs_cov at center (r=s=0): bilinear interpolation weights = 0.5 each
        fill!(ws.Bs_cov, 0.0)
        for j in 1:24
            ws.Bs_cov[1,j] = 0.5*ws.Bs_tp[1,j] + 0.5*ws.Bs_tp[2,j]  # avg A,B
            ws.Bs_cov[2,j] = 0.5*ws.Bs_tp[3,j] + 0.5*ws.Bs_tp[4,j]  # avg C,D
        end
        # Cs_cov at center (no phi2 — 1-point integration is already locking-free)
        t11c = center_Cs[1,1]*iJ11c + center_Cs[1,2]*iJ21c; t12c = center_Cs[1,1]*iJ12c + center_Cs[1,2]*iJ22c
        t21c = center_Cs[2,1]*iJ11c + center_Cs[2,2]*iJ21c; t22c = center_Cs[2,1]*iJ12c + center_Cs[2,2]*iJ22c
        ws.tmp2x2[1,1] = iJ11c*t11c + iJ21c*t21c; ws.tmp2x2[1,2] = iJ11c*t12c + iJ21c*t22c
        ws.tmp2x2[2,1] = iJ12c*t11c + iJ22c*t21c; ws.tmp2x2[2,2] = iJ12c*t12c + iJ22c*t22c
        # Ke += 4 * abs_detJc * Bs_cov' * Cs_cov * Bs_cov  (weight=4 for single center point)
        ts_mul!(ws.tmp2x24, ws.tmp2x2, ws.Bs_cov)
        ts_mul_At_add!(ws.Ke, ws.Bs_cov, ws.tmp2x24, 4.0 * abs_detJc)
    elseif selective_shear && !skip_all_shear && !exact_side_shear
        # Directional selective shear integration inspired by DKMQ24_2+:
        # e_xiz term on 1x2, e_etaz term on 2x1, cross-coupling on 2x2.
        # This targets thick flat orthotropic shells without replacing the
        # rest of the MITC4 shell operator.
        function accumulate_shear_terms!(r::Float64, s::Float64, mode::Symbol)
            dNr, dNs = shape_derivs_quad(r, s)
            J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
            J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
            J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
            J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
            detJ = J11*J22 - J12*J21
            abs_detJ = abs(detJ)
            if abs_detJ < 1e-12
                abs_detJ = 1e-12
            end
            inv_det = 1.0 / detJ
            iJ11 = J22*inv_det; iJ12 = -J12*inv_det
            iJ21 = -J21*inv_det; iJ22 = J11*inv_det

            w_eta_p = 0.5*(1.0+s); w_eta_m = 0.5*(1.0-s)
            w_xi_p  = 0.5*(1.0+r); w_xi_m  = 0.5*(1.0-r)
            fill!(ws.Bs_cov, 0.0)
            for j in 1:24
                ws.Bs_cov[1,j] = w_eta_m*ws.Bs_tp[1,j] + w_eta_p*ws.Bs_tp[2,j]
                ws.Bs_cov[2,j] = w_xi_m*ws.Bs_tp[3,j]  + w_xi_p*ws.Bs_tp[4,j]
            end

            t11 = Cs[1,1]*iJ11 + Cs[1,2]*iJ21; t12 = Cs[1,1]*iJ12 + Cs[1,2]*iJ22
            t21 = Cs[2,1]*iJ11 + Cs[2,2]*iJ21; t22 = Cs[2,1]*iJ12 + Cs[2,2]*iJ22
            c11 = phi2_shear*(iJ11*t11 + iJ21*t21)
            c12 = phi2_shear*(iJ11*t12 + iJ21*t22)
            c21 = phi2_shear*(iJ12*t11 + iJ22*t21)
            c22 = phi2_shear*(iJ12*t12 + iJ22*t22)

            if mode === :sx
                @inbounds @fastmath for j in 1:24, i in 1:24
                    ws.Ke[i,j] += abs_detJ * c11 * ws.Bs_cov[1,i] * ws.Bs_cov[1,j]
                end
            elseif mode === :sy
                @inbounds @fastmath for j in 1:24, i in 1:24
                    ws.Ke[i,j] += abs_detJ * c22 * ws.Bs_cov[2,i] * ws.Bs_cov[2,j]
                end
            else
                @inbounds @fastmath for j in 1:24, i in 1:24
                    ws.Ke[i,j] += abs_detJ * (
                        c12 * ws.Bs_cov[1,i] * ws.Bs_cov[2,j] +
                        c21 * ws.Bs_cov[2,i] * ws.Bs_cov[1,j]
                    )
                end
            end
            return nothing
        end

        if selective_shear_mode === :all
            for s in (-pt, pt)
                accumulate_shear_terms!(0.0, s, :sx)
            end
            for r in (-pt, pt)
                accumulate_shear_terms!(r, 0.0, :sy)
            end
            for gp in 1:4
                r, s = gauss_pts[gp][1], gauss_pts[gp][2]
                accumulate_shear_terms!(r, s, :cross)
            end
        elseif selective_shear_mode === :sx_only
            for s in (-pt, pt)
                accumulate_shear_terms!(0.0, s, :sx)
            end
            for gp in 1:4
                r, s = gauss_pts[gp][1], gauss_pts[gp][2]
                accumulate_shear_terms!(r, s, :sy)
                accumulate_shear_terms!(r, s, :cross)
            end
        elseif selective_shear_mode === :sy_only
            for r in (-pt, pt)
                accumulate_shear_terms!(r, 0.0, :sy)
            end
            for gp in 1:4
                r, s = gauss_pts[gp][1], gauss_pts[gp][2]
                accumulate_shear_terms!(r, s, :sx)
                accumulate_shear_terms!(r, s, :cross)
            end
        else
            error("unsupported selective_shear_mode=$(selective_shear_mode)")
        end
    end

    if exact_side_shear && !shear_center_only && !skip_all_shear
        add_quad4_plate_dkmq_exact_shear!(ws.Ke, coords, Cb, Cs, h)
    end

    # MacNeal 1978 RBF shear block — replaces MITC4+phi2 when JFEM_Q4_KERNEL=macneal.
    # Added after the GP loop because the MacNeal formulation integrates the 4
    # shear strain values directly (not per-GP via shape functions).
    if macneal_kernel && !shear_center_only && !skip_all_shear
        add_quad4_macneal_shear_rbf!(
            ws.Ke, coords, Cb, Cs, h;
            epsilon_rbf=macneal_rbf_eps,
            rigid_shear=macneal_rigid_shear,
        )
    end

    # Static condensation (BLAS-free for thread safety)
    if Bmb !== nothing
        if membrane_incomp
            # Combined 8×8 condensation (B coupling creates cross-coupling between membrane/bending incomp modes)
            K_ab_full = hcat(ws.K_ab .+ ws.K_ab_cross, ws.K_ab_bend .+ ws.K_ab_bend_cross)
            K_bb_full = [ws.K_bb ws.K_mb_incomp; ws.K_mb_incomp' ws.K_bb_bend]
            inv_Kbb = Matrix(inv(SMatrix{8,8}(K_bb_full)))
            tmp8x24 = zeros(8, 24)
            ts_mul!(tmp8x24, inv_Kbb, K_ab_full')
            @inbounds @fastmath for j in 1:24, i in 1:24
                s = 0.0
                for l in 1:8; s += K_ab_full[i,l] * tmp8x24[l,j]; end
                ws.Ke[i,j] -= s
            end
        elseif maximum(abs, ws.K_bb_bend) > 0.0
            # Bending-only 4×4 condensation (no membrane incompatible modes)
            # K_ab_bend_cross = Bm' * Bmb * Bi_bend — B-coupling cross term still relevant
            inv_Kbb_b = Matrix(inv(SMatrix{4,4}(ws.K_bb_bend)))
            K_ab_b = ws.K_ab_bend .+ ws.K_ab_bend_cross
            @inbounds @fastmath for j in 1:24, i in 1:24
                sb = 0.0
                for l in 1:4
                    tmp_b = 0.0
                    for q in 1:4; tmp_b += inv_Kbb_b[l,q] * K_ab_b[j,q]; end
                    sb += K_ab_b[i,l] * tmp_b
                end
                ws.Ke[i,j] -= sb
            end
        end
    else
        if bending_incomp && maximum(abs, ws.K_bb_bend) > 0.0
            # Separate 4×4 condensation for bending (always) and membrane (only if membrane_incomp)
            # (skipped when bend_ratio=0, i.e. membrane-only element with zero Cb)
            inv_Kbb_b = Matrix(inv(SMatrix{4,4}(ws.K_bb_bend)))
            inv_Kbb_m = membrane_incomp ? Matrix(inv(SMatrix{4,4}(ws.K_bb))) : nothing
            @inbounds @fastmath for j in 1:24, i in 1:24
                sb = 0.0
                for l in 1:4
                    tmp_b = 0.0
                    for q in 1:4
                        tmp_b += inv_Kbb_b[l,q] * ws.K_ab_bend[j,q]
                    end
                    sb += ws.K_ab_bend[i,l] * tmp_b
                end
                ws.Ke[i,j] -= sb
            end
            if membrane_incomp
                @inbounds @fastmath for j in 1:24, i in 1:24
                    sm = 0.0
                    for l in 1:4
                        tmp_m = 0.0
                        for q in 1:4
                            tmp_m += inv_Kbb_m[l,q] * ws.K_ab[j,q]
                        end
                        sm += ws.K_ab[i,l] * tmp_m
                    end
                    ws.Ke[i,j] -= sm
                end
            end
        else
            # Membrane-only 4×4 condensation (no bending incompatible modes)
            if membrane_incomp
                inv_Kbb_m = Matrix(inv(SMatrix{4,4}(ws.K_bb)))
                @inbounds @fastmath for j in 1:24, i in 1:24
                    sm = 0.0
                    for l in 1:4
                        tmp_m = 0.0
                        for q in 1:4
                            tmp_m += inv_Kbb_m[l,q] * ws.K_ab[j,q]
                        end
                        sm += ws.K_ab[i,l] * tmp_m
                    end
                    ws.Ke[i,j] -= sm
                end
            end
        end
    end

    # MacNeal 1978 warp correction (opt-in, partial). Activated only when:
    #   - JFEM_MACNEAL_WARP_ALPHA env var is set to a non-zero float
    #   - coords_3d is supplied (caller knows the 3D corner positions)
    # Calibrated against a warped-quad sweep: closes
    # ~70% of K[θ_x, T_x] warp gap on the iso warped probe at α=-1/3.
    # Default 0.0 = no correction = original JFEM behavior.
    if coords_3d !== nothing
        warp_alpha_raw = strip(get(ENV, "JFEM_MACNEAL_WARP_ALPHA", ""))
        warp_alpha = isempty(warp_alpha_raw) ? 0.0 :
            (tryparse(Float64, warp_alpha_raw) === nothing ? 0.0 : parse(Float64, warp_alpha_raw))
        if warp_alpha != 0.0
            v1, v2, v3 = quad4_center_frame_from_coords3d(coords_3d)
            cx = (coords_3d[1,1] + coords_3d[2,1] + coords_3d[3,1] + coords_3d[4,1]) * 0.25
            cy = (coords_3d[1,2] + coords_3d[2,2] + coords_3d[3,2] + coords_3d[4,2]) * 0.25
            cz = (coords_3d[1,3] + coords_3d[2,3] + coords_3d[3,3] + coords_3d[4,3]) * 0.25
            apply_macneal_warp_correction!(ws.Ke, coords_3d, (v1, v2, v3),
                                            [cx, cy, cz]; alpha=warp_alpha)
        end
    end

    return ws.Ke
end

@inline function dkq_edge_shape_derivs(xi::Float64, eta::Float64)
    dP_dxi = SVector(
        -xi * (1.0 - eta),
        0.5 * (1.0 - eta^2),
        -xi * (1.0 + eta),
        -0.5 * (1.0 - eta^2),
    )
    dP_deta = SVector(
        -0.5 * (1.0 - xi^2),
        -(1.0 + xi) * eta,
        0.5 * (1.0 - xi^2),
        -(1.0 - xi) * eta,
    )
    return dP_dxi, dP_deta
end

@inline function dkq_edge_shape_second_derivs(xi::Float64, eta::Float64)
    d2P_dxi2 = SVector(
        -(1.0 - eta),
        0.0,
        -(1.0 + eta),
        0.0,
    )
    d2P_deta2 = SVector(
        0.0,
        -(1.0 + xi),
        0.0,
        -(1.0 - xi),
    )
    d2P_dxideta = SVector(
        xi,
        -eta,
        -xi,
        eta,
    )
    return d2P_dxi2, d2P_deta2, d2P_dxideta
end

@inline function dkq_plate_hbar_matrix(Cb::AbstractMatrix)
    Hbar = zeros(2, 6)
    Hbar[1, 1] = Cb[1, 1]
    Hbar[1, 2] = Cb[3, 3]
    Hbar[1, 3] = 2.0 * Cb[1, 3]
    Hbar[1, 4] = Cb[1, 3]
    Hbar[1, 5] = Cb[2, 3]
    Hbar[1, 6] = Cb[1, 2] + Cb[3, 3]
    Hbar[2, 1] = Cb[1, 3]
    Hbar[2, 2] = Cb[2, 3]
    Hbar[2, 3] = Cb[1, 2] + Cb[3, 3]
    Hbar[2, 4] = Cb[3, 3]
    Hbar[2, 5] = Cb[2, 2]
    Hbar[2, 6] = 2.0 * Cb[2, 3]
    return Hbar
end

function dkq_plate_edge_relation(coords, Cb::AbstractMatrix, Cs::AbstractMatrix)
    Hct_inv = inv(Cs)
    Hbar = dkq_plate_hbar_matrix(Cb)

    edge_c = zeros(4)
    edge_s = zeros(4)
    edge_L = zeros(4)
    edge_pairs = ((1, 2), (2, 3), (3, 4), (4, 1))
    @inbounds for e in 1:4
        i, j = edge_pairs[e]
        dx = coords[j, 1] - coords[i, 1]
        dy = coords[j, 2] - coords[i, 2]
        L = sqrt(dx * dx + dy * dy)
        if L <= 1e-12
            continue
        end
        edge_c[e] = dx / L
        edge_s[e] = dy / L
        edge_L[e] = L
    end

    function fill_dkq_point_operators!(Bf_beta, Bf_alpha, Bc_beta, Bc_alpha, xi::Float64, eta::Float64)
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11 * J22 - J12 * J21
        if abs(detJ) < 1e-12
            detJ = detJ < 0.0 ? -1e-12 : 1e-12
        end
        abs_detJ = abs(detJ)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(Bf_beta, 0.0)
        fill!(Bf_alpha, 0.0)
        fill!(Bc_beta, 0.0)
        fill!(Bc_alpha, 0.0)

        d2N_dxideta = (0.25, -0.25, 0.25, -0.25)
        for a in 1:4
            dN_dx = iJ11 * dNr[a] + iJ12 * dNs[a]
            dN_dy = iJ21 * dNr[a] + iJ22 * dNs[a]
            col = (a - 1) * 3
            Bf_beta[1, col + 3] = dN_dx
            Bf_beta[2, col + 2] = -dN_dy
            Bf_beta[3, col + 2] = -dN_dx
            Bf_beta[3, col + 3] = dN_dy

            d2N_xx = 2.0 * iJ11 * iJ12 * d2N_dxideta[a]
            d2N_yy = 2.0 * iJ21 * iJ22 * d2N_dxideta[a]
            d2N_xy = (iJ11 * iJ22 + iJ12 * iJ21) * d2N_dxideta[a]
            Bc_beta[1, col + 3] = Hbar[1, 1] * d2N_xx + Hbar[1, 2] * d2N_yy + Hbar[1, 3] * d2N_xy
            Bc_beta[2, col + 3] = Hbar[2, 1] * d2N_xx + Hbar[2, 2] * d2N_yy + Hbar[2, 3] * d2N_xy
            Bc_beta[1, col + 2] = -(Hbar[1, 4] * d2N_xx + Hbar[1, 5] * d2N_yy + Hbar[1, 6] * d2N_xy)
            Bc_beta[2, col + 2] = -(Hbar[2, 4] * d2N_xx + Hbar[2, 5] * d2N_yy + Hbar[2, 6] * d2N_xy)
        end

        dP_dxi, dP_deta = dkq_edge_shape_derivs(xi, eta)
        d2P_dxi2, d2P_deta2, d2P_dxideta = dkq_edge_shape_second_derivs(xi, eta)
        for e in 1:4
            c = edge_c[e]
            s = edge_s[e]
            dP_dx = iJ11 * dP_dxi[e] + iJ12 * dP_deta[e]
            dP_dy = iJ21 * dP_dxi[e] + iJ22 * dP_deta[e]
            Bf_alpha[1, e] = c * dP_dx
            Bf_alpha[2, e] = s * dP_dy
            Bf_alpha[3, e] = c * dP_dy + s * dP_dx

            d2P_xx = iJ11^2 * d2P_dxi2[e] + iJ12^2 * d2P_deta2[e] + 2.0 * iJ11 * iJ12 * d2P_dxideta[e]
            d2P_yy = iJ21^2 * d2P_dxi2[e] + iJ22^2 * d2P_deta2[e] + 2.0 * iJ21 * iJ22 * d2P_dxideta[e]
            d2P_xy = iJ11 * iJ21 * d2P_dxi2[e] + iJ12 * iJ22 * d2P_deta2[e] +
                     (iJ11 * iJ22 + iJ12 * iJ21) * d2P_dxideta[e]

            Bc_alpha[1, e] =
                Hbar[1, 1] * (c * d2P_xx) +
                Hbar[1, 2] * (c * d2P_yy) +
                Hbar[1, 3] * (c * d2P_xy) +
                Hbar[1, 4] * (s * d2P_xx) +
                Hbar[1, 5] * (s * d2P_yy) +
                Hbar[1, 6] * (s * d2P_xy)
            Bc_alpha[2, e] =
                Hbar[2, 1] * (c * d2P_xx) +
                Hbar[2, 2] * (c * d2P_yy) +
                Hbar[2, 3] * (c * d2P_xy) +
                Hbar[2, 4] * (s * d2P_xx) +
                Hbar[2, 5] * (s * d2P_yy) +
                Hbar[2, 6] * (s * d2P_xy)
        end

        return abs_detJ
    end

    Bf_beta = zeros(3, 12)
    Bf_alpha = zeros(3, 4)
    Bc_beta = zeros(2, 12)
    Bc_alpha = zeros(2, 4)
    tmp3x12 = zeros(3, 12)
    tmp3x4 = zeros(3, 4)
    tmp2x12 = zeros(2, 12)
    tmp2x4 = zeros(2, 4)
    Kf11 = zeros(12, 12)
    Kf12 = zeros(12, 4)
    Kf22 = zeros(4, 4)
    Kbb = zeros(12, 12)
    Kba = zeros(12, 4)
    Kca = zeros(4, 4)

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
    @inbounds for gp in gauss_pts
        abs_detJ = fill_dkq_point_operators!(Bf_beta, Bf_alpha, Bc_beta, Bc_alpha, gp[1], gp[2])

        ts_mul!(tmp3x12, Cb, Bf_beta)
        ts_mul_At_add!(Kf11, Bf_beta, tmp3x12, abs_detJ)

        ts_mul!(tmp3x4, Cb, Bf_alpha)
        ts_mul_At_add!(Kf12, Bf_beta, tmp3x4, abs_detJ)
        ts_mul_At_add!(Kf22, Bf_alpha, tmp3x4, abs_detJ)

        ts_mul!(tmp2x4, Hct_inv, Bc_alpha)
        ts_mul!(tmp2x12, Hct_inv, Bc_beta)
        ts_mul_At_add!(Kbb, Bc_beta, tmp2x12, abs_detJ)
        ts_mul_At_add!(Kba, Bc_beta, tmp2x4, abs_detJ)
        ts_mul_At_add!(Kca, Bc_alpha, tmp2x4, abs_detJ)
    end

    Aw = zeros(4, 12)
    Aalpha = zeros(4, 4)
    edge_gp_coords = (
        (SVector(-pt, -1.0), SVector(pt, -1.0)),
        (SVector(1.0, -pt), SVector(1.0, pt)),
        (SVector(pt, 1.0), SVector(-pt, 1.0)),
        (SVector(-1.0, pt), SVector(-1.0, -pt)),
    )
    Bbar_beta = zeros(2, 12)
    Bbar_alpha = zeros(2, 4)
    for e in 1:4
        fill!(Bbar_beta, 0.0)
        fill!(Bbar_alpha, 0.0)
        for gp in edge_gp_coords[e]
            fill_dkq_point_operators!(Bf_beta, Bf_alpha, Bc_beta, Bc_alpha, gp[1], gp[2])
            @inbounds for jj in 1:12, ii in 1:2
                Bbar_beta[ii, jj] += 0.5 * Bc_beta[ii, jj]
            end
            @inbounds for jj in 1:4, ii in 1:2
                Bbar_alpha[ii, jj] += 0.5 * Bc_alpha[ii, jj]
            end
        end

        i, j = edge_pairs[e]
        L = edge_L[e]
        c = edge_c[e]
        s = edge_s[e]
        q1 = L * c
        q2 = L * s
        Aalpha[e, e] = 2.0 * L / 3.0
        @inbounds for col in 1:4
            Aalpha[e, col] -= q1 * (Hct_inv[1, 1] * Bbar_alpha[1, col] + Hct_inv[1, 2] * Bbar_alpha[2, col]) +
                              q2 * (Hct_inv[2, 1] * Bbar_alpha[1, col] + Hct_inv[2, 2] * Bbar_alpha[2, col])
        end

        wi = (i - 1) * 3 + 1
        rxi = (i - 1) * 3 + 2
        ryi = (i - 1) * 3 + 3
        wj = (j - 1) * 3 + 1
        rxj = (j - 1) * 3 + 2
        ryj = (j - 1) * 3 + 3
        Aw[e, wi] += 1.0
        Aw[e, wj] -= 1.0
        Aw[e, rxi] += 0.5 * L * s
        Aw[e, ryi] -= 0.5 * L * c
        Aw[e, rxj] += 0.5 * L * s
        Aw[e, ryj] -= 0.5 * L * c
        @inbounds for col in 1:12
            Aw[e, col] += q1 * (Hct_inv[1, 1] * Bbar_beta[1, col] + Hct_inv[1, 2] * Bbar_beta[2, col]) +
                          q2 * (Hct_inv[2, 1] * Bbar_beta[1, col] + Hct_inv[2, 2] * Bbar_beta[2, col])
        end
    end

    return Aalpha \ Aw, edge_c, edge_s, edge_L
end

@inline function dkmq_side_local_constitutive(c::Float64, s::Float64, Cb::AbstractMatrix, Cs::AbstractMatrix)
    # Exact side-local constitutive transformation from the composite DKMQ24 paper.
    Rb = @SMatrix [
        c^2       s^2        2.0*c*s;
        s^2       c^2       -2.0*c*s;
       -c*s       c*s        c^2 - s^2
    ]
    Rs = @SMatrix [
        c   s;
       -s   c
    ]
    Cb_loc = Matrix(Rb * SMatrix{3,3}(Cb) * transpose(Rb))
    Cs_loc = Matrix(Rs * SMatrix{2,2}(Cs) * transpose(Rs))
    return Cb_loc, Cs_loc
end

function dkmq_flat_plate_edge_relation(coords::AbstractMatrix, Cb::AbstractMatrix, Cs::AbstractMatrix)
    A_beta = zeros(4, 12)
    edge_c = zeros(4)
    edge_s = zeros(4)
    edge_L = zeros(4)
    edge_pairs = ((1, 2), (2, 3), (3, 4), (4, 1))

    @inbounds for e in 1:4
        i, j = edge_pairs[e]
        dx = coords[j, 1] - coords[i, 1]
        dy = coords[j, 2] - coords[i, 2]
        L = sqrt(dx * dx + dy * dy)
        L <= 1e-12 && continue

        c = dx / L
        s = dy / L
        edge_c[e] = c
        edge_s[e] = s
        edge_L[e] = L

        Cb_loc, Cs_loc = dkmq_side_local_constitutive(c, s, Cb, Cs)
        # DKMQ24 (Katili-Maknun-Batoz-Ibrahimbegović 2018, Comp. Struct. 202)
        # eq (69) — per-side hierarchical-rotation locking coefficient:
        #
        #   κ_k = ( Hs_inv_k[2,1] · Hb_k[3,2] + Hs_inv_k[2,2] · Hb_k[2,2] ) · 12 / L_k²
        #
        # The paper's index convention follows its [Rk1] rotation matrix
        # (which has S²_k in the [1,1] position). JFEM's [Rb] has C²_k in
        # the [1,1] position — i.e. transposed. The two conventions assign
        # the "along-side" axis to opposite indices. Mapping: paper's
        # Hb_k[2,2] corresponds to JFEM's Cb_loc[1,1] (along-side D_ss).
        # Likewise paper's Hs_inv_k[2,2] is JFEM's Hs_loc_inv[1,1] and
        # paper's Hs_inv_k[2,1] · Hb_k[3,2] is JFEM's Hs_loc_inv[1,2] ·
        # Cb_loc[3,1]. The off-diagonal coupling term captures bending-twist
        # interaction in unbalanced composite layups; it vanishes for any
        # orthotropic material aligned with the side (and for iso).
        #
        # Iso reduction (eq 72) is recovered when Cs_loc is diagonal and
        # Cb_loc has zero twist-bending coupling: κ = Db/Ds · 12/L².
        det_Cs_loc = Cs_loc[1,1]*Cs_loc[2,2] - Cs_loc[1,2]*Cs_loc[2,1]
        Hs_inv_11 = abs(det_Cs_loc) > 1e-30 ? Cs_loc[2,2] / det_Cs_loc : 1.0 / max(abs(Cs_loc[1,1]), 1e-30)
        Hs_inv_12 = abs(det_Cs_loc) > 1e-30 ? -Cs_loc[1,2] / det_Cs_loc : 0.0
        Hb_along  = Cb_loc[1, 1]   # paper's Hb_k[2,2] = D_ss
        Hb_couple = Cb_loc[3, 1]   # paper's Hb_k[3,2] = bending-twist coupling
        phi = (Hs_inv_12 * Hb_couple + Hs_inv_11 * Hb_along) * 12.0 / (L * L)
        if !isfinite(phi) || phi < 0.0
            # Fallback to iso form if the layup-coupling term flips sign in a
            # non-physical way (e.g. anti-symmetric layup with negative D_ss)
            Db = max(abs(Cb_loc[1, 1]), 1e-30)
            Ds = max(abs(Cs_loc[1, 1]), 1e-30)
            phi = 12.0 * Db / (L * L * Ds)
        end
        scale = 1.5 / (L * (1.0 + phi))
        rot = 0.5 * L * scale

        coli = (i - 1) * 3
        colj = (j - 1) * 3

        A_beta[e, coli + 1] += scale
        A_beta[e, colj + 1] -= scale

        # beta_t = c*theta_y - s*theta_x in the present plate kinematics.
        A_beta[e, coli + 2] +=  rot * s
        A_beta[e, coli + 3] += -rot * c
        A_beta[e, colj + 2] +=  rot * s
        A_beta[e, colj + 3] += -rot * c
    end

    return A_beta, edge_c, edge_s, edge_L
end

@inline function dkmq_plate_side_shear_operator!(
    Bs::AbstractMatrix,
    coords::AbstractMatrix,
    A_beta::AbstractMatrix,
    edge_L::AbstractVector,
    xi::Float64,
    eta::Float64,
)
    dNr, dNs = shape_derivs_quad(xi, eta)
    J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
    J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
    J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
    J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
    detJ = J11 * J22 - J12 * J21
    if abs(detJ) < 1e-12
        detJ = detJ < 0.0 ? -1e-12 : 1e-12
    end
    abs_detJ = abs(detJ)
    inv_det = 1.0 / detJ
    iJ11 = J22 * inv_det
    iJ12 = -J12 * inv_det
    iJ21 = -J21 * inv_det
    iJ22 = J11 * inv_det

    # Exact DKMQ side shear interpolation:
    # Bs = Bs4Δ * A_n, with side signs from the plate/shell derivation.
    c5 = 0.25 * (1.0 - eta) * edge_L[1]
    c6 = 0.25 * (1.0 + xi) * edge_L[2]
    c7 = -0.25 * (1.0 + eta) * edge_L[3]
    c8 = -0.25 * (1.0 - xi) * edge_L[4]

    @inbounds for j in 1:12
        Bs[1, j] =
            iJ11 * (c5 * A_beta[1, j] + c7 * A_beta[3, j]) +
            iJ12 * (c6 * A_beta[2, j] + c8 * A_beta[4, j])
        Bs[2, j] =
            iJ21 * (c5 * A_beta[1, j] + c7 * A_beta[3, j]) +
            iJ22 * (c6 * A_beta[2, j] + c8 * A_beta[4, j])
    end
    return abs_detJ
end

function add_quad4_plate_dkmq_exact_shear!(
    Ke::AbstractMatrix,
    coords::AbstractMatrix,
    Cb::AbstractMatrix,
    Cs::AbstractMatrix,
    h::Float64,
)
    if h < 1e-30 || maximum(abs, Cs) < 1e-30
        return Ke
    end

    A_beta, _, _, edge_L = dkmq_flat_plate_edge_relation(coords, Cb, Cs)
    Bs = zeros(2, 12)
    K_plate = zeros(12, 12)
    pt = 1.0 / sqrt(3.0)

    # DKMQ24_2+ selective integration of the shear rows:
    # b_sx with 1x2, b_sy with 2x1, and any shear-coupling cross-term with 2x2.
    row1_pts = ((0.0, -pt, 2.0), (0.0, pt, 2.0))
    row2_pts = ((-pt, 0.0, 2.0), (pt, 0.0, 2.0))
    full_pts = (
        (-pt, -pt, 1.0),
        (pt, -pt, 1.0),
        (pt, pt, 1.0),
        (-pt, pt, 1.0),
    )

    c11 = Cs[1, 1]
    c22 = Cs[2, 2]
    c12 = Cs[1, 2]
    c21 = Cs[2, 1]

    @inbounds for (xi, eta, weight) in row1_pts
        abs_detJ = dkmq_plate_side_shear_operator!(Bs, coords, A_beta, edge_L, xi, eta)
        for j in 1:12, i in 1:12
            K_plate[i, j] += h * weight * abs_detJ * c11 * Bs[1, i] * Bs[1, j]
        end
    end

    @inbounds for (xi, eta, weight) in row2_pts
        abs_detJ = dkmq_plate_side_shear_operator!(Bs, coords, A_beta, edge_L, xi, eta)
        for j in 1:12, i in 1:12
            K_plate[i, j] += h * weight * abs_detJ * c22 * Bs[2, i] * Bs[2, j]
        end
    end

    if abs(c12) > 1e-14 || abs(c21) > 1e-14
        @inbounds for (xi, eta, weight) in full_pts
            abs_detJ = dkmq_plate_side_shear_operator!(Bs, coords, A_beta, edge_L, xi, eta)
            for j in 1:12, i in 1:12
                K_plate[i, j] += h * weight * abs_detJ * (
                    c12 * Bs[1, i] * Bs[2, j] +
                    c21 * Bs[2, i] * Bs[1, j]
                )
            end
        end
    end

    plate_dofs = (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)
    @inbounds for j in 1:12, i in 1:12
        Ke[plate_dofs[i], plate_dofs[j]] += K_plate[i, j]
    end
    return Ke
end

# ---------------------------------------------------------------------------
# Tessler-Hughes 1983 MIN4 — anisoparametric Q4 plate bending + transverse shear.
#
# Reference: A. Tessler & T.J.R. Hughes, "An Improved Treatment Of Transverse
# Shear In The Mindlin-Type Four-Node Quadrilateral Element", CMAME 39 (1983)
# pp 311-335. Also see MYSTRAN's QPLT2/MIN4SH/BBMIN4/BSMIN4/CALC_PHI_SQ
# (Bill Case, MIT-licensed open-source NASTRAN clone).
#
# Algorithm:
#   * w (out-of-plane) interpolated biquadratic-serendipity; mid-side w_5..w_8
#     are eliminated via "continuous shear edge constraint" (γ_sz),_s = 0.
#     Result: bilinear PSH for w plus enrichments NXSH, NYSH multiplying
#     the corner rotations.
#   * θx, θy interpolated bilinearly.
#   * Bending B-matrix uses ONLY [P],_x, [P],_y (no enrichment, eq 6.2).
#   * Shear B-matrix uses [P],_x/y on w and [Nx]/[Ny] enrichments + PSH on θ
#     (eqs 5.6, 6.2).
#   * Shear correction factor φ² = C_b·ψ̂/(1+C_b·ψ̂)  (eq 4.21)
#     ψ̂ = BENSUM/SHRSUM (ratio of bending to shear stiffness diagonals).
#     C_b is an element constant ~O(1); MYSTRAN uses 3.6 (env-overridable).
#   * φ² scales ONLY the transverse-shear block, leaving Kb untouched.
#
# This function returns ONLY the bending + (φ²·shear) contributions to the
# 24×24 element matrix, embedded at the (uz, rx, ry) DOFs of each node. The
# caller is responsible for combining with a membrane+drilling kernel.
# Membrane/drilling can be obtained by calling stiffness_quad4_matrices with
# Cb=0, Cs=0.
# ---------------------------------------------------------------------------
function stiffness_quad4_min4_bending_shear(
    coords::AbstractMatrix{Float64},
    Cb::AbstractMatrix{Float64},
    Cs::AbstractMatrix{Float64};
    cbmin4::Float64 = 3.6,
    iord_bending::Int = 2,
    iord_shear::Int = 3,
)
    # Side differences: XSD[i] = X[i] - X[i+1] (i=1..4, wrap at 4→1).
    # Matches MYSTRAN convention (BD_CQUAD computes XSD/YSD that way; see JAC2D).
    X1, X2, X3, X4 = coords[1,1], coords[2,1], coords[3,1], coords[4,1]
    Y1, Y2, Y3, Y4 = coords[1,2], coords[2,2], coords[3,2], coords[4,2]
    XSD = SVector(X1 - X2, X2 - X3, X3 - X4, X4 - X1)
    YSD = SVector(Y1 - Y2, Y2 - Y3, Y3 - Y4, Y4 - Y1)

    # Gauss quadrature points and weights
    function gauss_1d(n::Int)
        if n == 2
            g = 1.0 / sqrt(3.0)
            return SVector(-g, g), SVector(1.0, 1.0)
        elseif n == 3
            g = sqrt(3.0/5.0)
            return SVector(-g, 0.0, g), SVector(5.0/9.0, 8.0/9.0, 5.0/9.0)
        else
            error("Unsupported Gauss order: $n (expected 2 or 3)")
        end
    end
    gp_b, gw_b = gauss_1d(iord_bending)
    gp_s, gw_s = gauss_1d(iord_shear)

    # Helpers: at given (ssi, ssj) return PSH, DPSHX, DNXSHX, DNYSHX, DETJ
    function shape_data(ssi::Float64, ssj::Float64)
        # Bilinear shapes and ξ,η-derivatives
        PSH = SVector(
            0.25*(1.0 - ssi)*(1.0 - ssj),
            0.25*(1.0 + ssi)*(1.0 - ssj),
            0.25*(1.0 + ssi)*(1.0 + ssj),
            0.25*(1.0 - ssi)*(1.0 + ssj),
        )
        DPSHG = @SMatrix [
            -0.25*(1.0 - ssj)   0.25*(1.0 - ssj)   0.25*(1.0 + ssj)  -0.25*(1.0 + ssj);
            -0.25*(1.0 - ssi)  -0.25*(1.0 + ssi)   0.25*(1.0 + ssi)   0.25*(1.0 - ssi)
        ]
        # Jacobian (MYSTRAN JAC2D convention)
        J11 = (-(1.0 - ssj)*XSD[1] + (1.0 + ssj)*XSD[3]) / 4.0
        J12 = (-(1.0 - ssj)*YSD[1] + (1.0 + ssj)*YSD[3]) / 4.0
        J21 = ( (1.0 - ssi)*XSD[4] - (1.0 + ssi)*XSD[2]) / 4.0
        J22 = ( (1.0 - ssi)*YSD[4] - (1.0 + ssi)*YSD[2]) / 4.0
        DETJ = J11*J22 - J12*J21
        inv_detj = 1.0 / DETJ
        # Inverse Jacobian
        JI11 =  J22 * inv_detj
        JI12 = -J12 * inv_detj
        JI21 = -J21 * inv_detj
        JI22 =  J11 * inv_detj
        JI = @SMatrix [JI11 JI12; JI21 JI22]
        DPSHX = JI * DPSHG
        # Tessler-Hughes constrained shapes NXSH, NYSH from MIN4SH:
        # virgin midside biquadratic shapes N5..N8
        XM  = 1.0 - ssi
        XP  = 1.0 + ssi
        YM  = 1.0 - ssj
        YP  = 1.0 + ssj
        X2M = 1.0 - ssi*ssi
        Y2M = 1.0 - ssj*ssj
        N5  = X2M*YM/2.0
        N6  = Y2M*XP/2.0
        N7  = X2M*YP/2.0
        N8  = Y2M*XM/2.0
        N5X = -ssi*YM;   N6X = Y2M/2.0;   N7X = -ssi*YP;  N8X = -Y2M/2.0
        N5Y = -X2M/2.0;  N6Y = -ssj*XP;   N7Y = X2M/2.0;  N8Y = -ssj*XM
        NXSH = SVector(
            (-YSD[4]*N8 + YSD[1]*N5)/8.0,
            (-YSD[1]*N5 + YSD[2]*N6)/8.0,
            (-YSD[2]*N6 + YSD[3]*N7)/8.0,
            (-YSD[3]*N7 + YSD[4]*N8)/8.0,
        )
        NYSH = SVector(
            (-XSD[4]*N8 + XSD[1]*N5)/8.0,
            (-XSD[1]*N5 + XSD[2]*N6)/8.0,
            (-XSD[2]*N6 + XSD[3]*N7)/8.0,
            (-XSD[3]*N7 + XSD[4]*N8)/8.0,
        )
        DNXSHG = @SMatrix [
            (-YSD[4]*N8X + YSD[1]*N5X)/8.0  (-YSD[1]*N5X + YSD[2]*N6X)/8.0  (-YSD[2]*N6X + YSD[3]*N7X)/8.0  (-YSD[3]*N7X + YSD[4]*N8X)/8.0;
            (-YSD[4]*N8Y + YSD[1]*N5Y)/8.0  (-YSD[1]*N5Y + YSD[2]*N6Y)/8.0  (-YSD[2]*N6Y + YSD[3]*N7Y)/8.0  (-YSD[3]*N7Y + YSD[4]*N8Y)/8.0
        ]
        DNYSHG = @SMatrix [
            (-XSD[4]*N8X + XSD[1]*N5X)/8.0  (-XSD[1]*N5X + XSD[2]*N6X)/8.0  (-XSD[2]*N6X + XSD[3]*N7X)/8.0  (-XSD[3]*N7X + XSD[4]*N8X)/8.0;
            (-XSD[4]*N8Y + XSD[1]*N5Y)/8.0  (-XSD[1]*N5Y + XSD[2]*N6Y)/8.0  (-XSD[2]*N6Y + XSD[3]*N7Y)/8.0  (-XSD[3]*N7Y + XSD[4]*N8Y)/8.0
        ]
        DNXSHX = JI * DNXSHG
        DNYSHX = JI * DNYSHG
        return PSH, DPSHX, DNXSHX, DNYSHX, DETJ
    end

    # --- Bending stiffness Kb (8×8) ---
    Kb = MMatrix{8, 8, Float64}(zeros(8, 8))
    Cb_static = SMatrix{3, 3, Float64}(Cb)
    for i in eachindex(gp_b)
        for j in eachindex(gp_b)
            PSH, DPSHX, _, _, DETJ = shape_data(gp_b[i], gp_b[j])
            # BBMIN4: 3×8 bending B (DOFs per node: θx, θy)
            BB = MMatrix{3, 8, Float64}(zeros(3, 8))
            @inbounds for jj in 1:4
                col_tx = 2*jj - 1
                col_ty = 2*jj
                BB[1, col_tx] = 0.0
                BB[2, col_tx] = -DPSHX[2, jj]
                BB[3, col_tx] = -DPSHX[1, jj]
                BB[1, col_ty] =  DPSHX[1, jj]
                BB[2, col_ty] =  0.0
                BB[3, col_ty] =  DPSHX[2, jj]
            end
            intfac = DETJ * gw_b[i] * gw_b[j]
            BB_static = SMatrix{3, 8, Float64}(BB)
            Kb .+= intfac .* (transpose(BB_static) * Cb_static * BB_static)
        end
    end

    # --- Shear stiffness Ks (12×12) ---
    Ks = MMatrix{12, 12, Float64}(zeros(12, 12))
    Cs_static = SMatrix{2, 2, Float64}(Cs)
    for i in eachindex(gp_s)
        for j in eachindex(gp_s)
            PSH, DPSHX, DNXSHX, DNYSHX, DETJ = shape_data(gp_s[i], gp_s[j])
            # BSMIN4: 2×12 shear B (DOFs per node: uz, θx, θy)
            BS = MMatrix{2, 12, Float64}(zeros(2, 12))
            @inbounds for jj in 1:4
                col_uz = 3*jj - 2
                col_tx = 3*jj - 1
                col_ty = 3*jj
                BS[1, col_uz] =  DPSHX[1, jj]
                BS[2, col_uz] =  DPSHX[2, jj]
                BS[1, col_tx] = -DNXSHX[1, jj]
                BS[2, col_tx] = -DNXSHX[2, jj] - PSH[jj]
                BS[1, col_ty] =  DNYSHX[1, jj] + PSH[jj]
                BS[2, col_ty] =  DNYSHX[2, jj]
            end
            intfac = DETJ * gw_s[i] * gw_s[j]
            BS_static = SMatrix{2, 12, Float64}(BS)
            Ks .+= intfac .* (transpose(BS_static) * Cs_static * BS_static)
        end
    end

    # φ² shear correction
    bensum = 0.0
    @inbounds for k in 1:8
        bensum += Kb[k, k]
    end
    # Shear diagonal rotation DOFs only: positions 2,3,5,6,8,9,11,12
    shrsum = Ks[2,2] + Ks[3,3] + Ks[5,5] + Ks[6,6] + Ks[8,8] + Ks[9,9] + Ks[11,11] + Ks[12,12]
    if abs(shrsum) < 1e-30
        phi_sq = 1.0
    else
        psi_hat = bensum / shrsum
        phi_sq  = cbmin4 * psi_hat / (1.0 + cbmin4 * psi_hat)
    end

    # Embed into 24×24 (JFEM DOF order [ux,uy,uz,rx,ry,rz] per node).
    # Bending DOFs: rx, ry of each node → 4,5,10,11,16,17,22,23
    # Shear DOFs: uz, rx, ry of each node → 3,4,5,9,10,11,15,16,17,21,22,23
    IDB = SVector{8,Int}(4, 5, 10, 11, 16, 17, 22, 23)
    IDS = SVector{12,Int}(3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)
    Ke = zeros(24, 24)
    @inbounds for j in 1:8, i in 1:8
        Ke[IDB[i], IDB[j]] += Kb[i, j]
    end
    @inbounds for j in 1:12, i in 1:12
        Ke[IDS[i], IDS[j]] += phi_sq * Ks[i, j]
    end

    return Ke, bensum, shrsum, phi_sq
end

# ---------------------------------------------------------------------------
# MacNeal 1978 CQUAD4 transverse shear with residual bending flexibility
# (Comp. Struct. 8:175-183). Reference formulation used by Nastran CQUAD4.
#
#   Strain values γx_a, γx_b at (ξ, η) = (0, ±1/√3)
#                 γy_c, γy_d at (ξ, η) = (±1/√3, 0)
#   Element shear stiffness: K_s = [D]ᵀ · ([Z_s] + [Z_b])⁻¹ · [D]
#     [D]     4×12 — strain values in terms of plate DOFs (w, θx, θy)
#     [Z_s]   physical shear compliance matrix (eq 23-25)
#     [Z_b]   residual bending flexibility matrix (eq 26-27)
#
# This REPLACES the MITC4+phi2 shear block when JFEM_Q4_KERNEL=macneal.
# ---------------------------------------------------------------------------
function add_quad4_macneal_shear_rbf!(
    Ke::AbstractMatrix{Float64},
    coords::AbstractMatrix{Float64},
    Cb::AbstractMatrix{Float64},
    Cs::AbstractMatrix{Float64},
    h::Float64;
    epsilon_rbf::Float64 = 0.04,
    rigid_shear::Bool = false,
)
    # Shortcut: skip if thickness or shear modulus is effectively zero
    if h < 1e-30 || (!rigid_shear && maximum(abs, Cs) < 1e-30)
        return Ke
    end

    # Sampling points: (ξ, η, component) where component=1 is γx, =2 is γy
    pt = 1.0 / sqrt(3.0)
    shear_pts = ((0.0, -pt, 1), (0.0,  pt, 1), (-pt, 0.0, 2), ( pt, 0.0, 2))

    D_mat = zeros(4, 12)
    J_pts = zeros(4)
    # Per-shear-sample-point physical extents for the residual-bending-flexibility
    # block (MacNeal eq 26, generalized to non-rectangular quads).
    #   pt_delta[1] = 2·J11 at (ξ=0, η=-1/√3)  → physical x-extent at γ_x sample a
    #   pt_delta[2] = 2·J11 at (ξ=0, η=+1/√3)  → physical x-extent at γ_x sample b
    #   pt_delta[3] = 2·J22 at (ξ=-1/√3, η=0)  → physical y-extent at γ_y sample c
    #   pt_delta[4] = 2·J22 at (ξ=+1/√3, η=0)  → physical y-extent at γ_y sample d
    # On a rectangle (and hence on any uniform-Jacobian quad) all γ_x extents collapse
    # to MacNeal's Δx and all γ_y extents to Δy, recovering the original eq (26).
    pt_delta = zeros(4)
    @inbounds for sp_idx in 1:4
        xi, eta, comp = shear_pts[sp_idx]
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1]+dNr[2]*coords[2,1]+dNr[3]*coords[3,1]+dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2]+dNr[2]*coords[2,2]+dNr[3]*coords[3,2]+dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1]+dNs[2]*coords[2,1]+dNs[3]*coords[3,1]+dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2]+dNs[2]*coords[2,2]+dNs[3]*coords[3,2]+dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        if abs(detJ) < 1e-14
            detJ = detJ < 0.0 ? -1e-14 : 1e-14
        end
        J_pts[sp_idx] = abs(detJ)
        # Save the diagonal Jacobian component aligned with the strain axis at this point.
        # γ_x samples at (ξ=0, η=±1/√3): the "beam strip" runs in ξ → x-extent ≈ 2·|J11|
        # γ_y samples at (ξ=±1/√3, η=0): the "beam strip" runs in η → y-extent ≈ 2·|J22|
        pt_delta[sp_idx] = comp == 1 ? 2.0 * abs(J11) : 2.0 * abs(J22)
        inv_det = 1.0/detJ
        iJ11 = J22*inv_det;  iJ12 = -J12*inv_det
        iJ21 = -J21*inv_det; iJ22 = J11*inv_det

        N1 = 0.25*(1-xi)*(1-eta); N2 = 0.25*(1+xi)*(1-eta)
        N3 = 0.25*(1+xi)*(1+eta); N4 = 0.25*(1-xi)*(1+eta)
        N_vals = (N1, N2, N3, N4)

        for k in 1:4
            dN_dx = iJ11*dNr[k] + iJ12*dNs[k]
            dN_dy = iJ21*dNr[k] + iJ22*dNs[k]
            Nk    = N_vals[k]
            col   = (k-1)*3  # plate DOFs: 1=w, 2=θx, 3=θy
            if comp == 1  # γ_xz = ∂w/∂x + θy
                D_mat[sp_idx, col+1] = dN_dx
                D_mat[sp_idx, col+3] = Nk
            else          # γ_yz = ∂w/∂y − θx
                D_mat[sp_idx, col+1] = dN_dy
                D_mat[sp_idx, col+2] = -Nk
            end
        end
    end

    # MacNeal projected side lengths Δx, Δy (eq after 26)
    Dx = 0.5 * (coords[2,1]+coords[3,1]-coords[1,1]-coords[4,1])
    Dy = 0.5 * (coords[3,2]+coords[4,2]-coords[1,2]-coords[2,2])
    Dx2 = Dx*Dx; Dy2 = Dy*Dy

    # Element area from center-Jacobian
    dNr_c = (-0.25, 0.25, 0.25, -0.25)
    dNs_c = (-0.25, -0.25, 0.25, 0.25)
    J11c = dNr_c[1]*coords[1,1]+dNr_c[2]*coords[2,1]+dNr_c[3]*coords[3,1]+dNr_c[4]*coords[4,1]
    J12c = dNr_c[1]*coords[1,2]+dNr_c[2]*coords[2,2]+dNr_c[3]*coords[3,2]+dNr_c[4]*coords[4,2]
    J21c = dNs_c[1]*coords[1,1]+dNs_c[2]*coords[2,1]+dNs_c[3]*coords[3,1]+dNs_c[4]*coords[4,1]
    J22c = dNs_c[1]*coords[1,2]+dNs_c[2]*coords[2,2]+dNs_c[3]*coords[3,2]+dNs_c[4]*coords[4,2]
    detJc = J11c*J22c - J12c*J21c
    A_elem = 4.0 * abs(detJc)

    # Aspect-ratio-adjusted coefficients (MacNeal eq 27):
    #   a = eps / (eps + (1 - eps) * (Dx/Dy)^2)
    #   b = eps / (eps + (1 - eps) * (Dy/Dx)^2)
    # The earlier bounded interpolation was numerically safe, but it was not
    # MacNeal's formula and made a=b=0.52 for square elements when eps=0.04.
    ε = clamp(epsilon_rbf, 1e-12, 1.0)
    a_param = ε / (ε + (1.0 - ε) * Dx2 / max(Dy2, 1e-30))
    b_param = ε / (ε + (1.0 - ε) * Dy2 / max(Dx2, 1e-30))

    flex_mode = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_RBF_BENDING_FLEX_MODE", "diag")))
    flex_x = 1.0 / max(abs(Cb[1,1]), 1e-30)
    flex_y = 1.0 / max(abs(Cb[2,2]), 1e-30)
    if flex_mode in ("full", "compliance", "matrix")
        Cb_sym = 0.5 .* (Cb .+ Cb')
        reg = 1e-12 * max(maximum(abs, Cb_sym), 1e-30)
        Cb_reg = Cb_sym + reg .* Matrix{Float64}(I, 3, 3)
        Sb = inv(Cb_reg)
        flex_x = max(abs(Sb[1,1]), 1e-30)
        flex_y = max(abs(Sb[2,2]), 1e-30)
    end
    inv_12A = 1.0 / (12.0 * A_elem)

    # Residual bending flexibility (MacNeal eq 26): two coupled 2x2 blocks
    # for the gamma_x(a,b) and gamma_y(c,d) shear samples.
    #
    # Tapered/non-rectangular quads: replace MacNeal's single Δx² with the
    # geometric outer product Δx_a·Δx_b at each (i,j) entry of the γ_x block,
    # using the per-shear-point physical extents `pt_delta[1..4]` recorded
    # above. Reduces exactly to the rectangular form when Δx_a=Δx_b=Δx and
    # Δy_c=Δy_d=Δy (so flat AR-aligned panels are unaffected).
    #
    # Eigenstructure check: when α=0 (uniform γ — e.g. torsion), the (a,b)
    # block becomes the rank-1 outer product [Δx_a; Δx_b][Δx_a; Δx_b]ᵀ — only
    # the physical-length-weighted SUM of γ_x is penalized; the orthogonal
    # differential mode is free. When α=1 (full bending), the block is
    # diagonal with entries Δx_a², Δx_b² — each sample is independently
    # penalized by its local span. Both limits are physically consistent
    # generalizations of MacNeal's original rectangular formulation.
    # Research switch: per-shear-sample-point Δ in MacNeal's eq (26).
    # Currently off by default — first ablation on the worst VTP taper showed
    # the per-GP form (with bilinear Δ ≈ 2·J_diag at sample) shifts K closer
    # to Nastran's overall norm but slightly worsens the directional error,
    # i.e. it's not a clean win. Leaving the implementation in place behind
    # an env switch so further exploration can compare against the legacy form.
    per_gp_delta = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_RBF_PER_GP_DELTA", "false"))) in ("1","true","yes","on")
    Zb = zeros(4, 4)
    length_mode = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_RBF_LENGTH_MODE", "paper")))
    swap_xy = length_mode in ("swap", "swapped", "cross")
    Lx2_rbf = swap_xy ? Dy2 : Dx2
    Ly2_rbf = swap_xy ? Dx2 : Dy2
    # MacNeal RBF magnitude (zb_scale) default — empirical Nastran calibration.
    # Probe-library result (2026-05-14): zb_scale = 0.65 reproduces Nastran's
    # CQUAD4 per-element K to within 3-5% across all flat-shell MAT1/MAT8/PCOMP
    # probes (aspect 1-20, thickness h/L 0.001-0.033, orthotropy E1/E2 1-20,
    # laminate 3-17 plies). JFEM's prior default of 1.0 over-applied the
    # MacNeal RBF by ~50%. Promoted to default 2026-05-14 after validation
    # showed zero parity change at this value (mean 6.40%, max 17.81%
    # unchanged) — the K/Kg cascade absorbs the per-element K shift for
    # SOL105 eigenvalues, but per-element K parity improves substantially.
    #
    # JFEM_Q4_MACNEAL_RBF_ZB_SCALE_LEGACY: set to "true" to restore the old
    # 1.0 default if a downstream pipeline depends on the previous K magnitude.
    legacy_zb = lowercase(strip(get(ENV, "JFEM_Q4_MACNEAL_RBF_ZB_SCALE_LEGACY", ""))) in ("1","true","yes","on")
    default_zb_scale = if legacy_zb
        rigid_shear ? (2.0 / 3.0) : 1.0
    elseif rigid_shear
        2.0 / 3.0
    else
        0.65
    end
    zb_scale_raw = tryparse(Float64, strip(get(ENV, "JFEM_Q4_MACNEAL_RBF_ZB_SCALE", string(default_zb_scale))))
    zb_scale = zb_scale_raw === nothing ? default_zb_scale : max(zb_scale_raw, 1e-12)
    if per_gp_delta && !swap_xy
        # Per-GP Δ at each shear sampling point. pt_delta[1,2] are γ_x x-extents;
        # pt_delta[3,4] are γ_y y-extents.
        Δa = pt_delta[1]; Δb = pt_delta[2]
        Δc = pt_delta[3]; Δd = pt_delta[4]
        scl_x = zb_scale * inv_12A * flex_x
        scl_y = zb_scale * inv_12A * flex_y
        Zb[1,1] = scl_x * (1.0 + a_param) * Δa * Δa
        Zb[2,2] = scl_x * (1.0 + a_param) * Δb * Δb
        Zb[1,2] = scl_x * (1.0 - a_param) * Δa * Δb
        Zb[2,1] = Zb[1,2]
        Zb[3,3] = scl_y * (1.0 + b_param) * Δc * Δc
        Zb[4,4] = scl_y * (1.0 + b_param) * Δd * Δd
        Zb[3,4] = scl_y * (1.0 - b_param) * Δc * Δd
        Zb[4,3] = Zb[3,4]
    else
        # Legacy MacNeal eq (26) with averaged Δx, Δy. Kept under env override
        # for ablation; also used when the diagnostic Δx/Δy swap is requested.
        zbx_diag = zb_scale * inv_12A * (1.0 + a_param) * Lx2_rbf * flex_x
        zbx_off  = zb_scale * inv_12A * (1.0 - a_param) * Lx2_rbf * flex_x
        zby_diag = zb_scale * inv_12A * (1.0 + b_param) * Ly2_rbf * flex_y
        zby_off  = zb_scale * inv_12A * (1.0 - b_param) * Ly2_rbf * flex_y
        Zb[1,1] = zbx_diag
        Zb[1,2] = zbx_off
        Zb[2,1] = zbx_off
        Zb[2,2] = zbx_diag
        Zb[3,3] = zby_diag
        Zb[3,4] = zby_off
        Zb[4,3] = zby_off
        Zb[4,4] = zby_diag
    end

    # Physical shear compliance (eq 23-25)
    # [V^s] = diag(√(2 J_p)); [V^s G^s V^s] has G_s = Cs for same-component pairs,
    # G_xy for cross-pairs (symmetric per eq 25)
    Zs = zeros(4, 4)
    if !rigid_shear
    G_xx = Cs[1,1]; G_yy = Cs[2,2]; G_xy = Cs[1,2]
    comps = (1, 1, 2, 2)
    VGV = zeros(4, 4)
    @inbounds for i in 1:4, j in 1:4
        ci = comps[i]; cj = comps[j]
        Jfac = sqrt(2.0*J_pts[i]) * sqrt(2.0*J_pts[j])
        if ci == cj
            if i == j
                VGV[i,j] = Jfac * (ci == 1 ? G_xx : G_yy)
            else
                # Different points, same component — no direct coupling
                # (MacNeal's integration is independent per point)
                VGV[i,j] = 0.0
            end
        else
            # Symmetric x-y coupling through G_xy (eq 25)
            VGV[i,j] = 0.5 * Jfac * G_xy
        end
    end

    # Add a tiny diagonal regularization to avoid singular VGV for near-zero G_xy cases
    for i in 1:4
        if VGV[i,i] < 1e-30
            VGV[i,i] = 1e-30
        end
    end

    # Enforce symmetry of VGV before inversion (protects against asymmetry
    # from accumulated floating-point differences in cross-coupling terms)
    VGV_sym = 0.5 * (VGV + VGV')
    Zs .= inv(VGV_sym)
    Zs .= 0.5 .* (Zs .+ Zs')
    end
    Z_total = Zs + Zb
    Z_total = 0.5 * (Z_total + Z_total')
    # K_plate = Dᵀ · inv(Z_total) · D
    K_plate = D_mat' * (Z_total \ D_mat)
    # Enforce exact symmetry on K_plate to avoid roundoff-level asymmetry
    # tripping the solver's positive-definiteness checks
    K_plate = 0.5 * (K_plate + K_plate')

    # Distribute to 24×24 Ke (plate DOFs at positions 3, 4, 5 per node)
    plate_dofs = (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)
    @inbounds for j in 1:12, i in 1:12
        Ke[plate_dofs[i], plate_dofs[j]] += K_plate[i, j]
    end
    return Ke
end

# ---------------------------------------------------------------------------
# MacNeal twist compatibility correction (eq 17):
#   χ̃_xy = 2·χ_xy(GP) − χ_xy(0)
# Rewrites the χ_xy row of Bb at each Gauss point by subtracting the
# center-Jacobian contribution, doubling the GP contribution. This stiffens
# twist-dominated modes (per MacNeal's analysis of linear-twist bending).
# Pass in the center B_xy row computed once per element.
# ---------------------------------------------------------------------------
@inline function macneal_twist_correct!(Bb::AbstractMatrix{Float64},
                                         Bb_center_row::AbstractVector{Float64})
    @inbounds for j in 1:length(Bb_center_row)
        Bb[3, j] = 2.0 * Bb[3, j] - Bb_center_row[j]
    end
    return Bb
end

"""
    apply_macneal_warp_correction!(Ke, coords_3d, frame_axes, centroid; alpha=-1/3)

Apply a partial reverse-engineering of MacNeal 1978's warp correction to an
element K matrix that was computed on the projected (flat-best-fit) quad.

For the warped-quad calibration, Convention G (this function):
adding rigid-offset translation↔rotation coupling scaled by α at each corner's
warp height z_i (signed distance from the mean plane, in element-local frame)
matches Nastran KGG[θ_x, T_x] within 4–12% on a single warped iso CQUAD4 test.

It does NOT capture the second-order localised K[T_z, T_x] coupling at the
warped corner that MacNeal mentions ("additional normal forces, not additional
moments") — that requires further derivation. This is therefore a **partial**
correction; gated behind `JFEM_MACNEAL_WARP_ALPHA` (default 0 = off).

The transformation is applied in element-LOCAL frame:
  T_off[u_x_real, θ_y_real] = +α·z_i      ← node-i diag block
  T_off[u_y_real, θ_x_real] = −α·z_i      ← node-i diag block
  K_warp = T_off^T · K_flat · T_off

`frame_axes = (v1, v2, v3)` defines the element-local frame; `centroid` is the
3D centroid (used to compute z_i = (P_i − centroid) · v3).
"""
function apply_macneal_warp_correction!(Ke::AbstractMatrix{Float64},
                                         coords_3d::AbstractMatrix{Float64},
                                         frame_axes,
                                         centroid::AbstractVector{Float64};
                                         alpha::Float64=-1.0/3.0)
    v3 = frame_axes[3]
    z = ntuple(i -> begin
        d1 = coords_3d[i, 1] - centroid[1]
        d2 = coords_3d[i, 2] - centroid[2]
        d3 = coords_3d[i, 3] - centroid[3]
        d1*v3[1] + d2*v3[2] + d3*v3[3]
    end, 4)
    # Build T_off: 24×24 identity plus per-corner rigid offset, in element-local frame.
    # In local frame, corner offset along v3 (local z) by z_i, so:
    #   u_x_real = u_x + (α·z_i) * θ_y
    #   u_y_real = u_y + (-α·z_i) * θ_x
    # Apply K_warp = T^T K T in-place using two GEMM-like operations.
    T = Matrix{Float64}(I, 24, 24)
    @inbounds for n in 0:3
        zi = alpha * z[n + 1]
        T[6n + 1, 6n + 5] = +zi
        T[6n + 2, 6n + 4] = -zi
    end
    K_warp = T' * Ke * T
    @inbounds for j in 1:24, i in 1:24
        Ke[i, j] = K_warp[i, j]
    end
    return Ke
end

@inline function quad4_is_axis_aligned_rectangle(coords::AbstractMatrix; tol::Float64=1e-8)
    e12 = SVector(coords[2,1] - coords[1,1], coords[2,2] - coords[1,2])
    e23 = SVector(coords[3,1] - coords[2,1], coords[3,2] - coords[2,2])
    e34 = SVector(coords[4,1] - coords[3,1], coords[4,2] - coords[3,2])
    e41 = SVector(coords[1,1] - coords[4,1], coords[1,2] - coords[4,2])

    function cross2(a::SVector{2,Float64}, b::SVector{2,Float64})
        return a[1] * b[2] - a[2] * b[1]
    end

    Lmax = max(norm(e12), norm(e23), norm(e34), norm(e41), 1e-12)
    area_scale = Lmax * Lmax

    if abs(cross2(e12, e34)) > tol * area_scale || abs(cross2(e23, e41)) > tol * area_scale
        return false
    end
    if abs(dot(e12, e23)) > tol * area_scale ||
       abs(dot(e23, e34)) > tol * area_scale ||
       abs(dot(e34, e41)) > tol * area_scale ||
       abs(dot(e41, e12)) > tol * area_scale
        return false
    end

    if abs(norm(e12) - norm(e34)) > tol * Lmax || abs(norm(e23) - norm(e41)) > tol * Lmax
        return false
    end

    return true
end

@inline function adini_plate_basis(x::Float64, y::Float64)
    return SVector(
        1.0,
        x,
        y,
        x^2,
        x * y,
        y^2,
        x^3,
        x^2 * y,
        x * y^2,
        y^3,
        x^3 * y,
        x * y^3,
    )
end

@inline function adini_plate_basis_dx(x::Float64, y::Float64)
    return SVector(
        0.0,
        1.0,
        0.0,
        2.0 * x,
        y,
        0.0,
        3.0 * x^2,
        2.0 * x * y,
        y^2,
        0.0,
        3.0 * x^2 * y,
        y^3,
    )
end

@inline function adini_plate_basis_dy(x::Float64, y::Float64)
    return SVector(
        0.0,
        0.0,
        1.0,
        0.0,
        x,
        2.0 * y,
        0.0,
        x^2,
        2.0 * x * y,
        3.0 * y^2,
        x^3,
        3.0 * x * y^2,
    )
end

@inline function adini_plate_basis_dxx(x::Float64, y::Float64)
    return SVector(
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        6.0 * x,
        2.0 * y,
        0.0,
        0.0,
        6.0 * x * y,
        0.0,
    )
end

@inline function adini_plate_basis_dyy(x::Float64, y::Float64)
    return SVector(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        2.0 * x,
        6.0 * y,
        0.0,
        6.0 * x * y,
    )
end

@inline function adini_plate_basis_dxy(x::Float64, y::Float64)
    return SVector(
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        2.0 * x,
        2.0 * y,
        0.0,
        3.0 * x^2,
        3.0 * y^2,
    )
end

function adini_plate_interpolation_inverse(coords::AbstractMatrix)
    T = zeros(12, 12)
    @inbounds for a in 1:4
        x = coords[a, 1]
        y = coords[a, 2]
        row = (a - 1) * 3
        T[row + 1, :] .= adini_plate_basis(x, y)
        T[row + 2, :] .= adini_plate_basis_dx(x, y)
        T[row + 3, :] .= adini_plate_basis_dy(x, y)
    end
    try
        return inv(T)
    catch
        return nothing
    end
end

@inline function adini_plate_dof_transform()
    Tqp = zeros(12, 12)
    @inbounds for a in 1:4
        base = (a - 1) * 3
        Tqp[base + 1, base + 1] = 1.0
        Tqp[base + 2, base + 3] = 1.0
        Tqp[base + 3, base + 2] = -1.0
    end
    return Tqp
end

@inline function interp_2x2_gauss_sigma(sigma_mem_gp::AbstractMatrix, xi::Float64, eta::Float64)
    pt = 1.0 / sqrt(3.0)
    lx1 = (pt - xi) / (2.0 * pt)
    lx2 = (xi + pt) / (2.0 * pt)
    ly1 = (pt - eta) / (2.0 * pt)
    ly2 = (eta + pt) / (2.0 * pt)

    w1 = lx1 * ly1
    w2 = lx2 * ly1
    w3 = lx2 * ly2
    w4 = lx1 * ly2

    return SVector(
        w1 * sigma_mem_gp[1, 1] + w2 * sigma_mem_gp[2, 1] + w3 * sigma_mem_gp[3, 1] + w4 * sigma_mem_gp[4, 1],
        w1 * sigma_mem_gp[1, 2] + w2 * sigma_mem_gp[2, 2] + w3 * sigma_mem_gp[3, 2] + w4 * sigma_mem_gp[4, 2],
        w1 * sigma_mem_gp[1, 3] + w2 * sigma_mem_gp[2, 3] + w3 * sigma_mem_gp[3, 3] + w4 * sigma_mem_gp[4, 3],
    )
end

function stiffness_quad4_plate_adini_matrices(
    coords,
    Cm,
    Cb,
    Cs,
    h,
    E_ref;
    k6rot=100.0,
    drill_scale::Float64=1.0,
    ws::Union{Nothing,Quad4Workspace}=nothing,
    membrane_incomp::Bool=true,
    curvature_membrane=nothing,
    membrane_shear_center_row::Bool=false,
    material_shear_rotation::Float64=0.0,
    membrane_assumed_mode::Symbol=:none,
)
    if !quad4_is_axis_aligned_rectangle(coords)
        return stiffness_quad4_plate_dkq_matrices(
            coords, Cm, Cb, Cs, h, E_ref;
            k6rot=k6rot,
            drill_scale=drill_scale,
            ws=ws,
            membrane_incomp=membrane_incomp,
            curvature_membrane=curvature_membrane,
            membrane_shear_center_row=membrane_shear_center_row,
            material_shear_rotation=material_shear_rotation,
            membrane_assumed_mode=membrane_assumed_mode,
        )
    end

    if ws === nothing
        ws = create_quad4_workspace()
    end

    Cb_zero = ws.Cb_buf
    Cs_zero = ws.Cs_buf
    fill!(Cb_zero, 0.0)
    fill!(Cs_zero, 0.0)

    Ke = stiffness_quad4_matrices(
        coords,
        Cm,
        Cb_zero,
        Cs_zero,
        h,
        E_ref;
        k6rot=k6rot,
        drill_scale=drill_scale,
        Bmb=nothing,
        ws=ws,
        bending_incomp=false,
        shear_center_only=true,
        no_phi2=true,
        membrane_incomp=membrane_incomp,
        curvature_membrane=curvature_membrane,
        membrane_shear_center_row=membrane_shear_center_row,
        material_shear_rotation=material_shear_rotation,
        membrane_assumed_mode=membrane_assumed_mode,
    )

    maximum(abs, Cb) < 1e-30 && return Ke

    T_inv = adini_plate_interpolation_inverse(coords)
    T_inv === nothing && return stiffness_quad4_plate_dkq_matrices(
        coords, Cm, Cb, Cs, h, E_ref;
        k6rot=k6rot,
        drill_scale=drill_scale,
        ws=ws,
        membrane_incomp=membrane_incomp,
        curvature_membrane=curvature_membrane,
        membrane_shear_center_row=membrane_shear_center_row,
        material_shear_rotation=material_shear_rotation,
        membrane_assumed_mode=membrane_assumed_mode,
    )

    x_min = minimum(view(coords, :, 1))
    x_max = maximum(view(coords, :, 1))
    y_min = minimum(view(coords, :, 2))
    y_max = maximum(view(coords, :, 2))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    hx = 0.5 * (x_max - x_min)
    hy = 0.5 * (y_max - y_min)
    if hx <= 1e-12 || hy <= 1e-12
        return stiffness_quad4_plate_dkq_matrices(
            coords, Cm, Cb, Cs, h, E_ref;
            k6rot=k6rot,
            drill_scale=drill_scale,
            ws=ws,
            membrane_incomp=membrane_incomp,
            curvature_membrane=curvature_membrane,
            membrane_shear_center_row=membrane_shear_center_row,
            material_shear_rotation=material_shear_rotation,
            membrane_assumed_mode=membrane_assumed_mode,
        )
    end

    quad_pts = (
        (-0.8611363115940526, 0.34785484513745385),
        (-0.33998104358485626, 0.6521451548625461),
        (0.33998104358485626, 0.6521451548625461),
        (0.8611363115940526, 0.34785484513745385),
    )

    Kq = zeros(12, 12)
    Bq = zeros(3, 12)
    tmp = zeros(3, 12)
    Tqp = adini_plate_dof_transform()

    @inbounds for (xi, wi) in quad_pts, (eta, wj) in quad_pts
        x = x_mid + hx * xi
        y = y_mid + hy * eta
        jac = hx * hy * wi * wj

        Bq[1, :] .= T_inv' * adini_plate_basis_dxx(x, y)
        Bq[2, :] .= T_inv' * adini_plate_basis_dyy(x, y)
        Bq[3, :] .= T_inv' * (2.0 .* adini_plate_basis_dxy(x, y))

        mul!(tmp, Cb, Bq)
        Kq .+= (Bq' * tmp) .* jac
    end

    Kplate = Tqp' * Kq * Tqp
    plate_dofs = (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)
    @inbounds for j in 1:12, i in 1:12
        Ke[plate_dofs[i], plate_dofs[j]] += Kplate[i, j]
    end

    return Ke
end

function geometric_stiffness_quad4_plate_adini(coords::AbstractMatrix, sigma_mem::AbstractVector, h::Float64)
    sigma_gp = zeros(4, 3)
    @inbounds for gp in 1:4
        sigma_gp[gp, 1] = sigma_mem[1]
        sigma_gp[gp, 2] = sigma_mem[2]
        sigma_gp[gp, 3] = sigma_mem[3]
    end
    return geometric_stiffness_quad4_plate_adini(coords, sigma_gp, h)
end

function geometric_stiffness_quad4_plate_adini(coords::AbstractMatrix, sigma_mem_gp::AbstractMatrix, h::Float64)
    Kg = zeros(24, 24)
    h < 1e-30 && return Kg
    quad4_is_axis_aligned_rectangle(coords) || return geometric_stiffness_quad4(
        coords, sigma_mem_gp, h;
        trans_mode=:normal_only,
        rot_grad_scale=0.0,
        membrane_shear_center_row=false,
    )

    T_inv = adini_plate_interpolation_inverse(coords)
    T_inv === nothing && return geometric_stiffness_quad4(
        coords, sigma_mem_gp, h;
        trans_mode=:normal_only,
        rot_grad_scale=0.0,
        membrane_shear_center_row=false,
    )

    x_min = minimum(view(coords, :, 1))
    x_max = maximum(view(coords, :, 1))
    y_min = minimum(view(coords, :, 2))
    y_max = maximum(view(coords, :, 2))
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    hx = 0.5 * (x_max - x_min)
    hy = 0.5 * (y_max - y_min)
    if hx <= 1e-12 || hy <= 1e-12
        return Kg
    end

    quad_pts = (
        (-0.8611363115940526, 0.34785484513745385),
        (-0.33998104358485626, 0.6521451548625461),
        (0.33998104358485626, 0.6521451548625461),
        (0.8611363115940526, 0.34785484513745385),
    )

    Kq = zeros(12, 12)
    gx = zeros(12)
    gy = zeros(12)
    Tqp = adini_plate_dof_transform()

    @inbounds for (xi, wi) in quad_pts, (eta, wj) in quad_pts
        x = x_mid + hx * xi
        y = y_mid + hy * eta
        jac = hx * hy * wi * wj

        gx .= T_inv' * adini_plate_basis_dx(x, y)
        gy .= T_inv' * adini_plate_basis_dy(x, y)
        sigma = interp_2x2_gauss_sigma(sigma_mem_gp, xi, eta)
        s_xx = sigma[1]
        s_yy = sigma[2]
        s_xy = sigma[3]

        @inbounds for j in 1:12, i in 1:12
            Kq[i, j] += h * jac * (
                s_xx * gx[i] * gx[j] +
                s_yy * gy[i] * gy[j] +
                s_xy * (gx[i] * gy[j] + gy[i] * gx[j])
            )
        end
    end

    Kplate = Tqp' * Kq * Tqp
    plate_dofs = (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)
    @inbounds for j in 1:12, i in 1:12
        Kg[plate_dofs[i], plate_dofs[j]] += Kplate[i, j]
    end
    return Kg
end

function add_quad4_plate_dkq_bending!(
    Ke::AbstractMatrix,
    coords,
    Cb,
    Cs,
    A_beta_override=nothing,
)
    maximum(abs, Cb) < 1e-30 && return Ke

    A_beta = isnothing(A_beta_override) ? first(dkq_plate_edge_relation(coords, Cb, Cs)) : A_beta_override
    Hct_inv = inv(Cs)
    Hbar = dkq_plate_hbar_matrix(Cb)

    function fill_dkq_point_operators!(Bf_beta, Bf_alpha, Bc_beta, Bc_alpha, xi::Float64, eta::Float64)
        dNr, dNs = shape_derivs_quad(xi, eta)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11 * J22 - J12 * J21
        if abs(detJ) < 1e-12
            detJ = detJ < 0.0 ? -1e-12 : 1e-12
        end
        abs_detJ = abs(detJ)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        fill!(Bf_beta, 0.0)
        fill!(Bf_alpha, 0.0)
        fill!(Bc_beta, 0.0)
        fill!(Bc_alpha, 0.0)

        d2N_dxideta = (0.25, -0.25, 0.25, -0.25)
        for a in 1:4
            dN_dx = iJ11 * dNr[a] + iJ12 * dNs[a]
            dN_dy = iJ21 * dNr[a] + iJ22 * dNs[a]
            col = (a - 1) * 3
            Bf_beta[1, col + 3] = dN_dx
            Bf_beta[2, col + 2] = -dN_dy
            Bf_beta[3, col + 2] = -dN_dx
            Bf_beta[3, col + 3] = dN_dy

            d2N_xx = 2.0 * iJ11 * iJ12 * d2N_dxideta[a]
            d2N_yy = 2.0 * iJ21 * iJ22 * d2N_dxideta[a]
            d2N_xy = (iJ11 * iJ22 + iJ12 * iJ21) * d2N_dxideta[a]
            Bc_beta[1, col + 3] = Hbar[1, 1] * d2N_xx + Hbar[1, 2] * d2N_yy + Hbar[1, 3] * d2N_xy
            Bc_beta[2, col + 3] = Hbar[2, 1] * d2N_xx + Hbar[2, 2] * d2N_yy + Hbar[2, 3] * d2N_xy
            Bc_beta[1, col + 2] = -(Hbar[1, 4] * d2N_xx + Hbar[1, 5] * d2N_yy + Hbar[1, 6] * d2N_xy)
            Bc_beta[2, col + 2] = -(Hbar[2, 4] * d2N_xx + Hbar[2, 5] * d2N_yy + Hbar[2, 6] * d2N_xy)
        end

        dP_dxi, dP_deta = dkq_edge_shape_derivs(xi, eta)
        d2P_dxi2, d2P_deta2, d2P_dxideta = dkq_edge_shape_second_derivs(xi, eta)
        edge_c = zeros(4)
        edge_s = zeros(4)
        for e in 1:4
            i, j = ((1, 2), (2, 3), (3, 4), (4, 1))[e]
            dx = coords[j, 1] - coords[i, 1]
            dy = coords[j, 2] - coords[i, 2]
            L = sqrt(dx * dx + dy * dy)
            if L > 1e-12
                edge_c[e] = dx / L
                edge_s[e] = dy / L
            end
        end
        for e in 1:4
            c = edge_c[e]
            s = edge_s[e]
            dP_dx = iJ11 * dP_dxi[e] + iJ12 * dP_deta[e]
            dP_dy = iJ21 * dP_dxi[e] + iJ22 * dP_deta[e]
            Bf_alpha[1, e] = c * dP_dx
            Bf_alpha[2, e] = s * dP_dy
            Bf_alpha[3, e] = c * dP_dy + s * dP_dx

            d2P_xx = iJ11^2 * d2P_dxi2[e] + iJ12^2 * d2P_deta2[e] + 2.0 * iJ11 * iJ12 * d2P_dxideta[e]
            d2P_yy = iJ21^2 * d2P_dxi2[e] + iJ22^2 * d2P_deta2[e] + 2.0 * iJ21 * iJ22 * d2P_dxideta[e]
            d2P_xy = iJ11 * iJ21 * d2P_dxi2[e] + iJ12 * iJ22 * d2P_deta2[e] +
                     (iJ11 * iJ22 + iJ12 * iJ21) * d2P_dxideta[e]

            Bc_alpha[1, e] =
                Hbar[1, 1] * (c * d2P_xx) +
                Hbar[1, 2] * (c * d2P_yy) +
                Hbar[1, 3] * (c * d2P_xy) +
                Hbar[1, 4] * (s * d2P_xx) +
                Hbar[1, 5] * (s * d2P_yy) +
                Hbar[1, 6] * (s * d2P_xy)
            Bc_alpha[2, e] =
                Hbar[2, 1] * (c * d2P_xx) +
                Hbar[2, 2] * (c * d2P_yy) +
                Hbar[2, 3] * (c * d2P_xy) +
                Hbar[2, 4] * (s * d2P_xx) +
                Hbar[2, 5] * (s * d2P_yy) +
                Hbar[2, 6] * (s * d2P_xy)
        end

        return abs_detJ
    end

    Bf_beta = zeros(3, 12)
    Bf_alpha = zeros(3, 4)
    Bc_beta = zeros(2, 12)
    Bc_alpha = zeros(2, 4)
    tmp3x12 = zeros(3, 12)
    tmp3x4 = zeros(3, 4)
    tmp2x12 = zeros(2, 12)
    tmp2x4 = zeros(2, 4)
    Kf11 = zeros(12, 12)
    Kf12 = zeros(12, 4)
    Kf22 = zeros(4, 4)
    Kbb = zeros(12, 12)
    Kba = zeros(12, 4)
    Kca = zeros(4, 4)

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
    @inbounds for gp in gauss_pts
        abs_detJ = fill_dkq_point_operators!(Bf_beta, Bf_alpha, Bc_beta, Bc_alpha, gp[1], gp[2])

        ts_mul!(tmp3x12, Cb, Bf_beta)
        ts_mul_At_add!(Kf11, Bf_beta, tmp3x12, abs_detJ)

        ts_mul!(tmp3x4, Cb, Bf_alpha)
        ts_mul_At_add!(Kf12, Bf_beta, tmp3x4, abs_detJ)
        ts_mul_At_add!(Kf22, Bf_alpha, tmp3x4, abs_detJ)

        ts_mul!(tmp2x4, Hct_inv, Bc_alpha)
        ts_mul!(tmp2x12, Hct_inv, Bc_beta)
        ts_mul_At_add!(Kbb, Bc_beta, tmp2x12, abs_detJ)
        ts_mul_At_add!(Kba, Bc_beta, tmp2x4, abs_detJ)
        ts_mul_At_add!(Kca, Bc_alpha, tmp2x4, abs_detJ)
    end

    Kplate = Kf11 .+ Kbb .+
             (Kf12 .+ Kba) * A_beta .+
             A_beta' * (Kf12' .+ Kba') .+
             A_beta' * (Kf22 .+ Kca) * A_beta

    plate_dofs = (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)
    @inbounds for j in 1:12, i in 1:12
        Ke[plate_dofs[i], plate_dofs[j]] += Kplate[i, j]
    end

    return Ke
end

function stiffness_quad4_plate_dkq_matrices(
    coords,
    Cm,
    Cb,
    Cs,
    h,
    E_ref;
    k6rot=100.0,
    drill_scale::Float64=1.0,
    ws::Union{Nothing,Quad4Workspace}=nothing,
    membrane_incomp::Bool=true,
    curvature_membrane=nothing,
    membrane_shear_center_row::Bool=false,
    material_shear_rotation::Float64=0.0,
    membrane_assumed_mode::Symbol=:none,
    )
    if ws === nothing
        ws = create_quad4_workspace()
    end

    Cb_zero = ws.Cb_buf
    Cs_zero = ws.Cs_buf
    fill!(Cb_zero, 0.0)
    fill!(Cs_zero, 0.0)

    Ke = stiffness_quad4_matrices(
        coords,
        Cm,
        Cb_zero,
        Cs_zero,
        h,
        E_ref;
        k6rot=k6rot,
        drill_scale=drill_scale,
        Bmb=nothing,
        ws=ws,
        bending_incomp=false,
        shear_center_only=true,
        no_phi2=true,
        membrane_incomp=membrane_incomp,
        curvature_membrane=curvature_membrane,
        membrane_shear_center_row=membrane_shear_center_row,
        material_shear_rotation=material_shear_rotation,
        membrane_assumed_mode=membrane_assumed_mode,
    )
    return add_quad4_plate_dkq_bending!(Ke, coords, Cb, Cs)
end

function stiffness_quad4_plate_dkmq_matrices(
    coords,
    Cm,
    Cb,
    Cs,
    h,
    E_ref;
    k6rot=100.0,
    drill_scale::Float64=1.0,
    ws::Union{Nothing,Quad4Workspace}=nothing,
    membrane_incomp::Bool=true,
    curvature_membrane=nothing,
    membrane_shear_center_row::Bool=false,
    material_shear_rotation::Float64=0.0,
    membrane_assumed_mode::Symbol=:none,
)
    if ws === nothing
        ws = create_quad4_workspace()
    end

    Ke = stiffness_quad4_membrane_normal_rot_matrices(
        coords,
        Cm,
        h;
        curvature_membrane=curvature_membrane,
    )
    A_beta, _, _, _ = dkmq_flat_plate_edge_relation(coords, Cb, Cs)
    add_quad4_plate_dkmq_exact_shear!(Ke, coords, Cb, Cs, h)
    return add_quad4_plate_dkq_bending!(Ke, coords, Cb, Cs, A_beta)
end

function add_quad4_membrane_translation_geometric!(
    Kg::AbstractMatrix,
    coords::AbstractMatrix,
    sigma_mem_gp::AbstractMatrix,
    h::Float64,
)
    h < 1e-30 && return Kg

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (
        (SVector(-pt, -pt), 1),
        (SVector(pt, -pt), 2),
        (SVector(pt, pt), 3),
        (SVector(-pt, pt), 4),
    )

    @inbounds for (gp, gp_idx) in gauss_pts
        xi = gp[1]
        eta = gp[2]
        dNr, dNs = shape_derivs_quad(xi, eta)

        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11 * J22 - J12 * J21
        if abs(detJ) < 1e-12
            detJ = detJ < 0.0 ? -1e-12 : 1e-12
        end
        abs_detJ = abs(detJ)
        inv_det = 1.0 / detJ
        iJ11 = J22 * inv_det
        iJ12 = -J12 * inv_det
        iJ21 = -J21 * inv_det
        iJ22 = J11 * inv_det

        s_xx = sigma_mem_gp[gp_idx, 1]
        s_yy = sigma_mem_gp[gp_idx, 2]
        s_xy = sigma_mem_gp[gp_idx, 3]
        scale = h * abs_detJ
        for j in 1:4
            dNj_dx = iJ11 * dNr[j] + iJ12 * dNs[j]
            dNj_dy = iJ21 * dNr[j] + iJ22 * dNs[j]
            for i in 1:4
                dNi_dx = iJ11 * dNr[i] + iJ12 * dNs[i]
                dNi_dy = iJ21 * dNr[i] + iJ22 * dNs[i]
                val = scale * (
                    s_xx * dNi_dx * dNj_dx +
                    s_yy * dNi_dy * dNj_dy +
                    s_xy * (dNi_dx * dNj_dy + dNi_dy * dNj_dx)
                )
                bi = (i - 1) * 6
                bj = (j - 1) * 6
                Kg[bi + 1, bj + 1] += val
                Kg[bi + 2, bj + 2] += val
            end
        end
    end

    return Kg
end

function geometric_stiffness_quad4_plate_with_edge_relation(
    coords::AbstractMatrix,
    sigma_mem_gp::AbstractMatrix,
    h::Float64,
    A_beta::AbstractMatrix,
    edge_c::AbstractVector,
    edge_s::AbstractVector,
    include_membrane_translations::Bool=true,
)
    Kg = zeros(24, 24)
    h < 1e-30 && return Kg

    G_beta = zeros(2, 12)
    G_alpha = zeros(2, 4)
    G_eff = zeros(2, 12)
    K_plate = zeros(12, 12)
    plate_dofs = (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (
        (SVector(-pt, -pt), 1),
        (SVector(pt, -pt), 2),
        (SVector(pt, pt), 3),
        (SVector(-pt, pt), 4),
    )

    @inbounds for (gp, gp_idx) in gauss_pts
        xi = gp[1]
        eta = gp[2]
        dNr, dNs = shape_derivs_quad(xi, eta)
        Nvals = SVector(
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        )

        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11 * J22 - J12 * J21
        abs_detJ = abs(detJ)
        abs_detJ < 1e-12 && (abs_detJ = 1e-12)

        fill!(G_beta, 0.0)
        fill!(G_alpha, 0.0)
        for a in 1:4
            col = (a - 1) * 3
            G_beta[1, col + 3] = -Nvals[a]
            G_beta[2, col + 2] = Nvals[a]
        end

        P1 = 0.5 * (1.0 - eta) * (1.0 - xi^2)
        P2 = 0.5 * (1.0 + xi) * (1.0 - eta^2)
        P3 = 0.5 * (1.0 + eta) * (1.0 - xi^2)
        P4 = 0.5 * (1.0 - xi) * (1.0 - eta^2)
        Pvals = (P1, P2, P3, P4)
        for e in 1:4
            G_alpha[1, e] = -edge_c[e] * Pvals[e]
            G_alpha[2, e] = -edge_s[e] * Pvals[e]
        end

        copyto!(G_eff, G_beta)
        for e in 1:4
            for j in 1:12
                coeff = A_beta[e, j]
                if coeff != 0.0
                    G_eff[1, j] += G_alpha[1, e] * coeff
                    G_eff[2, j] += G_alpha[2, e] * coeff
                end
            end
        end

        s_xx = sigma_mem_gp[gp_idx, 1]
        s_yy = sigma_mem_gp[gp_idx, 2]
        s_xy = sigma_mem_gp[gp_idx, 3]
        @inbounds for j in 1:12, i in 1:12
            K_plate[i, j] += h * abs_detJ * (
                s_xx * G_eff[1, i] * G_eff[1, j] +
                s_yy * G_eff[2, i] * G_eff[2, j] +
                s_xy * (G_eff[1, i] * G_eff[2, j] + G_eff[2, i] * G_eff[1, j])
            )
        end
    end

    @inbounds for j in 1:12, i in 1:12
        Kg[plate_dofs[i], plate_dofs[j]] += K_plate[i, j]
    end
    if include_membrane_translations
        add_quad4_membrane_translation_geometric!(Kg, coords, sigma_mem_gp, h)
    end
    return Kg
end

function geometric_stiffness_quad4_plate_dkmq(coords::AbstractMatrix, sigma_mem::AbstractVector, h::Float64,
                                              Cb::AbstractMatrix, Cs::AbstractMatrix)
    sigma_gp = zeros(4, 3)
    @inbounds for gp in 1:4
        sigma_gp[gp, 1] = sigma_mem[1]
        sigma_gp[gp, 2] = sigma_mem[2]
        sigma_gp[gp, 3] = sigma_mem[3]
    end
    return geometric_stiffness_quad4_plate_dkmq(coords, sigma_gp, h, Cb, Cs)
end

function geometric_stiffness_quad4_plate_dkmq(coords::AbstractMatrix, sigma_mem_gp::AbstractMatrix, h::Float64,
                                              Cb::AbstractMatrix, Cs::AbstractMatrix)
    if h < 1e-30 || maximum(abs, Cb) < 1e-30
        return zeros(24, 24)
    end
    A_beta, edge_c, edge_s, _ = dkmq_flat_plate_edge_relation(coords, Cb, Cs)
    return geometric_stiffness_quad4_plate_with_edge_relation(coords, sigma_mem_gp, h, A_beta, edge_c, edge_s)
end

function geometric_stiffness_quad4_plate_dkq(coords::AbstractMatrix, sigma_mem::AbstractVector, h::Float64,
                                             Cb::AbstractMatrix, Cs::AbstractMatrix)
    sigma_gp = zeros(4, 3)
    @inbounds for gp in 1:4
        sigma_gp[gp, 1] = sigma_mem[1]
        sigma_gp[gp, 2] = sigma_mem[2]
        sigma_gp[gp, 3] = sigma_mem[3]
    end
    return geometric_stiffness_quad4_plate_dkq(coords, sigma_gp, h, Cb, Cs)
end

function geometric_stiffness_quad4_plate_dkq(coords::AbstractMatrix, sigma_mem_gp::AbstractMatrix, h::Float64,
                                             Cb::AbstractMatrix, Cs::AbstractMatrix)
    A_beta, edge_c, edge_s, _ = dkq_plate_edge_relation(coords, Cb, Cs)
    return geometric_stiffness_quad4_plate_with_edge_relation(coords, sigma_mem_gp, h, A_beta, edge_c, edge_s)
end

function compute_principal_2d(s11, s22, s12)
    s_avg = (s11 + s22) / 2.0
    radius = sqrt(((s11 - s22) / 2.0)^2 + s12^2)
    return s_avg + radius, s_avg - radius
end

function quad4_mitc4_center_shear_resultant(coords, u_elem, G, h; ts_t=5.0/6.0)
    tying_pts = (SVector(0.0, -1.0), SVector(0.0, 1.0), SVector(-1.0, 0.0), SVector(1.0, 0.0))
    Bs_tp = zeros(4, 24)
    for tp_idx in 1:4
        xi_tp, eta_tp = tying_pts[tp_idx][1], tying_pts[tp_idx][2]
        dNr, dNs = shape_derivs_quad(xi_tp, eta_tp)
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        N_tp = (
            0.25*(1.0-xi_tp)*(1.0-eta_tp),
            0.25*(1.0+xi_tp)*(1.0-eta_tp),
            0.25*(1.0+xi_tp)*(1.0+eta_tp),
            0.25*(1.0-xi_tp)*(1.0+eta_tp),
        )
        if tp_idx <= 2
            for k in 1:4
                idx = (k-1)*6
                Bs_tp[tp_idx, idx+3] = dNr[k]
                Bs_tp[tp_idx, idx+4] = -J12 * N_tp[k]
                Bs_tp[tp_idx, idx+5] =  J11 * N_tp[k]
            end
        else
            for k in 1:4
                idx = (k-1)*6
                Bs_tp[tp_idx, idx+3] = dNs[k]
                Bs_tp[tp_idx, idx+4] = -J22 * N_tp[k]
                Bs_tp[tp_idx, idx+5] =  J21 * N_tp[k]
            end
        end
    end

    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J11c = dNr_c[1]*coords[1,1] + dNr_c[2]*coords[2,1] + dNr_c[3]*coords[3,1] + dNr_c[4]*coords[4,1]
    J12c = dNr_c[1]*coords[1,2] + dNr_c[2]*coords[2,2] + dNr_c[3]*coords[3,2] + dNr_c[4]*coords[4,2]
    J21c = dNs_c[1]*coords[1,1] + dNs_c[2]*coords[2,1] + dNs_c[3]*coords[3,1] + dNs_c[4]*coords[4,1]
    J22c = dNs_c[1]*coords[1,2] + dNs_c[2]*coords[2,2] + dNs_c[3]*coords[3,2] + dNs_c[4]*coords[4,2]
    detJc = J11c*J22c - J12c*J21c
    abs_detJc = max(abs(detJc), 1e-30)
    inv_detc = 1.0 / detJc
    iJ11c =  J22c * inv_detc
    iJ12c = -J12c * inv_detc
    iJ21c = -J21c * inv_detc
    iJ22c =  J11c * inv_detc

    Bs_cov = zeros(2, 24)
    for j in 1:24
        Bs_cov[1,j] = 0.5 * (Bs_tp[1,j] + Bs_tp[2,j])
        Bs_cov[2,j] = 0.5 * (Bs_tp[3,j] + Bs_tp[4,j])
    end

    phi2_shear = 1.0
    _alpha = PHI2_ALPHA[]
    if _alpha > 0.0
        L_char_sq = max(4.0 * abs_detJc, 1e-30)
        phi2_shear = min(1.0, _alpha * h^2 / L_char_sq)
    end

    gamma_cov = Bs_cov * u_elem
    k_shear = ts_t * G * h
    t11 = k_shear * iJ11c
    t12 = k_shear * iJ12c
    t21 = k_shear * iJ21c
    t22 = k_shear * iJ22c
    Q_cov = phi2_shear .* [
        iJ11c*t11 + iJ21c*t21  iJ11c*t12 + iJ21c*t22;
        iJ12c*t11 + iJ22c*t21  iJ12c*t12 + iJ22c*t22
    ] * gamma_cov

    return [
        J11c * Q_cov[1] + J21c * Q_cov[2],
        J12c * Q_cov[1] + J22c * Q_cov[2],
    ]
end

function stress_strain_quad4(coords, u_elem, E, nu, h, t_shell; bend_ratio=1.0, Cm_override=nothing, for_kg=false, curvature_membrane=nothing, membrane_shear_center_row::Bool=false, material_shear_rotation::Float64=0.0, membrane_assumed_mode::Symbol=:none, membrane_incomp_center_jacobian::Bool=false)
    const_mem = E / (1 - nu^2)
    D_mem = const_mem .* [1 nu 0; nu 1 0; 0 0 (1-nu)/2]
    # For PCOMP elements, use CLT Cm for incompatible mode condensation
    # (must match the Cm used in stiffness assembly for consistent strain recovery)
    Cm = isnothing(Cm_override) ? D_mem * h : Cm_override

    dNr, dNs = shape_derivs_quad(0.0, 0.0)
    J = [dNr'; dNs'] * coords
    invJ = inv(J); dN_dxy = invJ * [dNr'; dNs']
    iJ11c = invJ[1,1]; iJ12c = invJ[1,2]
    iJ21c = invJ[2,1]; iJ22c = invJ[2,2]

    Bm = zeros(3, 24); Bb = zeros(3, 24)

    for k in 1:4
        idx = (k-1)*6
        N_k = 0.25
        Bm[1, idx+1]=dN_dxy[1,k]; Bm[2, idx+2]=dN_dxy[2,k]
        Bm[3, idx+1]=dN_dxy[2,k]; Bm[3, idx+2]=dN_dxy[1,k]
        if curvature_membrane !== nothing
            Bm[1, idx+3] = -N_k * curvature_membrane[1]
            Bm[2, idx+3] = -N_k * curvature_membrane[2]
            Bm[3, idx+3] = -2.0 * N_k * curvature_membrane[3]
        end
        Bb[1, idx+5] = dN_dxy[1,k];
        Bb[2, idx+4] = -dN_dxy[2,k];
        Bb[3, idx+5] = dN_dxy[2,k];
        Bb[3, idx+4] = -dN_dxy[1,k];
    end

    # For Kg assembly, use compatible strain at center only (no incompatible modes).
    # Incompatible modes are internal bubble functions that improve element stiffness
    # but should not contribute to the physical membrane stress used for Kg.
    if for_kg
        eps_mem = Bm * u_elem
        kappa = Bb * u_elem
        N = Cm * eps_mem
        M = -bend_ratio * (D_mem * kappa) * (h^3/12.0)
        G = E / (2*(1+nu))
        Q = bend_ratio <= 1e-12 ? [0.0, 0.0] : quad4_mitc4_center_shear_resultant(coords, u_elem, G, h)
        z1 = -h/2.0; z2 = h/2.0
        strain_z1 = eps_mem .+ z1 .* kappa
        stress_z1 = D_mem * strain_z1
        strain_z2 = eps_mem .+ z2 .* kappa
        stress_z2 = D_mem * strain_z2
        return N, M, Q, stress_z1, stress_z2, strain_z1, strain_z2
    end

    # Recover incompatible mode amplitudes via static condensation
    # α = -K_bb^{-1} * K_ba * u  (K_ba = K_ab')
    # Recompute K_ab and K_bb (membrane incompatible coupling)
    K_ab_sr = zeros(24, 4); K_bb_sr = zeros(4, 4)
    pt = 1.0/sqrt(3.0)
    gauss_pts = [-pt -pt; pt -pt; pt pt; -pt pt]
    for i in 1:4
        r, s = gauss_pts[i,1], gauss_pts[i,2]
        dNr_g, dNs_g = shape_derivs_quad(r, s)
        J_g = [dNr_g'; dNs_g'] * coords
        detJ_g = abs(det(J_g))
        if detJ_g < 1e-12; detJ_g = 1e-12; end
        iJ = inv(J_g)
        dN_dxy_g = iJ * [dNr_g'; dNs_g']

        Bm_g = zeros(3, 24)
        for k in 1:4
            idx = (k-1)*6
            N_k = 0.25*(1 + (k==2||k==3 ? r : -r))*(1 + (k>=3 ? s : -s))
            Bm_g[1, idx+1] = dN_dxy_g[1,k]; Bm_g[2, idx+2] = dN_dxy_g[2,k]
            Bm_g[3, idx+1] = dN_dxy_g[2,k]; Bm_g[3, idx+2] = dN_dxy_g[1,k]
            if curvature_membrane !== nothing
                Bm_g[1, idx+3] = -N_k * curvature_membrane[1]
                Bm_g[2, idx+3] = -N_k * curvature_membrane[2]
                Bm_g[3, idx+3] = -2.0 * N_k * curvature_membrane[3]
            end
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(Bm_g, dN_dxy[1,:], dN_dxy[2,:], curvature_membrane, material_shear_rotation)
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
            apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
        end

        Bi = zeros(3, 4)
        fill_quad4_membrane_incompatible_B!(
            Bi,
            r,
            s,
            iJ[1,1],
            iJ[1,2],
            iJ[2,1],
            iJ[2,2],
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            membrane_incomp_center_jacobian,
        )

        K_ab_sr .+= (Bm_g' * Cm * Bi) .* detJ_g
        K_bb_sr .+= (Bi' * Cm * Bi) .* detJ_g
    end

    alpha = -(K_bb_sr \ (K_ab_sr' * u_elem))

    # Incompatible mode B-matrix at center (ξ=η=0)
    # φ1 = 1-ξ², dφ1/dξ = -2ξ = 0 at center; φ2 = 1-η², dφ2/dη = -2η = 0 at center
    # So the incompatible mode derivatives are zero at center.
    # The strain correction from incompatible modes is zero at center.
    # BUT the forces N = ∫ σ dA are affected because the incompatible modes
    # change the strain field at the Gauss points.

    # For stress recovery, compute the average strain including incompatible modes
    # by integrating over Gauss points
    eps_mem_avg = zeros(3)
    kappa_avg = zeros(3)
    total_area = 0.0
    for i in 1:4
        r, s = gauss_pts[i,1], gauss_pts[i,2]
        dNr_g, dNs_g = shape_derivs_quad(r, s)
        J_g = [dNr_g'; dNs_g'] * coords
        detJ_g = abs(det(J_g))
        if detJ_g < 1e-12; detJ_g = 1e-12; end
        iJ = inv(J_g)
        dN_dxy_g = iJ * [dNr_g'; dNs_g']

        # Standard membrane strain at this GP
        Bm_g = zeros(3, 24)
        for k in 1:4
            idx = (k-1)*6
            Bm_g[1, idx+1] = dN_dxy_g[1,k]; Bm_g[2, idx+2] = dN_dxy_g[2,k]
            Bm_g[3, idx+1] = dN_dxy_g[2,k]; Bm_g[3, idx+2] = dN_dxy_g[1,k]
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(Bm_g, dN_dxy[1,:], dN_dxy[2,:], nothing, material_shear_rotation)
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, nothing)
            apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
        end

        # Bending strain at this GP
        Bb_g = zeros(3, 24)
        for k in 1:4
            idx = (k-1)*6
            Bb_g[1, idx+5] = dN_dxy_g[1,k]
            Bb_g[2, idx+4] = -dN_dxy_g[2,k]
            Bb_g[3, idx+5] = dN_dxy_g[2,k]
            Bb_g[3, idx+4] = -dN_dxy_g[1,k]
        end

        # Incompatible mode strain at this GP
        Bi = zeros(3, 4)
        fill_quad4_membrane_incompatible_B!(
            Bi,
            r,
            s,
            iJ[1,1],
            iJ[1,2],
            iJ[2,1],
            iJ[2,2],
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            membrane_incomp_center_jacobian,
        )

        eps_gp = Bm_g * u_elem .+ Bi * alpha
        kappa_gp = Bb_g * u_elem

        eps_mem_avg .+= eps_gp .* detJ_g
        kappa_avg .+= kappa_gp .* detJ_g
        total_area += detJ_g
    end
    eps_mem_avg ./= total_area
    kappa_avg ./= total_area

    eps_mem = eps_mem_avg
    kappa = kappa_avg

    N = Cm * eps_mem
    M = -bend_ratio * (D_mem * kappa) * (h^3/12.0)

    G = E / (2*(1+nu))
    Q = bend_ratio <= 1e-12 ? [0.0, 0.0] : quad4_mitc4_center_shear_resultant(coords, u_elem, G, h)

    z1 = -h/2.0; z2 = h/2.0

    strain_z1 = eps_mem .+ z1 .* kappa
    stress_z1 = D_mem * strain_z1
    strain_z2 = eps_mem .+ z2 .* kappa
    stress_z2 = D_mem * strain_z2

    return N, M, Q, stress_z1, stress_z2, strain_z1, strain_z2
end

function quad4_bilinear_corner_forces(coords, u_elem, E, nu, h;
                                      bend_ratio=1.0,
                                      Cm_override=nothing,
                                      Cb_override=nothing,
                                      curvature_membrane=nothing,
                                      membrane_shear_center_row::Bool=false,
                                      material_shear_rotation::Float64=0.0,
                                      membrane_assumed_mode::Symbol=:none,
                                      membrane_incomp_center_jacobian::Bool=false)
    const_mem = E / (1 - nu^2)
    D_mem = const_mem .* [1 nu 0; nu 1 0; 0 0 (1-nu)/2]
    Cm = isnothing(Cm_override) ? D_mem * h : Cm_override
    Cb = isnothing(Cb_override) ? bend_ratio * D_mem * (h^3 / 12.0) : Cb_override

    dNr, dNs = shape_derivs_quad(0.0, 0.0)
    J = [dNr'; dNs'] * coords
    invJ = inv(J)
    dN_dxy = invJ * [dNr'; dNs']
    iJ11c = invJ[1,1]; iJ12c = invJ[1,2]
    iJ21c = invJ[2,1]; iJ22c = invJ[2,2]

    pt = 1.0 / sqrt(3.0)
    gauss_pts = [-pt -pt; pt -pt; pt pt; -pt pt]

    K_ab_sr = zeros(24, 4)
    K_bb_sr = zeros(4, 4)
    for i in 1:4
        r, s = gauss_pts[i,1], gauss_pts[i,2]
        dNr_g, dNs_g = shape_derivs_quad(r, s)
        J_g = [dNr_g'; dNs_g'] * coords
        detJ_g = abs(det(J_g))
        if detJ_g < 1e-12
            detJ_g = 1e-12
        end
        iJ = inv(J_g)
        dN_dxy_g = iJ * [dNr_g'; dNs_g']

        Bm_g = zeros(3, 24)
        for k in 1:4
            idx = (k-1)*6
            N_k = 0.25 * (1 + (k==2 || k==3 ? r : -r)) * (1 + (k>=3 ? s : -s))
            Bm_g[1, idx+1] = dN_dxy_g[1,k]
            Bm_g[2, idx+2] = dN_dxy_g[2,k]
            Bm_g[3, idx+1] = dN_dxy_g[2,k]
            Bm_g[3, idx+2] = dN_dxy_g[1,k]
            if curvature_membrane !== nothing
                Bm_g[1, idx+3] = -N_k * curvature_membrane[1]
                Bm_g[2, idx+3] = -N_k * curvature_membrane[2]
                Bm_g[3, idx+3] = -2.0 * N_k * curvature_membrane[3]
            end
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(Bm_g, dN_dxy[1,:], dN_dxy[2,:], curvature_membrane, material_shear_rotation)
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
            apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
        end

        Bi = zeros(3, 4)
        fill_quad4_membrane_incompatible_B!(
            Bi,
            r,
            s,
            iJ[1,1],
            iJ[1,2],
            iJ[2,1],
            iJ[2,2],
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            membrane_incomp_center_jacobian,
        )

        K_ab_sr .+= (Bm_g' * Cm * Bi) .* detJ_g
        K_bb_sr .+= (Bi' * Cm * Bi) .* detJ_g
    end

    alpha = -(K_bb_sr \ (K_ab_sr' * u_elem))
    N_gp = zeros(4, 3)
    M_gp = zeros(4, 3)

    for i in 1:4
        r, s = gauss_pts[i,1], gauss_pts[i,2]
        dNr_g, dNs_g = shape_derivs_quad(r, s)
        J_g = [dNr_g'; dNs_g'] * coords
        iJ = inv(J_g)
        dN_dxy_g = iJ * [dNr_g'; dNs_g']

        Bm_g = zeros(3, 24)
        for k in 1:4
            idx = (k-1)*6
            Bm_g[1, idx+1] = dN_dxy_g[1,k]
            Bm_g[2, idx+2] = dN_dxy_g[2,k]
            Bm_g[3, idx+1] = dN_dxy_g[2,k]
            Bm_g[3, idx+2] = dN_dxy_g[1,k]
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(Bm_g, dN_dxy[1,:], dN_dxy[2,:], nothing, material_shear_rotation)
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, nothing)
            apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
        end

        Bb_g = zeros(3, 24)
        for k in 1:4
            idx = (k-1)*6
            Bb_g[1, idx+5] = dN_dxy_g[1,k]
            Bb_g[2, idx+4] = -dN_dxy_g[2,k]
            Bb_g[3, idx+5] = dN_dxy_g[2,k]
            Bb_g[3, idx+4] = -dN_dxy_g[1,k]
        end

        Bi = zeros(3, 4)
        fill_quad4_membrane_incompatible_B!(
            Bi,
            r,
            s,
            iJ[1,1],
            iJ[1,2],
            iJ[2,1],
            iJ[2,2],
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            membrane_incomp_center_jacobian,
        )

        eps_gp = Bm_g * u_elem .+ Bi * alpha
        kappa_gp = Bb_g * u_elem
        N_gp[i, :] .= Cm * eps_gp
        M_gp[i, :] .= -Cb * kappa_gp
    end

    corner_points = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    N_corners = zeros(4, 3)
    M_corners = zeros(4, 3)
    for (i, (r, s)) in enumerate(corner_points)
        N_corners[i, :] .= interp_2x2_gauss_sigma(N_gp, r, s)
        M_corners[i, :] .= interp_2x2_gauss_sigma(M_gp, r, s)
    end

    return N_corners, M_corners
end

function quad4_membrane_force_field(coords, u_elem, E, nu, h;
                                    Cm_override=nothing,
                                    compatible_only=false,
                                    use_incompatible_modes::Bool=true,
                                    use_enhanced_modes::Bool=false,
                                    curvature_membrane=nothing,
                                    membrane_shear_center_row::Bool=false,
                                    material_shear_rotation::Float64=0.0,
                                    membrane_assumed_mode::Symbol=:none,
                                    membrane_incomp_center_jacobian::Bool=false)
    const_mem = E / (1 - nu^2)
    D_mem = const_mem .* [1 nu 0; nu 1 0; 0 0 (1-nu)/2]
    Cm = isnothing(Cm_override) ? D_mem * h : Cm_override

    pt = 1.0 / sqrt(3.0)
    gauss_pts = [-pt -pt; pt -pt; pt pt; -pt pt]
    total_area = 0.0
    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J_c = [dNr_c'; dNs_c'] * coords
    invJ_c = inv(J_c)
    dN_dxy_c = invJ_c * [dNr_c'; dNs_c']
    iJ11c = invJ_c[1,1]; iJ12c = invJ_c[1,2]
    iJ21c = invJ_c[2,1]; iJ22c = invJ_c[2,2]

    alpha = zeros(use_enhanced_modes ? 6 : 4)
    if !compatible_only && (use_enhanced_modes || use_incompatible_modes)
        K_ab_sr = zeros(24, 4)
        K_bb_sr = zeros(4, 4)
        if use_enhanced_modes
            K_ab_sr = zeros(24, 6)
            K_bb_sr = zeros(6, 6)
        end
        for i in 1:4
            r, s = gauss_pts[i,1], gauss_pts[i,2]
            dNr_g, dNs_g = shape_derivs_quad(r, s)
            J_g = [dNr_g'; dNs_g'] * coords
            detJ_g = abs(det(J_g))
            if detJ_g < 1e-12; detJ_g = 1e-12; end
            iJ = inv(J_g)
            dN_dxy_g = iJ * [dNr_g'; dNs_g']

            Bm_g = zeros(3, 24)
            for k in 1:4
                idx = (k-1)*6
                N_k = 0.25*(1 + (k==2||k==3 ? r : -r))*(1 + (k>=3 ? s : -s))
                Bm_g[1, idx+1] = dN_dxy_g[1,k]
                Bm_g[2, idx+2] = dN_dxy_g[2,k]
                Bm_g[3, idx+1] = dN_dxy_g[2,k]
                Bm_g[3, idx+2] = dN_dxy_g[1,k]
                if curvature_membrane !== nothing
                    Bm_g[1, idx+3] = -N_k * curvature_membrane[1]
                    Bm_g[2, idx+3] = -N_k * curvature_membrane[2]
                    Bm_g[3, idx+3] = -2.0 * N_k * curvature_membrane[3]
                end
            end
            if membrane_shear_center_row
                project_material_membrane_shear!(
                    Bm_g,
                    dN_dxy_c[1,:],
                    dN_dxy_c[2,:],
                    curvature_membrane,
                    material_shear_rotation,
                )
            elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
                apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
            end

            if use_enhanced_modes
                Bi = zeros(3, 6)
                fill_quad4_membrane_enhanced_B!(
                    Bi,
                    r,
                    s,
                    iJ[1,1],
                    iJ[1,2],
                    iJ[2,1],
                    iJ[2,2],
                    iJ11c,
                    iJ12c,
                    iJ21c,
                    iJ22c,
                    membrane_incomp_center_jacobian,
                )
            else
                Bi = zeros(3, 4)
                fill_quad4_membrane_incompatible_B!(
                    Bi,
                    r,
                    s,
                    iJ[1,1],
                    iJ[1,2],
                    iJ[2,1],
                    iJ[2,2],
                    iJ11c,
                    iJ12c,
                    iJ21c,
                    iJ22c,
                    membrane_incomp_center_jacobian,
                )
            end

            K_ab_sr .+= (Bm_g' * Cm * Bi) .* detJ_g
            K_bb_sr .+= (Bi' * Cm * Bi) .* detJ_g
        end
        alpha = -(K_bb_sr \ (K_ab_sr' * u_elem))
    end

    N_gp = zeros(4, 3)
    N_avg = zeros(3)
    area_w = zeros(4)

    for i in 1:4
        r, s = gauss_pts[i,1], gauss_pts[i,2]
        dNr_g, dNs_g = shape_derivs_quad(r, s)
        J_g = [dNr_g'; dNs_g'] * coords
        detJ_g = abs(det(J_g))
        if detJ_g < 1e-12; detJ_g = 1e-12; end
        iJ = inv(J_g)
        dN_dxy_g = iJ * [dNr_g'; dNs_g']

        Bm_g = zeros(3, 24)
        for k in 1:4
            idx = (k-1)*6
            N_k = 0.25*(1 + (k==2||k==3 ? r : -r))*(1 + (k>=3 ? s : -s))
            Bm_g[1, idx+1] = dN_dxy_g[1,k]
            Bm_g[2, idx+2] = dN_dxy_g[2,k]
            Bm_g[3, idx+1] = dN_dxy_g[2,k]
            Bm_g[3, idx+2] = dN_dxy_g[1,k]
            if curvature_membrane !== nothing
                Bm_g[1, idx+3] = -N_k * curvature_membrane[1]
                Bm_g[2, idx+3] = -N_k * curvature_membrane[2]
                Bm_g[3, idx+3] = -2.0 * N_k * curvature_membrane[3]
            end
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(
                Bm_g,
                dN_dxy_c[1,:],
                dN_dxy_c[2,:],
                curvature_membrane,
                material_shear_rotation,
            )
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
            apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
        end

        eps_gp = Bm_g * u_elem
        if !compatible_only && (use_enhanced_modes || use_incompatible_modes)
            if use_enhanced_modes
                Bi = zeros(3, 6)
                fill_quad4_membrane_enhanced_B!(
                    Bi,
                    r,
                    s,
                    iJ[1,1],
                    iJ[1,2],
                    iJ[2,1],
                    iJ[2,2],
                    iJ11c,
                    iJ12c,
                    iJ21c,
                    iJ22c,
                    membrane_incomp_center_jacobian,
                )
            else
                Bi = zeros(3, 4)
                fill_quad4_membrane_incompatible_B!(
                    Bi,
                    r,
                    s,
                    iJ[1,1],
                    iJ[1,2],
                    iJ[2,1],
                    iJ[2,2],
                    iJ11c,
                    iJ12c,
                    iJ21c,
                    iJ22c,
                    membrane_incomp_center_jacobian,
                )
            end
            eps_gp .+= Bi * alpha
        end

        N_vec = Cm * eps_gp
        N_gp[i, 1] = N_vec[1]
        N_gp[i, 2] = N_vec[2]
        N_gp[i, 3] = N_vec[3]
        N_avg .+= N_vec .* detJ_g
        area_w[i] = detJ_g
        total_area += detJ_g
    end

    if total_area > 0.0
        N_avg ./= total_area
    end

    return N_gp, N_avg, area_w
end

function quad4_membrane_cst_resultant(coords::AbstractMatrix,
                                      u_elem::AbstractVector,
                                      Cm::AbstractMatrix,
                                      tri::NTuple{3,Int})
    x1 = coords[tri[1], 1]; y1 = coords[tri[1], 2]
    x2 = coords[tri[2], 1]; y2 = coords[tri[2], 2]
    x3 = coords[tri[3], 1]; y3 = coords[tri[3], 2]
    A2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = 0.5 * abs(A2)
    A <= 1e-12 && return zeros(3)
    b = [y2 - y3, y3 - y1, y1 - y2] ./ (2.0 * A)
    c = [x3 - x2, x1 - x3, x2 - x1] ./ (2.0 * A)
    eps = zeros(3)
    @inbounds for (a, node) in enumerate(tri)
        base = (node - 1) * 6
        ux = u_elem[base + 1]
        uy = u_elem[base + 2]
        eps[1] += b[a] * ux
        eps[2] += c[a] * uy
        eps[3] += c[a] * ux + b[a] * uy
    end
    return Cm * eps
end

function quad4_membrane_cst_resultant_xyu(coords::AbstractMatrix,
                                          uxy::AbstractMatrix,
                                          Cm::AbstractMatrix,
                                          tri::NTuple{3,Int})
    x1 = coords[tri[1], 1]; y1 = coords[tri[1], 2]
    x2 = coords[tri[2], 1]; y2 = coords[tri[2], 2]
    x3 = coords[tri[3], 1]; y3 = coords[tri[3], 2]
    A2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = 0.5 * abs(A2)
    A <= 1e-12 && return zeros(3)
    b = [y2 - y3, y3 - y1, y1 - y2] ./ (2.0 * A)
    c = [x3 - x2, x1 - x3, x2 - x1] ./ (2.0 * A)
    eps = zeros(3)
    @inbounds for (a, node) in enumerate(tri)
        ux = uxy[node, 1]
        uy = uxy[node, 2]
        eps[1] += b[a] * ux
        eps[2] += c[a] * uy
        eps[3] += c[a] * ux + b[a] * uy
    end
    return Cm * eps
end

function quad4_interpolate_corner_resultants_to_gp(N_corner::AbstractMatrix)
    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
    N_gp = zeros(4, 3)
    @inbounds for (gp, rs) in enumerate(gauss_pts)
        r = rs[1]; s = rs[2]
        Nvals = SVector(
            0.25 * (1.0 - r) * (1.0 - s),
            0.25 * (1.0 + r) * (1.0 - s),
            0.25 * (1.0 + r) * (1.0 + s),
            0.25 * (1.0 - r) * (1.0 + s),
        )
        for k in 1:4, comp in 1:3
            N_gp[gp, comp] += Nvals[k] * N_corner[k, comp]
        end
    end
    return N_gp
end

function quad4_gauss_area_weights(coords::AbstractMatrix)
    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
    area_w = zeros(4)
    @inbounds for (gp, rs) in enumerate(gauss_pts)
        dNr, dNs = shape_derivs_quad(rs[1], rs[2])
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        area_w[gp] = max(abs(J11 * J22 - J12 * J21), 1e-12)
    end
    return area_w
end

function quad4_preserve_membrane_average!(N_gp::AbstractMatrix,
                                          target_avg::AbstractVector,
                                          area_w::Union{Nothing,AbstractVector}=nothing)
    current = zeros(3)
    if area_w === nothing
        current .= vec(mean(N_gp; dims=1))
    else
        total_area = max(sum(area_w), 1e-12)
        @inbounds for gp in 1:size(N_gp, 1), comp in 1:3
            current[comp] += area_w[gp] * N_gp[gp, comp] / total_area
        end
    end
    @inbounds for gp in 1:size(N_gp, 1), comp in 1:3
        N_gp[gp, comp] += target_avg[comp] - current[comp]
    end
    return N_gp
end

function quad4_edge_aspect_ratio(coords::AbstractMatrix)
    l12 = hypot(coords[2, 1] - coords[1, 1], coords[2, 2] - coords[1, 2])
    l23 = hypot(coords[3, 1] - coords[2, 1], coords[3, 2] - coords[2, 2])
    l34 = hypot(coords[4, 1] - coords[3, 1], coords[4, 2] - coords[3, 2])
    l41 = hypot(coords[1, 1] - coords[4, 1], coords[1, 2] - coords[4, 2])
    return max(l12, l23, l34, l41) / max(min(l12, l23, l34, l41), 1e-12)
end

function quad4_membrane_force_field_triangle_recovery(
    coords::AbstractMatrix,
    u_elem::AbstractVector,
    Cm::AbstractMatrix,
    target_avg::AbstractVector;
    mode::Symbol=:tri_aspect,
    aspect_switch::Float64=2.0,
)
    N123 = quad4_membrane_cst_resultant(coords, u_elem, Cm, (1, 2, 3))
    N134 = quad4_membrane_cst_resultant(coords, u_elem, Cm, (1, 3, 4))
    N124 = quad4_membrane_cst_resultant(coords, u_elem, Cm, (1, 2, 4))
    N234 = quad4_membrane_cst_resultant(coords, u_elem, Cm, (2, 3, 4))

    eff_mode = mode
    if mode === :tri_aspect
        eff_mode = quad4_edge_aspect_ratio(coords) >= aspect_switch ?
            :tri_incident_interp : :tri_center_adj
    end

    N_gp = zeros(4, 3)
    if eff_mode === :tri_incident_interp
        N_corner = zeros(4, 3)
        N_corner[1, :] .= (N123 .+ N134 .+ N124) ./ 3.0
        N_corner[2, :] .= (N123 .+ N124 .+ N234) ./ 3.0
        N_corner[3, :] .= (N123 .+ N134 .+ N234) ./ 3.0
        N_corner[4, :] .= (N134 .+ N124 .+ N234) ./ 3.0
        N_gp .= quad4_interpolate_corner_resultants_to_gp(N_corner)
    elseif eff_mode === :tri_diagavg
        pt = 1.0 / sqrt(3.0)
        gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
        @inbounds for (gp, rs) in enumerate(gauss_pts)
            r = rs[1]; s = rs[2]
            n13 = r >= s ? N123 : N134
            n24 = r + s <= 0.0 ? N124 : N234
            N_gp[gp, :] .= 0.5 .* (n13 .+ n24)
        end
    else
        coords5 = zeros(5, 2)
        coords5[1:4, :] .= coords
        coords5[5, :] .= vec(mean(coords; dims=1))
        uxy5 = zeros(5, 2)
        @inbounds for k in 1:4
            base = (k - 1) * 6
            uxy5[k, 1] = u_elem[base + 1]
            uxy5[k, 2] = u_elem[base + 2]
        end
        uxy5[5, :] .= vec(mean(uxy5[1:4, :]; dims=1))
        N12c = quad4_membrane_cst_resultant_xyu(coords5, uxy5, Cm, (1, 2, 5))
        N23c = quad4_membrane_cst_resultant_xyu(coords5, uxy5, Cm, (2, 3, 5))
        N34c = quad4_membrane_cst_resultant_xyu(coords5, uxy5, Cm, (3, 4, 5))
        N41c = quad4_membrane_cst_resultant_xyu(coords5, uxy5, Cm, (4, 1, 5))
        N_gp[1, :] .= (N12c .+ N41c) ./ 2.0
        N_gp[2, :] .= (N12c .+ N23c) ./ 2.0
        N_gp[3, :] .= (N23c .+ N34c) ./ 2.0
        N_gp[4, :] .= (N34c .+ N41c) ./ 2.0
    end

    return quad4_preserve_membrane_average!(N_gp, target_avg, quad4_gauss_area_weights(coords))
end

function quad4_membrane_incompatible_condensation_map(coords::AbstractMatrix,
                                                      Cm::AbstractMatrix;
                                                      curvature_membrane=nothing,
                                                      membrane_shear_center_row::Bool=false,
                                                      material_shear_rotation::Float64=0.0,
                                                      membrane_assumed_mode::Symbol=:none,
                                                      membrane_incomp_center_jacobian::Bool=false)
    K_ab = zeros(24, 4)
    K_bb = zeros(4, 4)

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J_c = [dNr_c'; dNs_c'] * coords
    invJ_c = inv(J_c)
    dN_dxy_c = invJ_c * [dNr_c'; dNs_c']
    iJ11c = invJ_c[1,1]; iJ12c = invJ_c[1,2]
    iJ21c = invJ_c[2,1]; iJ22c = invJ_c[2,2]

    @inbounds for gp in gauss_pts
        r, s = gp[1], gp[2]
        dNr_g, dNs_g = shape_derivs_quad(r, s)
        J_g = [dNr_g'; dNs_g'] * coords
        detJ_g = abs(det(J_g))
        if detJ_g < 1e-12
            detJ_g = 1e-12
        end
        iJ = inv(J_g)
        dN_dxy_g = iJ * [dNr_g'; dNs_g']

        Bm_g = zeros(3, 24)
        for k in 1:4
            idx = (k - 1) * 6
            N_k = 0.25 * (1 + (k == 2 || k == 3 ? r : -r)) * (1 + (k >= 3 ? s : -s))
            Bm_g[1, idx + 1] = dN_dxy_g[1, k]
            Bm_g[2, idx + 2] = dN_dxy_g[2, k]
            Bm_g[3, idx + 1] = dN_dxy_g[2, k]
            Bm_g[3, idx + 2] = dN_dxy_g[1, k]
            if curvature_membrane !== nothing
                Bm_g[1, idx + 3] = -N_k * curvature_membrane[1]
                Bm_g[2, idx + 3] = -N_k * curvature_membrane[2]
                Bm_g[3, idx + 3] = -2.0 * N_k * curvature_membrane[3]
            end
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(
                Bm_g,
                dN_dxy_c[1, :],
                dN_dxy_c[2, :],
                curvature_membrane,
                material_shear_rotation,
            )
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
            apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
        end

        Bi = zeros(3, 4)
        fill_quad4_membrane_incompatible_B!(
            Bi,
            r,
            s,
            iJ[1,1],
            iJ[1,2],
            iJ[2,1],
            iJ[2,2],
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            membrane_incomp_center_jacobian,
        )

        K_ab .+= (Bm_g' * Cm * Bi) .* detJ_g
        K_bb .+= (Bi' * Cm * Bi) .* detJ_g
    end

    return -(K_bb \ K_ab')
end

function quad4_membrane_enhanced_condensation_map(coords::AbstractMatrix,
                                                  Cm::AbstractMatrix;
                                                  curvature_membrane=nothing,
                                                  membrane_shear_center_row::Bool=false,
                                                  material_shear_rotation::Float64=0.0,
                                                  membrane_assumed_mode::Symbol=:none,
                                                  membrane_incomp_center_jacobian::Bool=false)
    K_ab = zeros(24, 6)
    K_bb = zeros(6, 6)

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt, -pt), SVector(pt, -pt), SVector(pt, pt), SVector(-pt, pt))
    dNr_c, dNs_c = shape_derivs_quad(0.0, 0.0)
    J_c = [dNr_c'; dNs_c'] * coords
    invJ_c = inv(J_c)
    dN_dxy_c = invJ_c * [dNr_c'; dNs_c']
    iJ11c = invJ_c[1,1]; iJ12c = invJ_c[1,2]
    iJ21c = invJ_c[2,1]; iJ22c = invJ_c[2,2]

    @inbounds for gp in gauss_pts
        r, s = gp[1], gp[2]
        dNr_g, dNs_g = shape_derivs_quad(r, s)
        J_g = [dNr_g'; dNs_g'] * coords
        detJ_g = abs(det(J_g))
        if detJ_g < 1e-12
            detJ_g = 1e-12
        end
        iJ = inv(J_g)
        dN_dxy_g = iJ * [dNr_g'; dNs_g']

        Bm_g = zeros(3, 24)
        for k in 1:4
            idx = (k - 1) * 6
            N_k = 0.25 * (1 + (k == 2 || k == 3 ? r : -r)) * (1 + (k >= 3 ? s : -s))
            Bm_g[1, idx + 1] = dN_dxy_g[1, k]
            Bm_g[2, idx + 2] = dN_dxy_g[2, k]
            Bm_g[3, idx + 1] = dN_dxy_g[2, k]
            Bm_g[3, idx + 2] = dN_dxy_g[1, k]
            if curvature_membrane !== nothing
                Bm_g[1, idx + 3] = -N_k * curvature_membrane[1]
                Bm_g[2, idx + 3] = -N_k * curvature_membrane[2]
                Bm_g[3, idx + 3] = -2.0 * N_k * curvature_membrane[3]
            end
        end
        if membrane_shear_center_row
            project_material_membrane_shear!(
                Bm_g,
                dN_dxy_c[1, :],
                dN_dxy_c[2, :],
                curvature_membrane,
                material_shear_rotation,
            )
        elseif use_membrane_ans_mitc4plus(membrane_assumed_mode, coords, curvature_membrane)
            apply_membrane_ans_mitc4plus!(Bm_g, coords, r, s)
        end

        Bi = zeros(3, 6)
        fill_quad4_membrane_enhanced_B!(
            Bi,
            r,
            s,
            iJ[1,1],
            iJ[1,2],
            iJ[2,1],
            iJ[2,2],
            iJ11c,
            iJ12c,
            iJ21c,
            iJ22c,
            membrane_incomp_center_jacobian,
        )

        K_ab .+= (Bm_g' * Cm * Bi) .* detJ_g
        K_bb .+= (Bi' * Cm * Bi) .* detJ_g
    end

    return -(K_bb \ K_ab')
end

@inline function add_geometric_gradient_block!(Kg::AbstractMatrix,
                                               gdx::AbstractVector,
                                               gdy::AbstractVector,
                                               scale::Float64,
                                               s_xx::Float64,
                                               s_yy::Float64,
                                               s_xy::Float64)
    @inbounds @fastmath for j in eachindex(gdx), i in eachindex(gdx)
        Kg[i, j] += scale * (
            s_xx * gdx[i] * gdx[j] +
            s_yy * gdy[i] * gdy[j] +
            s_xy * (gdx[i] * gdy[j] + gdy[i] * gdx[j])
        )
    end
    return Kg
end

@inline function shell_geometric_metric3(s_xx::Float64, s_yy::Float64, s_xy::Float64,
                                         ax::SVector{3,Float64}, ay::SVector{3,Float64},
                                         bx::SVector{3,Float64}, by::SVector{3,Float64})
    return s_xx * dot(ax, bx) +
           s_yy * dot(ay, by) +
           s_xy * (dot(ax, by) + dot(ay, bx))
end

@inline function principal_stress_2d_components(s_xx::Float64, s_yy::Float64, s_xy::Float64)
    mean_s = 0.5 * (s_xx + s_yy)
    half_d = 0.5 * (s_xx - s_yy)
    radius = sqrt(half_d * half_d + s_xy * s_xy)
    if radius <= 1e-30
        return mean_s, 1.0, 0.0, mean_s, 0.0, 1.0
    end
    theta = 0.5 * atan(2.0 * s_xy, s_xx - s_yy)
    c1 = cos(theta)
    s1 = sin(theta)
    return mean_s + radius, c1, s1, mean_s - radius, -s1, c1
end

@inline function add_geometric_principal_transverse_direction!(
    Kg::AbstractMatrix,
    dux_dx::AbstractVector,
    dux_dy::AbstractVector,
    duy_dx::AbstractVector,
    duy_dy::AbstractVector,
    duz_dx::AbstractVector,
    duz_dy::AbstractVector,
    scale::Float64,
    lambda::Float64,
    c::Float64,
    s::Float64,
    p22_factor::Float64,
    p12_factor::Float64,
    z_factor::Float64,
)
    abs(lambda) <= 1e-30 && return Kg
    p11 = s * s
    p22 = p22_factor * c * c
    p12 = p12_factor * -c * s
    factor = scale * lambda
    @inbounds @fastmath for j in eachindex(dux_dx), i in eachindex(dux_dx)
        gux_i = c * dux_dx[i] + s * dux_dy[i]
        guy_i = c * duy_dx[i] + s * duy_dy[i]
        guz_i = c * duz_dx[i] + s * duz_dy[i]
        gux_j = c * dux_dx[j] + s * dux_dy[j]
        guy_j = c * duy_dx[j] + s * duy_dy[j]
        guz_j = c * duz_dx[j] + s * duz_dy[j]
        Kg[i, j] += factor * (
            p11 * gux_i * gux_j +
            p22 * guy_i * guy_j +
            p12 * (gux_i * guy_j + guy_i * gux_j) +
            z_factor * guz_i * guz_j
        )
    end
    return Kg
end

@inline function add_geometric_principal_transverse_block!(
    Kg::AbstractMatrix,
    dux_dx::AbstractVector,
    dux_dy::AbstractVector,
    duy_dx::AbstractVector,
    duy_dy::AbstractVector,
    duz_dx::AbstractVector,
    duz_dy::AbstractVector,
    scale::Float64,
    s_xx::Float64,
    s_yy::Float64,
    s_xy::Float64,
    shear_yy_factor::Float64=1.0,
    shear_xy_factor::Float64=1.0,
    shear_z_factor::Float64=1.0,
    shear_ratio_min::Float64=1.0,
)
    l1, c1, s1, l2, c2, s2 = principal_stress_2d_components(s_xx, s_yy, s_xy)
    denom = abs(s_xx) + abs(s_yy) + abs(s_xy)
    shear_ratio = denom > 1e-30 ? abs(s_xy) / denom : 0.0
    p22_factor = shear_ratio >= shear_ratio_min ? shear_yy_factor : 1.0
    p12_factor = shear_ratio >= shear_ratio_min ? shear_xy_factor : 1.0
    z_factor = shear_ratio >= shear_ratio_min ? shear_z_factor : 1.0
    add_geometric_principal_transverse_direction!(
        Kg, dux_dx, dux_dy, duy_dx, duy_dy, duz_dx, duz_dy,
        scale, l1, c1, s1, p22_factor, p12_factor, z_factor)
    add_geometric_principal_transverse_direction!(
        Kg, dux_dx, dux_dy, duy_dx, duy_dy, duz_dx, duz_dy,
        scale, l2, c2, s2, p22_factor, p12_factor, z_factor)
    return Kg
end

@inline function add_geometric_principal_transverse_pair_direction!(
    Kg::AbstractMatrix,
    row0::Int,
    col0::Int,
    dNi_dx::Float64,
    dNi_dy::Float64,
    dNj_dx::Float64,
    dNj_dy::Float64,
    scale::Float64,
    lambda::Float64,
    c::Float64,
    s::Float64,
    p22_factor::Float64,
    p12_factor::Float64,
    z_factor::Float64,
)
    abs(lambda) <= 1e-30 && return Kg
    gi = c * dNi_dx + s * dNi_dy
    gj = c * dNj_dx + s * dNj_dy
    val = scale * lambda * gi * gj
    p11 = s * s
    p22 = p22_factor * c * c
    p12 = p12_factor * -c * s
    Kg[row0 + 1, col0 + 1] += val * p11
    Kg[row0 + 1, col0 + 2] += val * p12
    Kg[row0 + 2, col0 + 1] += val * p12
    Kg[row0 + 2, col0 + 2] += val * p22
    Kg[row0 + 3, col0 + 3] += z_factor * val
    return Kg
end

@inline function add_geometric_principal_transverse_pair!(
    Kg::AbstractMatrix,
    row0::Int,
    col0::Int,
    dNi_dx::Float64,
    dNi_dy::Float64,
    dNj_dx::Float64,
    dNj_dy::Float64,
    scale::Float64,
    s_xx::Float64,
    s_yy::Float64,
    s_xy::Float64,
    shear_yy_factor::Float64=1.0,
    shear_xy_factor::Float64=1.0,
    shear_z_factor::Float64=1.0,
    shear_ratio_min::Float64=1.0,
)
    l1, c1, s1, l2, c2, s2 = principal_stress_2d_components(s_xx, s_yy, s_xy)
    denom = abs(s_xx) + abs(s_yy) + abs(s_xy)
    shear_ratio = denom > 1e-30 ? abs(s_xy) / denom : 0.0
    p22_factor = shear_ratio >= shear_ratio_min ? shear_yy_factor : 1.0
    p12_factor = shear_ratio >= shear_ratio_min ? shear_xy_factor : 1.0
    z_factor = shear_ratio >= shear_ratio_min ? shear_z_factor : 1.0
    add_geometric_principal_transverse_pair_direction!(
        Kg, row0, col0, dNi_dx, dNi_dy, dNj_dx, dNj_dy,
        scale, l1, c1, s1, p22_factor, p12_factor, z_factor)
    add_geometric_principal_transverse_pair_direction!(
        Kg, row0, col0, dNi_dx, dNi_dy, dNj_dx, dNj_dy,
        scale, l2, c2, s2, p22_factor, p12_factor, z_factor)
    return Kg
end

function quad4_membrane_force_field_covariant(coords3d::AbstractMatrix,
                                              u_nodes_global::AbstractMatrix,
                                              basis1::SVector{3,Float64},
                                              basis2::SVector{3,Float64},
                                              Cm::AbstractMatrix)
    pt = 1.0 / sqrt(3.0)
    gauss_pts = [-pt -pt; pt -pt; pt pt; -pt pt]

    N_gp = zeros(4, 3)
    N_avg = zeros(3)
    area_w = zeros(4)
    total_area = 0.0

    @inbounds for i in 1:4
        r, s = gauss_pts[i,1], gauss_pts[i,2]
        dNr, dNs = shape_derivs_quad(r, s)

        a_r = SVector(
            dNr[1]*coords3d[1,1] + dNr[2]*coords3d[2,1] + dNr[3]*coords3d[3,1] + dNr[4]*coords3d[4,1],
            dNr[1]*coords3d[1,2] + dNr[2]*coords3d[2,2] + dNr[3]*coords3d[3,2] + dNr[4]*coords3d[4,2],
            dNr[1]*coords3d[1,3] + dNr[2]*coords3d[2,3] + dNr[3]*coords3d[3,3] + dNr[4]*coords3d[4,3],
        )
        a_s = SVector(
            dNs[1]*coords3d[1,1] + dNs[2]*coords3d[2,1] + dNs[3]*coords3d[3,1] + dNs[4]*coords3d[4,1],
            dNs[1]*coords3d[1,2] + dNs[2]*coords3d[2,2] + dNs[3]*coords3d[3,2] + dNs[4]*coords3d[4,2],
            dNs[1]*coords3d[1,3] + dNs[2]*coords3d[2,3] + dNs[3]*coords3d[3,3] + dNs[4]*coords3d[4,3],
        )

        g11 = dot(a_r, a_r)
        g12 = dot(a_r, a_s)
        g22 = dot(a_s, a_s)
        detg = g11 * g22 - g12 * g12
        if abs(detg) < 1e-14
            detg = detg < 0.0 ? -1e-14 : 1e-14
        end
        invg11 = g22 / detg
        invg12 = -g12 / detg
        invg22 = g11 / detg

        a_r_contra = invg11 * a_r + invg12 * a_s
        a_s_contra = invg12 * a_r + invg22 * a_s

        u_r = SVector(
            dNr[1]*u_nodes_global[1,1] + dNr[2]*u_nodes_global[2,1] + dNr[3]*u_nodes_global[3,1] + dNr[4]*u_nodes_global[4,1],
            dNr[1]*u_nodes_global[1,2] + dNr[2]*u_nodes_global[2,2] + dNr[3]*u_nodes_global[3,2] + dNr[4]*u_nodes_global[4,2],
            dNr[1]*u_nodes_global[1,3] + dNr[2]*u_nodes_global[2,3] + dNr[3]*u_nodes_global[3,3] + dNr[4]*u_nodes_global[4,3],
        )
        u_s = SVector(
            dNs[1]*u_nodes_global[1,1] + dNs[2]*u_nodes_global[2,1] + dNs[3]*u_nodes_global[3,1] + dNs[4]*u_nodes_global[4,1],
            dNs[1]*u_nodes_global[1,2] + dNs[2]*u_nodes_global[2,2] + dNs[3]*u_nodes_global[3,2] + dNs[4]*u_nodes_global[4,2],
            dNs[1]*u_nodes_global[1,3] + dNs[2]*u_nodes_global[2,3] + dNs[3]*u_nodes_global[3,3] + dNs[4]*u_nodes_global[4,3],
        )

        grad_u = u_r * a_r_contra' + u_s * a_s_contra'
        eps11 = dot(basis1, grad_u * basis1)
        eps22 = dot(basis2, grad_u * basis2)
        gam12 = dot(basis1, grad_u * basis2) + dot(basis2, grad_u * basis1)
        N_vec = Cm * SVector(eps11, eps22, gam12)

        N_gp[i, 1] = N_vec[1]
        N_gp[i, 2] = N_vec[2]
        N_gp[i, 3] = N_vec[3]

        dA = norm(cross(a_r, a_s))
        N_avg .+= N_vec .* dA
        area_w[i] = dA
        total_area += dA
    end

    if total_area > 0.0
        N_avg ./= total_area
    end

    return N_gp, N_avg, area_w
end

function stiffness_tria3_generic(coords, E, nu, h; bend_ratio=1.0, ts_t=5.0/6.0, k6rot=100.0)
    T = promote_type(eltype(coords), typeof(E), typeof(nu), typeof(h))
    oneT = one(T)
    zeroT = zero(T)
    G = E / (2*(1+nu))
    Dbase = T[oneT nu zeroT; nu oneT zeroT; zeroT zeroT (oneT - nu) / 2]
    Dm = (E * h / (oneT - nu^2)) .* Dbase
    Db = bend_ratio * (E * h^3 / (12 * (oneT - nu^2))) .* Dbase
    Ds = ts_t * G * h .* T[oneT zeroT; zeroT oneT]
    return stiffness_tria3_matrices_generic(coords, Dm, Db, Ds, h, G; bend_ratio=bend_ratio, k6rot=k6rot)
end

function stiffness_tria3(coords, E, nu, h; bend_ratio=1.0, ts_t=5.0/6.0, k6rot=100.0)
    return stiffness_tria3_generic(coords, E, nu, h; bend_ratio=bend_ratio, ts_t=ts_t, k6rot=k6rot)
end

const TRIA3_MACRO_QUADS = ((1, 4, 7, 6), (2, 5, 7, 4), (3, 6, 7, 5))
const QUAD4_PLATE_DOF_IDX = (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23)

@inline function _tria3_virtual_quad_points(coords::AbstractMatrix)
    T = eltype(coords)
    pts = Matrix{T}(undef, 7, 2)

    x1 = coords[1,1]; y1 = coords[1,2]
    x2 = coords[2,1]; y2 = coords[2,2]
    x3 = coords[3,1]; y3 = coords[3,2]

    pts[1,1] = x1; pts[1,2] = y1
    pts[2,1] = x2; pts[2,2] = y2
    pts[3,1] = x3; pts[3,2] = y3
    pts[4,1] = (x1 + x2) / 2; pts[4,2] = (y1 + y2) / 2
    pts[5,1] = (x2 + x3) / 2; pts[5,2] = (y2 + y3) / 2
    pts[6,1] = (x3 + x1) / 2; pts[6,2] = (y3 + y1) / 2
    pts[7,1] = (x1 + x2 + x3) / 3; pts[7,2] = (y1 + y2 + y3) / 3

    return pts
end

@inline function _tria3_virtual_quad_area(qc::AbstractMatrix)
    a1 = (qc[2,1] - qc[1,1]) * (qc[4,2] - qc[1,2]) - (qc[4,1] - qc[1,1]) * (qc[2,2] - qc[1,2])
    a2 = (qc[3,1] - qc[2,1]) * (qc[4,2] - qc[2,2]) - (qc[4,1] - qc[2,1]) * (qc[3,2] - qc[2,2])
    return (abs(a1) + abs(a2)) / 2
end

function tria3_plate_macro_data(coords, Cm, Cb, Cs, h, E_ref, pressure=nothing; bend_ratio=1.0, k6rot=100.0)
    T = promote_type(eltype(coords), eltype(Cm), eltype(Cb), eltype(Cs), typeof(h), typeof(E_ref))
    pts = _tria3_virtual_quad_points(coords)
    zero_cond = zeros(T, 9, 9)
    zero_map = zeros(T, 12, 9)
    zero_load = pressure === nothing ? nothing : zeros(T, 9)

    bend_ratio <= T(1e-12) && return (Kcond=zero_cond, Aint=zero_map, pts=pts, fcond=zero_load)

    K = zeros(T, 21, 21)
    f = pressure === nothing ? nothing : zeros(T, 21)
    plate_idx = collect(QUAD4_PLATE_DOF_IDX)

    for quad in TRIA3_MACRO_QUADS
        qc = pts[[quad[1], quad[2], quad[3], quad[4]], :]
        Ke_full = stiffness_quad4_matrices(qc, Cm, Cb, Cs, h, E_ref; bend_ratio=bend_ratio, k6rot=k6rot)
        Ke_plate = Ke_full[plate_idx, plate_idx]

        edofs = Int[]
        for nid in quad
            append!(edofs, (3*(nid-1)+1):(3*(nid-1)+3))
        end
        K[edofs, edofs] .+= Ke_plate

        if f !== nothing
            fe = zeros(T, 12)
            qA = pressure * _tria3_virtual_quad_area(qc)
            fe[1] = qA / 4
            fe[4] = qA / 4
            fe[7] = qA / 4
            fe[10] = qA / 4
            f[edofs] .+= fe
        end
    end

    ext = 1:9
    int = 10:21
    Kee = K[ext, ext]
    Kei = K[ext, int]
    Kie = K[int, ext]
    Kii = K[int, int]

    Fii = lu(Kii)
    Aint = -(Fii \ Kie)
    Kcond = Kee + Kei * Aint

    fcond = nothing
    if f !== nothing
        fcond = f[ext] - Kei * (Fii \ f[int])
    end

    return (Kcond=Kcond, Aint=Aint, pts=pts, fcond=fcond)
end

function tria3_plate_macro_pressure_load(coords, E, nu, h, pressure; bend_ratio=1.0, k6rot=100.0)
    T = promote_type(eltype(coords), typeof(E), typeof(nu), typeof(h), typeof(pressure))
    D = (T(E) / (one(T) - T(nu)^2)) .* Matrix{T}([one(T) T(nu) zero(T); T(nu) one(T) zero(T); zero(T) zero(T) (one(T)-T(nu))/T(2)])
    Cm = D * T(h)
    Cb = D * (T(h)^3 / T(12))
    G = T(E) / (T(2) * (one(T) + T(nu)))
    Cs = zeros(T, 2, 2)
    shear_scale = T(5) / T(6) * G * T(h)
    Cs[1,1] = shear_scale
    Cs[2,2] = shear_scale
    macro_data = tria3_plate_macro_data(coords, Cm, Cb, Cs, T(h), G, T(pressure); bend_ratio=bend_ratio, k6rot=k6rot)
    return macro_data.fcond === nothing ? zeros(T, 9) : macro_data.fcond
end

function tria3_plate_macro_shear_resultant(coords, u_plate, E, nu, h; bend_ratio=1.0, k6rot=100.0)
    T = promote_type(eltype(coords), eltype(u_plate), typeof(E), typeof(nu), typeof(h))
    bend_ratio <= T(1e-12) && return zeros(T, 2)

    D = (T(E) / (one(T) - T(nu)^2)) .* Matrix{T}([one(T) T(nu) zero(T); T(nu) one(T) zero(T); zero(T) zero(T) (one(T)-T(nu))/T(2)])
    Cm = D * T(h)
    Cb = D * (T(h)^3 / T(12))
    G = T(E) / (T(2) * (one(T) + T(nu)))
    Cs = zeros(T, 2, 2)
    shear_scale = T(5) / T(6) * G * T(h)
    Cs[1,1] = shear_scale
    Cs[2,2] = shear_scale
    macro_data = tria3_plate_macro_data(coords, Cm, Cb, Cs, T(h), G; bend_ratio=bend_ratio, k6rot=k6rot)

    u_all = Vector{T}(undef, 21)
    u_all[1:9] = u_plate
    u_all[10:21] = macro_data.Aint * u_plate

    Q_sum = zeros(T, 2)
    A_sum = zero(T)
    u_quad = zeros(T, 24)
    plate_idx = collect(QUAD4_PLATE_DOF_IDX)

    for quad in TRIA3_MACRO_QUADS
        fill!(u_quad, zero(T))
        qc = macro_data.pts[[quad[1], quad[2], quad[3], quad[4]], :]
        edofs = Int[]
        for nid in quad
            append!(edofs, (3*(nid-1)+1):(3*(nid-1)+3))
        end
        u_quad[plate_idx] .= u_all[edofs]

        _, _, Q_quad, _, _, _, _ = stress_strain_quad4(qc, u_quad, E, nu, h, h; bend_ratio=bend_ratio)
        area = _tria3_virtual_quad_area(qc)
        Q_sum .+= area .* Q_quad
        A_sum += area
    end

    A_sum <= T(1e-12) && return zeros(T, 2)
    return Q_sum ./ A_sum
end

function tria3_plate_macro_average_moment(coords, u_elem, E, nu, h; bend_ratio=1.0, k6rot=100.0)
    T = promote_type(eltype(coords), eltype(u_elem), typeof(E), typeof(nu), typeof(h))
    bend_ratio <= T(1e-12) && return zeros(T, 3)

    D = (T(E) / (one(T) - T(nu)^2)) .* Matrix{T}([one(T) T(nu) zero(T); T(nu) one(T) zero(T); zero(T) zero(T) (one(T)-T(nu))/T(2)])
    Cm = D * T(h)
    Cb = D * (T(h)^3 / T(12))
    G = T(E) / (T(2) * (one(T) + T(nu)))
    Cs = zeros(T, 2, 2)
    shear_scale = T(5) / T(6) * G * T(h)
    Cs[1,1] = shear_scale
    Cs[2,2] = shear_scale
    macro_data = tria3_plate_macro_data(coords, Cm, Cb, Cs, T(h), G; bend_ratio=bend_ratio, k6rot=k6rot)

    u_plate = T[
        u_elem[3], u_elem[4], u_elem[5],
        u_elem[9], u_elem[10], u_elem[11],
        u_elem[15], u_elem[16], u_elem[17],
    ]
    u_all = Vector{T}(undef, 21)
    u_all[1:9] = u_plate
    u_all[10:21] = macro_data.Aint * u_plate

    M_sum = zeros(T, 3)
    A_sum = zero(T)
    u_quad = zeros(T, 24)
    plate_idx = collect(QUAD4_PLATE_DOF_IDX)

    for quad in TRIA3_MACRO_QUADS
        fill!(u_quad, zero(T))
        qc = macro_data.pts[[quad[1], quad[2], quad[3], quad[4]], :]
        edofs = Int[]
        for nid in quad
            append!(edofs, (3*(nid-1)+1):(3*(nid-1)+3))
        end
        u_quad[plate_idx] .= u_all[edofs]

        _, M_quad, _, _, _, _, _ = stress_strain_quad4(qc, u_quad, E, nu, h, h; bend_ratio=bend_ratio)
        area = _tria3_virtual_quad_area(qc)
        M_sum .+= area .* M_quad
        A_sum += area
    end

    A_sum <= T(1e-12) && return zeros(T, 3)
    return M_sum ./ A_sum
end

# Overload accepting pre-computed constitutive matrices (for orthotropic MAT8)
function stiffness_tria3_matrices_generic(coords, Dm, Db, Ds, h, G_ref; bend_ratio=1.0, k6rot=100.0, Bmb=nothing)
    T = promote_type(eltype(coords), eltype(Dm), eltype(Db), eltype(Ds), typeof(h), typeof(G_ref))
    x, y = coords[:,1], coords[:,2]
    A2 = x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])
    A = T(0.5) * abs(A2)
    if A < T(1e-12); return zeros(T, 18, 18); end

    Ke = zeros(T, 18, 18)

    # --- Membrane (constant strain triangle, CST) ---
    bv = T[y[2]-y[3], y[3]-y[1], y[1]-y[2]] ./ (2*A)
    cv = T[x[3]-x[2], x[1]-x[3], x[2]-x[1]] ./ (2*A)
    Bm = zeros(T, 3, 6)
    for i in 1:3; Bm[1, i*2-1]=bv[i]; Bm[2, i*2]=cv[i]; Bm[3, i*2-1]=cv[i]; Bm[3, i*2]=bv[i]; end
    K_mem = Bm' * Dm * Bm * A
    m_idx = [1,2, 7,8, 13,14]
    Ke[m_idx, m_idx] = K_mem

    # --- Bending: Isoparametric Mindlin-Reissner Triangle ---
    Bb = zeros(T, 3, 9)
    for i in 1:3
        col_rx = 3*(i-1) + 2  # θx
        col_ry = 3*(i-1) + 3  # θy
        Bb[1, col_ry] = bv[i]     # κxx
        Bb[2, col_rx] = -cv[i]    # κyy
        Bb[3, col_rx] = -bv[i]    # κxy: -∂θx/∂x
        Bb[3, col_ry] = cv[i]     # κxy: +∂θy/∂y
    end
    Kb = Bb' * Db * Bb * A

    # Transverse shear at centroid (1-point integration)
    # γxz = ∂w/∂x + θy, γyz = ∂w/∂y - θx
    Bs = zeros(T, 2, 9)
    for i in 1:3
        col_w  = 3*(i-1) + 1
        col_rx = 3*(i-1) + 2
        col_ry = 3*(i-1) + 3
        Bs[1, col_w]  = bv[i]
        Bs[1, col_ry] = one(T)/T(3)    # N[i] = 1/3 at centroid
        Bs[2, col_w]  = cv[i]
        Bs[2, col_rx] = -one(T)/T(3)
    end
    Ks = Bs' * Ds * Bs * A

    # TRIA3 uses 1-point centroidal shear integration (locking-free for constant shear).
    # phi2=1.0 gives the correct Reissner-Mindlin stiffness, consistent with Nastran CTRIA3.
    b_idx = [3,4,5, 9,10,11, 15,16,17]
    macro_data = tria3_plate_macro_data(coords, Dm, Db, Ds, h, G_ref; bend_ratio=bend_ratio, k6rot=k6rot)
    Ke[b_idx, b_idx] += macro_data.Kcond

    # B matrix coupling (membrane-bending): cross-blocks between m_idx and b_idx
    if Bmb !== nothing
        K_mb = Bm' * Bmb * Bb * A  # 6x9 coupling block
        Ke[m_idx, b_idx] += K_mb
        Ke[b_idx, m_idx] += K_mb'
    end

    # --- Hughes-Brezzi drilling rotation coupling ---
    # ε_drill = θz - (1/2)(∂v/∂x - ∂u/∂y), penalized with alpha_drill * ε_drill²
    alpha_drill = (k6rot / 1e5) * G_ref * h
    Bd = zeros(T, 1, 18)
    for i in 1:3
        idx = (i-1)*6
        Bd[1, idx+1] = T(0.5) * cv[i]    # +(1/2)*∂N/∂y (from ∂u/∂y)
        Bd[1, idx+2] = -T(0.5) * bv[i]   # -(1/2)*∂N/∂x (from -∂v/∂x)
        Bd[1, idx+6] = one(T)/T(3)       # N_i = 1/3 at centroid
    end
    Ke .+= alpha_drill .* (Bd' * Bd) .* A

    return Ke
end

function stiffness_tria3_matrices(coords, Dm, Db, Ds, h, G_ref; bend_ratio=1.0, k6rot=100.0, Bmb=nothing)
    return stiffness_tria3_matrices_generic(coords, Dm, Db, Ds, h, G_ref; bend_ratio=bend_ratio, k6rot=k6rot, Bmb=Bmb)
end

function stress_strain_tria3(coords, u_elem, E, nu, h; bend_ratio=1.0, Cm_override=nothing)
    x, y = coords[:,1], coords[:,2]
    A = 0.5 * abs(x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2]))
    if A < 1e-12; return zeros(3), zeros(3), zeros(2), zeros(3), zeros(3), zeros(3), zeros(3); end

    b = [y[2]-y[3], y[3]-y[1], y[1]-y[2]] ./ (2*A)
    c = [x[3]-x[2], x[1]-x[3], x[2]-x[1]] ./ (2*A)

    # Membrane strain
    Bm = zeros(3, 6)
    for i in 1:3; Bm[1, i*2-1]=b[i]; Bm[2, i*2]=c[i]; Bm[3, i*2-1]=c[i]; Bm[3, i*2]=b[i]; end
    D = (E / (1 - nu^2)) .* [1 nu 0; nu 1 0; 0 0 (1-nu)/2]
    u_mem = [u_elem[1], u_elem[2], u_elem[7], u_elem[8], u_elem[13], u_elem[14]]
    eps_mem = Bm * u_mem

    # Bending curvature
    Bb = zeros(3, 6)
    for i in 1:3
        Bb[1, i*2]   = b[i]    # dθy/dx
        Bb[2, i*2-1] = -c[i]   # -dθx/dy
        Bb[3, i*2]   = c[i]    # dθy/dy
        Bb[3, i*2-1] = -b[i]   # -dθx/dx
    end
    u_rot = [u_elem[4], u_elem[5], u_elem[10], u_elem[11], u_elem[16], u_elem[17]]
    kappa = Bb * u_rot

    # Membrane forces and bending moments
    N = Cm_override !== nothing ? Cm_override * eps_mem : (D * eps_mem) * h
    M = -bend_ratio * (D * kappa) * (h^3/12.0)

    u_plate = [u_elem[3], u_elem[4], u_elem[5], u_elem[9], u_elem[10], u_elem[11], u_elem[15], u_elem[16], u_elem[17]]
    Q = tria3_plate_macro_shear_resultant(coords, u_plate, E, nu, h; bend_ratio=bend_ratio)

    # Stresses at top/bottom surfaces
    z1 = -h/2.0; z2 = h/2.0
    strain_z1 = eps_mem .+ z1 .* kappa
    stress_z1 = D * strain_z1
    strain_z2 = eps_mem .+ z2 .* kappa
    stress_z2 = D * strain_z2

    return N, M, Q, stress_z1, stress_z2, strain_z1, strain_z2
end

# =============================================================================
# GEOMETRIC (DIFFERENTIAL) STIFFNESS MATRICES FOR SOL105 LINEAR BUCKLING
# =============================================================================

# Consistent geometric stiffness for beam element (Przemieniecki)
# P = axial force from SOL101 (positive = tension, negative = compression)
# Local DOFs: [u1,v1,w1,θx1,θy1,θz1, u2,v2,w2,θx2,θy2,θz2]
function geometric_stiffness_frame3d(L::Float64, P::Float64)
    kg = zeros(12, 12)
    if L < 1e-9 || abs(P) < 1e-30; return kg; end

    c1 = 6.0 * P / (5.0 * L)
    c2 = P / 10.0
    c3 = 2.0 * P * L / 15.0
    c4 = -P * L / 30.0

    # Lateral y-direction (DOFs 2,6,8,12)
    kg[2,2] = c1;   kg[8,8] = c1
    kg[2,8] = -c1;  kg[8,2] = -c1
    kg[2,6] = c2;   kg[6,2] = c2
    kg[2,12] = c2;  kg[12,2] = c2
    kg[6,6] = c3;   kg[12,12] = c3
    kg[6,12] = c4;  kg[12,6] = c4
    kg[8,6] = -c2;  kg[6,8] = -c2
    kg[8,12] = -c2; kg[12,8] = -c2

    # Lateral z-direction (DOFs 3,5,9,11)
    kg[3,3] = c1;   kg[9,9] = c1
    kg[3,9] = -c1;  kg[9,3] = -c1
    kg[3,5] = -c2;  kg[5,3] = -c2
    kg[3,11] = -c2; kg[11,3] = -c2
    kg[5,5] = c3;   kg[11,11] = c3
    kg[5,11] = c4;  kg[11,5] = c4
    kg[9,5] = c2;   kg[5,9] = c2
    kg[9,11] = c2;  kg[11,9] = c2

    return kg
end

# Geometric stiffness for rod/truss element.
# CROD/CONROD have axial and torsional stiffness only, so the initial-stress
# operator must act on the transverse translations only. Reusing the beam-column
# geometric stiffness adds bending-rotation terms the rod element does not have.
function geometric_stiffness_rod(L::Float64, P::Float64)
    kg = zeros(12, 12)
    abs(L) < 1e-30 && return kg

    c = P / L

    # Local transverse y-direction (DOFs 2,8)
    kg[2,2] = c
    kg[2,8] = -c
    kg[8,2] = -c
    kg[8,8] = c

    # Local transverse z-direction (DOFs 3,9)
    kg[3,3] = c
    kg[3,9] = -c
    kg[9,3] = -c
    kg[9,9] = c

    return kg
end

# Geometric stiffness for CQUAD4 shell element (24×24)
# Uses membrane stress state [σxx, σyy, σxy] from SOL101.
# coords = 4×2 local coordinates (same as stiffness computation).
function geometric_stiffness_quad4(coords::AbstractMatrix, sigma_mem::AbstractVector, h::Float64;
                                   trans_mode::Symbol=:all,
                                   curvature::Union{Nothing,SVector{3,Float64}}=nothing,
                                   curvature_sign::Float64=1.0,
                                   rot_grad_scale::Float64=0.0,
                                   membrane_shear_center_row::Bool=false,
                                   Cm::Union{Nothing,AbstractMatrix}=nothing,
                                    membrane_incomp::Bool=false,
                                    membrane_enhanced::Bool=false,
                                    material_shear_rotation::Float64=0.0,
                                    membrane_assumed_mode::Symbol=:none,
                                    membrane_incomp_center_jacobian::Bool=false,
                                    principal_shear_yy_factor::Float64=1.0,
                                    principal_shear_xy_factor::Float64=1.0,
                                    principal_shear_z_factor::Float64=1.0,
                                    principal_shear_ratio_min::Float64=1.0)
    sigma_gp = zeros(4, 3)
    @inbounds for gp in 1:4
        sigma_gp[gp, 1] = sigma_mem[1]
        sigma_gp[gp, 2] = sigma_mem[2]
        sigma_gp[gp, 3] = sigma_mem[3]
    end
    return geometric_stiffness_quad4(coords, sigma_gp, h;
                                     trans_mode=trans_mode,
                                     curvature=curvature,
                                     curvature_sign=curvature_sign,
                                     rot_grad_scale=rot_grad_scale,
                                     membrane_shear_center_row=membrane_shear_center_row,
                                     Cm=Cm,
                                     membrane_incomp=membrane_incomp,
                                     membrane_enhanced=membrane_enhanced,
                                     material_shear_rotation=material_shear_rotation,
                                     membrane_assumed_mode=membrane_assumed_mode,
                                     membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
                                     principal_shear_yy_factor=principal_shear_yy_factor,
                                     principal_shear_xy_factor=principal_shear_xy_factor,
                                     principal_shear_z_factor=principal_shear_z_factor,
                                     principal_shear_ratio_min=principal_shear_ratio_min)
end

function geometric_stiffness_quad4(coords::AbstractMatrix, sigma_mem_gp::AbstractMatrix, h::Float64;
                                    trans_mode::Symbol=:all,
                                    curvature::Union{Nothing,SVector{3,Float64}}=nothing,
                                    curvature_sign::Float64=1.0,
                                    rot_grad_scale::Float64=0.0,
                                    membrane_shear_center_row::Bool=false,
                                    Cm::Union{Nothing,AbstractMatrix}=nothing,
                                    membrane_incomp::Bool=false,
                                    membrane_enhanced::Bool=false,
                                    material_shear_rotation::Float64=0.0,
                                    membrane_assumed_mode::Symbol=:none,
                                    membrane_incomp_center_jacobian::Bool=false,
                                    principal_shear_yy_factor::Float64=1.0,
                                    principal_shear_xy_factor::Float64=1.0,
                                    principal_shear_z_factor::Float64=1.0,
                                    principal_shear_ratio_min::Float64=1.0)
    Kg = zeros(24, 24)
    if h < 1e-30; return Kg; end

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt,-pt), SVector(pt,-pt), SVector(pt,pt), SVector(-pt,pt))
    dNr_c = SVector(-0.25, 0.25, 0.25, -0.25)
    dNs_c = SVector(-0.25, -0.25, 0.25, 0.25)
    J11_c = dNr_c[1]*coords[1,1] + dNr_c[2]*coords[2,1] + dNr_c[3]*coords[3,1] + dNr_c[4]*coords[4,1]
    J12_c = dNr_c[1]*coords[1,2] + dNr_c[2]*coords[2,2] + dNr_c[3]*coords[3,2] + dNr_c[4]*coords[4,2]
    J21_c = dNs_c[1]*coords[1,1] + dNs_c[2]*coords[2,1] + dNs_c[3]*coords[3,1] + dNs_c[4]*coords[4,1]
    J22_c = dNs_c[1]*coords[1,2] + dNs_c[2]*coords[2,2] + dNs_c[3]*coords[3,2] + dNs_c[4]*coords[4,2]
    detJ_c = J11_c*J22_c - J12_c*J21_c
    abs(detJ_c) < 1e-12 && (detJ_c = detJ_c < 0.0 ? -1e-12 : 1e-12)
    inv_det_c = 1.0 / detJ_c
    iJ11_c = J22_c*inv_det_c; iJ12_c = -J12_c*inv_det_c
    iJ21_c = -J21_c*inv_det_c; iJ22_c = J11_c*inv_det_c
    dNdx_c = zeros(4)
    dNdy_c = zeros(4)
    if membrane_shear_center_row
        @inbounds for i in 1:4
            dNdx_c[i] = iJ11_c*dNr_c[i] + iJ12_c*dNs_c[i]
            dNdy_c[i] = iJ21_c*dNr_c[i] + iJ22_c*dNs_c[i]
        end
    end
    membrane_A =
        if membrane_enhanced && Cm !== nothing
            quad4_membrane_enhanced_condensation_map(
                coords, Cm;
                curvature_membrane=(trans_mode === :curvature && curvature !== nothing ? curvature_sign * curvature : nothing),
                membrane_shear_center_row=membrane_shear_center_row,
                material_shear_rotation=material_shear_rotation,
                membrane_assumed_mode=membrane_assumed_mode,
                membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
            )
        elseif membrane_incomp && Cm !== nothing
            quad4_membrane_incompatible_condensation_map(
                coords, Cm;
                curvature_membrane=(trans_mode === :curvature && curvature !== nothing ? curvature_sign * curvature : nothing),
                membrane_shear_center_row=membrane_shear_center_row,
                material_shear_rotation=material_shear_rotation,
                membrane_assumed_mode=membrane_assumed_mode,
                membrane_incomp_center_jacobian=membrane_incomp_center_jacobian,
            )
        else
            nothing
        end

    @inbounds @fastmath for gp in 1:4
        s_xx = sigma_mem_gp[gp, 1]
        s_yy = sigma_mem_gp[gp, 2]
        s_xy = sigma_mem_gp[gp, 3]
        r, s = gauss_pts[gp][1], gauss_pts[gp][2]
        dNr, dNs = shape_derivs_quad(r, s)
        Nvals = SVector(
            0.25 * (1-r) * (1-s),
            0.25 * (1+r) * (1-s),
            0.25 * (1+r) * (1+s),
            0.25 * (1-r) * (1+s),
        )

        # Jacobian
        J11 = dNr[1]*coords[1,1] + dNr[2]*coords[2,1] + dNr[3]*coords[3,1] + dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2] + dNr[2]*coords[2,2] + dNr[3]*coords[3,2] + dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1] + dNs[2]*coords[2,1] + dNs[3]*coords[3,1] + dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2] + dNs[2]*coords[2,2] + dNs[3]*coords[3,2] + dNs[4]*coords[4,2]
        detJ = J11*J22 - J12*J21
        abs_detJ = abs(detJ)
        if abs_detJ < 1e-12; abs_detJ = 1e-12; end
        inv_det = 1.0 / detJ
        iJ11 = J22*inv_det; iJ12 = -J12*inv_det
        iJ21 = -J21*inv_det; iJ22 = J11*inv_det

        if membrane_A !== nothing
            dux_dx = zeros(24); dux_dy = zeros(24)
            duy_dx = zeros(24); duy_dy = zeros(24)
            duz_dx = zeros(24); duz_dy = zeros(24)
            ux_val = zeros(24); uy_val = zeros(24); uz_val = zeros(24)
            for i in 1:4
                dNi_dx = iJ11*dNr[i] + iJ12*dNs[i]
                dNi_dy = iJ21*dNr[i] + iJ22*dNs[i]
                row0 = (i - 1) * 6
                dux_dx[row0 + 1] = dNi_dx
                dux_dy[row0 + 1] = dNi_dy
                duy_dx[row0 + 2] = dNi_dx
                duy_dy[row0 + 2] = dNi_dy
                duz_dx[row0 + 3] = dNi_dx
                duz_dy[row0 + 3] = dNi_dy
                ux_val[row0 + 1] = Nvals[i]
                uy_val[row0 + 2] = Nvals[i]
                uz_val[row0 + 3] = Nvals[i]
            end

            phi1 = 1.0 - r * r
            phi2 = 1.0 - s * s
            psi = r * s
            miJ11, miJ12, miJ21, miJ22 = quad4_membrane_incompatible_jacobian_components(
                membrane_incomp_center_jacobian,
                iJ11, iJ12, iJ21, iJ22,
                iJ11_c, iJ12_c, iJ21_c, iJ22_c,
            )
            dphi1_dx = miJ11 * (-2.0 * r)
            dphi1_dy = miJ21 * (-2.0 * r)
            dphi2_dx = miJ12 * (-2.0 * s)
            dphi2_dy = miJ22 * (-2.0 * s)
            dpsi_dx = miJ11 * s + miJ12 * r
            dpsi_dy = miJ21 * s + miJ22 * r
            if size(membrane_A, 1) == 6
                for a in 1:24
                    a1 = membrane_A[1, a]
                    a2 = membrane_A[2, a]
                    a3 = membrane_A[3, a]
                    a4 = membrane_A[4, a]
                    a5 = membrane_A[5, a]
                    a6 = membrane_A[6, a]
                    dux_dx[a] += dphi1_dx * a1 + dphi2_dx * a3 + dpsi_dx * a5
                    dux_dy[a] += dphi1_dy * a1 + dphi2_dy * a3 + dpsi_dy * a5
                    duy_dx[a] += dphi1_dx * a2 + dphi2_dx * a4 + dpsi_dx * a6
                    duy_dy[a] += dphi1_dy * a2 + dphi2_dy * a4 + dpsi_dy * a6
                    ux_val[a] += phi1 * a1 + phi2 * a3 + psi * a5
                    uy_val[a] += phi1 * a2 + phi2 * a4 + psi * a6
                end
            else
                for a in 1:24
                    a1 = membrane_A[1, a]
                    a2 = membrane_A[2, a]
                    a3 = membrane_A[3, a]
                    a4 = membrane_A[4, a]
                    dux_dx[a] += dphi1_dx * a1 + dphi2_dx * a3
                    dux_dy[a] += dphi1_dy * a1 + dphi2_dy * a3
                    duy_dx[a] += dphi1_dx * a2 + dphi2_dx * a4
                    duy_dy[a] += dphi1_dy * a2 + dphi2_dy * a4
                    ux_val[a] += phi1 * a1 + phi2 * a3
                    uy_val[a] += phi1 * a2 + phi2 * a4
                end
            end

            scale = h * abs_detJ
            if trans_mode === :curvature
                k11 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[1]
                k22 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[2]
                k12 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[3]
                for a in 1:24
                    gx1_a = dux_dx[a] - uz_val[a] * k11
                    gx2_a = duy_dx[a] - uz_val[a] * k12
                    gx3_a = duz_dx[a] + ux_val[a] * k11 + uy_val[a] * k12
                    gy1_a = dux_dy[a] - uz_val[a] * k12
                    gy2_a = duy_dy[a] - uz_val[a] * k22
                    gy3_a = duz_dy[a] + ux_val[a] * k12 + uy_val[a] * k22
                    for b in 1:24
                        gx1_b = dux_dx[b] - uz_val[b] * k11
                        gx2_b = duy_dx[b] - uz_val[b] * k12
                        gx3_b = duz_dx[b] + ux_val[b] * k11 + uy_val[b] * k12
                        gy1_b = dux_dy[b] - uz_val[b] * k12
                        gy2_b = duy_dy[b] - uz_val[b] * k22
                        gy3_b = duz_dy[b] + ux_val[b] * k12 + uy_val[b] * k22
                        Kg[a, b] += scale * (
                            s_xx * (gx1_a * gx1_b + gx2_a * gx2_b + gx3_a * gx3_b) +
                            s_yy * (gy1_a * gy1_b + gy2_a * gy2_b + gy3_a * gy3_b) +
                            s_xy * (
                                gx1_a * gy1_b + gx2_a * gy2_b + gx3_a * gy3_b +
                                gy1_a * gx1_b + gy2_a * gx2_b + gy3_a * gx3_b
                            )
                        )
                    end
                end
                if rot_grad_scale > 0.0
                    for i in 1:4
                        dNi_dx = iJ11*dNr[i] + iJ12*dNs[i]
                        dNi_dy = iJ21*dNr[i] + iJ22*dNs[i]
                        Ni = Nvals[i]
                        rx_x_i = SVector(0.0, dNi_dx, Ni * k12)
                        rx_y_i = SVector(0.0, dNi_dy, Ni * k22)
                        ry_x_i = SVector(dNi_dx, 0.0, Ni * k11)
                        ry_y_i = SVector(dNi_dy, 0.0, Ni * k12)
                        for j in 1:4
                            dNj_dx = iJ11*dNr[j] + iJ12*dNs[j]
                            dNj_dy = iJ21*dNr[j] + iJ22*dNs[j]
                            Nj = Nvals[j]
                            rot_scale = rot_grad_scale * (h^3 / 12.0) * abs_detJ
                            rx_x_j = SVector(0.0, dNj_dx, Nj * k12)
                            rx_y_j = SVector(0.0, dNj_dy, Nj * k22)
                            ry_x_j = SVector(dNj_dx, 0.0, Nj * k11)
                            ry_y_j = SVector(dNj_dy, 0.0, Nj * k12)
                            rx_val = rot_scale * shell_geometric_metric3(
                                s_xx, s_yy, s_xy, rx_x_i, rx_y_i, rx_x_j, rx_y_j)
                            ry_val = rot_scale * shell_geometric_metric3(
                                s_xx, s_yy, s_xy, ry_x_i, ry_y_i, ry_x_j, ry_y_j)
                            rx_ry_val = -rot_scale * shell_geometric_metric3(
                                s_xx, s_yy, s_xy, rx_x_i, rx_y_i, ry_x_j, ry_y_j)
                            ry_rx_val = -rot_scale * shell_geometric_metric3(
                                s_xx, s_yy, s_xy, ry_x_i, ry_y_i, rx_x_j, rx_y_j)
                            row_rx = (i-1)*6 + 4
                            col_rx = (j-1)*6 + 4
                            row_ry = (i-1)*6 + 5
                            col_ry = (j-1)*6 + 5
                            Kg[row_rx, col_rx] += rx_val
                            Kg[row_ry, col_ry] += ry_val
                            Kg[row_rx, col_ry] += rx_ry_val
                            Kg[row_ry, col_rx] += ry_rx_val
                        end
                    end
                end
            elseif trans_mode === :normal_only
                add_geometric_gradient_block!(Kg, duz_dx, duz_dy, scale, s_xx, s_yy, s_xy)
            elseif trans_mode === :principal_transverse
                add_geometric_principal_transverse_block!(
                    Kg,
                    dux_dx,
                    dux_dy,
                    duy_dx,
                    duy_dy,
                    duz_dx,
                    duz_dy,
                    scale,
                    s_xx,
                    s_yy,
                    s_xy,
                    principal_shear_yy_factor,
                    principal_shear_xy_factor,
                    principal_shear_z_factor,
                    principal_shear_ratio_min,
                )
            else
                add_geometric_gradient_block!(Kg, dux_dx, dux_dy, scale, s_xx, s_yy, s_xy)
                add_geometric_gradient_block!(Kg, duy_dx, duy_dy, scale, s_xx, s_yy, s_xy)
                add_geometric_gradient_block!(Kg, duz_dx, duz_dy, scale, s_xx, s_yy, s_xy)
                if rot_grad_scale > 0.0
                    for i in 1:4
                        dNi_dx = iJ11*dNr[i] + iJ12*dNs[i]
                        dNi_dy = iJ21*dNr[i] + iJ22*dNs[i]
                        for j in 1:4
                            dNj_dx = iJ11*dNr[j] + iJ12*dNs[j]
                            dNj_dy = iJ21*dNr[j] + iJ22*dNs[j]
                            sxy_term = if membrane_shear_center_row
                                dNdx_c[i] * dNdy_c[j] + dNdy_c[i] * dNdx_c[j]
                            else
                                dNi_dx * dNj_dy + dNi_dy * dNj_dx
                            end
                            rot_val = rot_grad_scale * (h^3 / 12.0) * abs_detJ * (
                                s_xx * dNi_dx * dNj_dx +
                                s_yy * dNi_dy * dNj_dy +
                                s_xy * sxy_term
                            )
                            row_rx = (i-1)*6 + 4
                            col_rx = (j-1)*6 + 4
                            row_ry = (i-1)*6 + 5
                            col_ry = (j-1)*6 + 5
                            Kg[row_rx, col_rx] += rot_val
                            Kg[row_ry, col_ry] += rot_val
                        end
                    end
                end
            end
        else
            # Shape function derivatives in physical coordinates + geometric stiffness
            for i in 1:4
                dNi_dx = iJ11*dNr[i] + iJ12*dNs[i]
                dNi_dy = iJ21*dNr[i] + iJ22*dNs[i]
                for j in 1:4
                    dNj_dx = iJ11*dNr[j] + iJ12*dNs[j]
                    dNj_dy = iJ21*dNr[j] + iJ22*dNs[j]
                    if trans_mode === :curvature
                    Ni = Nvals[i]
                    Nj = Nvals[j]
                    k11 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[1]
                    k22 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[2]
                    k12 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[3]
                    Axi11 = dNi_dx; Axi12 = 0.0;    Axi13 = -Ni * k11
                    Axi21 = 0.0;    Axi22 = dNi_dx; Axi23 = -Ni * k12
                    Axi31 = Ni * k11; Axi32 = Ni * k12; Axi33 = dNi_dx
                    Ayi11 = dNi_dy; Ayi12 = 0.0;    Ayi13 = -Ni * k12
                    Ayi21 = 0.0;    Ayi22 = dNi_dy; Ayi23 = -Ni * k22
                    Ayi31 = Ni * k12; Ayi32 = Ni * k22; Ayi33 = dNi_dy
                    Axj11 = dNj_dx; Axj12 = 0.0;    Axj13 = -Nj * k11
                    Axj21 = 0.0;    Axj22 = dNj_dx; Axj23 = -Nj * k12
                    Axj31 = Nj * k11; Axj32 = Nj * k12; Axj33 = dNj_dx
                    Ayj11 = dNj_dy; Ayj12 = 0.0;    Ayj13 = -Nj * k12
                    Ayj21 = 0.0;    Ayj22 = dNj_dy; Ayj23 = -Nj * k22
                    Ayj31 = Nj * k12; Ayj32 = Nj * k22; Ayj33 = dNj_dy
                    scale = h * abs_detJ
                    row0 = (i-1)*6
                    col0 = (j-1)*6
                    Kg[row0+1, col0+1] += scale * (
                        s_xx * (Axi11*Axj11 + Axi21*Axj21 + Axi31*Axj31) +
                        s_yy * (Ayi11*Ayj11 + Ayi21*Ayj21 + Ayi31*Ayj31) +
                        s_xy * (Axi11*Ayj11 + Axi21*Ayj21 + Axi31*Ayj31 +
                                Ayi11*Axj11 + Ayi21*Axj21 + Ayi31*Axj31)
                    )
                    Kg[row0+1, col0+2] += scale * (
                        s_xx * (Axi11*Axj12 + Axi21*Axj22 + Axi31*Axj32) +
                        s_yy * (Ayi11*Ayj12 + Ayi21*Ayj22 + Ayi31*Ayj32) +
                        s_xy * (Axi11*Ayj12 + Axi21*Ayj22 + Axi31*Ayj32 +
                                Ayi11*Axj12 + Ayi21*Axj22 + Ayi31*Axj32)
                    )
                    Kg[row0+1, col0+3] += scale * (
                        s_xx * (Axi11*Axj13 + Axi21*Axj23 + Axi31*Axj33) +
                        s_yy * (Ayi11*Ayj13 + Ayi21*Ayj23 + Ayi31*Ayj33) +
                        s_xy * (Axi11*Ayj13 + Axi21*Ayj23 + Axi31*Ayj33 +
                                Ayi11*Axj13 + Ayi21*Axj23 + Ayi31*Axj33)
                    )
                    Kg[row0+2, col0+1] += scale * (
                        s_xx * (Axi12*Axj11 + Axi22*Axj21 + Axi32*Axj31) +
                        s_yy * (Ayi12*Ayj11 + Ayi22*Ayj21 + Ayi32*Ayj31) +
                        s_xy * (Axi12*Ayj11 + Axi22*Ayj21 + Axi32*Ayj31 +
                                Ayi12*Axj11 + Ayi22*Axj21 + Ayi32*Axj31)
                    )
                    Kg[row0+2, col0+2] += scale * (
                        s_xx * (Axi12*Axj12 + Axi22*Axj22 + Axi32*Axj32) +
                        s_yy * (Ayi12*Ayj12 + Ayi22*Ayj22 + Ayi32*Ayj32) +
                        s_xy * (Axi12*Ayj12 + Axi22*Ayj22 + Axi32*Ayj32 +
                                Ayi12*Axj12 + Ayi22*Axj22 + Ayi32*Axj32)
                    )
                    Kg[row0+2, col0+3] += scale * (
                        s_xx * (Axi12*Axj13 + Axi22*Axj23 + Axi32*Axj33) +
                        s_yy * (Ayi12*Ayj13 + Ayi22*Ayj23 + Ayi32*Ayj33) +
                        s_xy * (Axi12*Ayj13 + Axi22*Ayj23 + Axi32*Ayj33 +
                                Ayi12*Axj13 + Ayi22*Axj23 + Ayi32*Axj33)
                    )
                    Kg[row0+3, col0+1] += scale * (
                        s_xx * (Axi13*Axj11 + Axi23*Axj21 + Axi33*Axj31) +
                        s_yy * (Ayi13*Ayj11 + Ayi23*Ayj21 + Ayi33*Ayj31) +
                        s_xy * (Axi13*Ayj11 + Axi23*Ayj21 + Axi33*Ayj31 +
                                Ayi13*Axj11 + Ayi23*Axj21 + Ayi33*Axj31)
                    )
                    Kg[row0+3, col0+2] += scale * (
                        s_xx * (Axi13*Axj12 + Axi23*Axj22 + Axi33*Axj32) +
                        s_yy * (Ayi13*Ayj12 + Ayi23*Ayj22 + Ayi33*Ayj32) +
                        s_xy * (Axi13*Ayj12 + Axi23*Ayj22 + Axi33*Ayj32 +
                                Ayi13*Axj12 + Ayi23*Axj22 + Ayi33*Axj32)
                    )
                    Kg[row0+3, col0+3] += scale * (
                        s_xx * (Axi13*Axj13 + Axi23*Axj23 + Axi33*Axj33) +
                        s_yy * (Ayi13*Ayj13 + Ayi23*Ayj23 + Ayi33*Ayj33) +
                        s_xy * (Axi13*Ayj13 + Axi23*Ayj23 + Axi33*Ayj33 +
                                Ayi13*Axj13 + Ayi23*Axj23 + Ayi33*Axj33)
                    )
                    if rot_grad_scale > 0.0
                        rot_scale = rot_grad_scale * (h^3 / 12.0) * abs_detJ
                        rx_x_i = SVector(Axi12, Axi22, Axi32)
                        rx_y_i = SVector(Ayi12, Ayi22, Ayi32)
                        ry_x_i = SVector(Axi11, Axi21, Axi31)
                        ry_y_i = SVector(Ayi11, Ayi21, Ayi31)
                        rx_x_j = SVector(Axj12, Axj22, Axj32)
                        rx_y_j = SVector(Ayj12, Ayj22, Ayj32)
                        ry_x_j = SVector(Axj11, Axj21, Axj31)
                        ry_y_j = SVector(Ayj11, Ayj21, Ayj31)
                        Kg[row0+4, col0+4] += rot_scale * shell_geometric_metric3(
                            s_xx, s_yy, s_xy, rx_x_i, rx_y_i, rx_x_j, rx_y_j)
                        Kg[row0+5, col0+5] += rot_scale * shell_geometric_metric3(
                            s_xx, s_yy, s_xy, ry_x_i, ry_y_i, ry_x_j, ry_y_j)
                        Kg[row0+4, col0+5] += -rot_scale * shell_geometric_metric3(
                            s_xx, s_yy, s_xy, rx_x_i, rx_y_i, ry_x_j, ry_y_j)
                        Kg[row0+5, col0+4] += -rot_scale * shell_geometric_metric3(
                            s_xx, s_yy, s_xy, ry_x_i, ry_y_i, rx_x_j, rx_y_j)
                    end
                    else
                        sxy_term = if membrane_shear_center_row
                            dNdx_c[i] * dNdy_c[j] + dNdy_c[i] * dNdx_c[j]
                        else
                            dNi_dx * dNj_dy + dNi_dy * dNj_dx
                        end
                        val = h * abs_detJ * (s_xx * dNi_dx * dNj_dx +
                                               s_yy * dNi_dy * dNj_dy +
                                               s_xy * sxy_term)
                        if trans_mode === :normal_only
                            row = (i-1)*6 + 3
                            col = (j-1)*6 + 3
                            Kg[row, col] += val
                        elseif trans_mode === :principal_transverse
                            row0 = (i-1)*6
                            col0 = (j-1)*6
                            add_geometric_principal_transverse_pair!(
                                Kg,
                                row0,
                                col0,
                                dNi_dx,
                                dNi_dy,
                                dNj_dx,
                                dNj_dy,
                                h * abs_detJ,
                                s_xx,
                                s_yy,
                                s_xy,
                                principal_shear_yy_factor,
                                principal_shear_xy_factor,
                                principal_shear_z_factor,
                                principal_shear_ratio_min,
                            )
                        else
                            for d in 1:3
                                row = (i-1)*6 + d
                                col = (j-1)*6 + d
                                Kg[row, col] += val
                            end
                            if rot_grad_scale > 0.0
                                rot_val = rot_grad_scale * (h^3 / 12.0) * abs_detJ * (
                                    s_xx * dNi_dx * dNj_dx +
                                    s_yy * dNi_dy * dNj_dy +
                                    s_xy * sxy_term
                                )
                                row_rx = (i-1)*6 + 4
                                col_rx = (j-1)*6 + 4
                                row_ry = (i-1)*6 + 5
                                col_ry = (j-1)*6 + 5
                                Kg[row_rx, col_rx] += rot_val
                                Kg[row_ry, col_ry] += rot_val
                            end
                        end
                    end
                end
            end
        end
    end

    return Kg
end

# Geometric stiffness for CTRIA3 shell element (18×18)
# Constant strain triangle — single integration point.
function geometric_stiffness_tria3(coords::AbstractMatrix, sigma_mem::AbstractVector, h::Float64;
                                   trans_mode::Symbol=:all,
                                   curvature::Union{Nothing,SVector{3,Float64}}=nothing,
                                   curvature_sign::Float64=1.0)
    Kg = zeros(18, 18)
    if h < 1e-30; return Kg; end

    x, y = coords[:,1], coords[:,2]
    A2 = x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2])
    A = 0.5 * abs(A2)
    if A < 1e-12; return Kg; end

    s_xx = sigma_mem[1]; s_yy = sigma_mem[2]; s_xy = sigma_mem[3]

    # Shape function derivatives (constant for CST)
    bv = [y[2]-y[3], y[3]-y[1], y[1]-y[2]] ./ (2*A)  # dN/dx
    cv = [x[3]-x[2], x[1]-x[3], x[2]-x[1]] ./ (2*A)  # dN/dy
    Nctr = 1.0 / 3.0

    for i in 1:3
        for j in 1:3
            if trans_mode === :curvature
                k11 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[1]
                k22 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[2]
                k12 = isnothing(curvature) ? 0.0 : curvature_sign * curvature[3]
                Axi11 = bv[i]; Axi12 = 0.0;   Axi13 = -Nctr * k11
                Axi21 = 0.0;   Axi22 = bv[i]; Axi23 = -Nctr * k12
                Axi31 = Nctr * k11; Axi32 = Nctr * k12; Axi33 = bv[i]
                Ayi11 = cv[i]; Ayi12 = 0.0;   Ayi13 = -Nctr * k12
                Ayi21 = 0.0;   Ayi22 = cv[i]; Ayi23 = -Nctr * k22
                Ayi31 = Nctr * k12; Ayi32 = Nctr * k22; Ayi33 = cv[i]
                Axj11 = bv[j]; Axj12 = 0.0;   Axj13 = -Nctr * k11
                Axj21 = 0.0;   Axj22 = bv[j]; Axj23 = -Nctr * k12
                Axj31 = Nctr * k11; Axj32 = Nctr * k12; Axj33 = bv[j]
                Ayj11 = cv[j]; Ayj12 = 0.0;   Ayj13 = -Nctr * k12
                Ayj21 = 0.0;   Ayj22 = cv[j]; Ayj23 = -Nctr * k22
                Ayj31 = Nctr * k12; Ayj32 = Nctr * k22; Ayj33 = cv[j]
                scale = h * A
                row0 = (i-1)*6
                col0 = (j-1)*6
                Kg[row0+1, col0+1] += scale * (
                    s_xx * (Axi11*Axj11 + Axi21*Axj21 + Axi31*Axj31) +
                    s_yy * (Ayi11*Ayj11 + Ayi21*Ayj21 + Ayi31*Ayj31) +
                    s_xy * (Axi11*Ayj11 + Axi21*Ayj21 + Axi31*Ayj31 +
                            Ayi11*Axj11 + Ayi21*Axj21 + Ayi31*Axj31)
                )
                Kg[row0+1, col0+2] += scale * (
                    s_xx * (Axi11*Axj12 + Axi21*Axj22 + Axi31*Axj32) +
                    s_yy * (Ayi11*Ayj12 + Ayi21*Ayj22 + Ayi31*Ayj32) +
                    s_xy * (Axi11*Ayj12 + Axi21*Ayj22 + Axi31*Ayj32 +
                            Ayi11*Axj12 + Ayi21*Axj22 + Ayi31*Axj32)
                )
                Kg[row0+1, col0+3] += scale * (
                    s_xx * (Axi11*Axj13 + Axi21*Axj23 + Axi31*Axj33) +
                    s_yy * (Ayi11*Ayj13 + Ayi21*Ayj23 + Ayi31*Ayj33) +
                    s_xy * (Axi11*Ayj13 + Axi21*Ayj23 + Axi31*Ayj33 +
                            Ayi11*Axj13 + Ayi21*Axj23 + Ayi31*Axj33)
                )
                Kg[row0+2, col0+1] += scale * (
                    s_xx * (Axi12*Axj11 + Axi22*Axj21 + Axi32*Axj31) +
                    s_yy * (Ayi12*Ayj11 + Ayi22*Ayj21 + Ayi32*Ayj31) +
                    s_xy * (Axi12*Ayj11 + Axi22*Ayj21 + Axi32*Ayj31 +
                            Ayi12*Axj11 + Ayi22*Axj21 + Ayi32*Axj31)
                )
                Kg[row0+2, col0+2] += scale * (
                    s_xx * (Axi12*Axj12 + Axi22*Axj22 + Axi32*Axj32) +
                    s_yy * (Ayi12*Ayj12 + Ayi22*Ayj22 + Ayi32*Ayj32) +
                    s_xy * (Axi12*Ayj12 + Axi22*Ayj22 + Axi32*Ayj32 +
                            Ayi12*Axj12 + Ayi22*Axj22 + Ayi32*Axj32)
                )
                Kg[row0+2, col0+3] += scale * (
                    s_xx * (Axi12*Axj13 + Axi22*Axj23 + Axi32*Axj33) +
                    s_yy * (Ayi12*Ayj13 + Ayi22*Ayj23 + Ayi32*Ayj33) +
                    s_xy * (Axi12*Ayj13 + Axi22*Ayj23 + Axi32*Ayj33 +
                            Ayi12*Axj13 + Ayi22*Axj23 + Ayi32*Axj33)
                )
                Kg[row0+3, col0+1] += scale * (
                    s_xx * (Axi13*Axj11 + Axi23*Axj21 + Axi33*Axj31) +
                    s_yy * (Ayi13*Ayj11 + Ayi23*Ayj21 + Ayi33*Ayj31) +
                    s_xy * (Axi13*Ayj11 + Axi23*Ayj21 + Axi33*Ayj31 +
                            Ayi13*Axj11 + Ayi23*Axj21 + Ayi33*Axj31)
                )
                Kg[row0+3, col0+2] += scale * (
                    s_xx * (Axi13*Axj12 + Axi23*Axj22 + Axi33*Axj32) +
                    s_yy * (Ayi13*Ayj12 + Ayi23*Ayj22 + Ayi33*Ayj32) +
                    s_xy * (Axi13*Ayj12 + Axi23*Ayj22 + Axi33*Ayj32 +
                            Ayi13*Axj12 + Ayi23*Axj22 + Ayi33*Axj32)
                )
                Kg[row0+3, col0+3] += scale * (
                    s_xx * (Axi13*Axj13 + Axi23*Axj23 + Axi33*Axj33) +
                    s_yy * (Ayi13*Ayj13 + Ayi23*Ayj23 + Ayi33*Ayj33) +
                    s_xy * (Axi13*Ayj13 + Axi23*Ayj23 + Axi33*Ayj33 +
                            Ayi13*Axj13 + Ayi23*Axj23 + Ayi33*Axj33)
                )
            else
                val = h * A * (s_xx * bv[i] * bv[j] +
                               s_yy * cv[i] * cv[j] +
                               s_xy * (bv[i] * cv[j] + cv[i] * bv[j]))
                if trans_mode === :normal_only
                    row = (i-1)*6 + 3
                    col = (j-1)*6 + 3
                    Kg[row, col] += val
                else
                    for d in 1:3
                        row = (i-1)*6 + d
                        col = (j-1)*6 + d
                        Kg[row, col] += val
                    end
                end
            end
        end
    end

    return Kg
end

function geometric_stiffness_quad4_covariant(coords3d::AbstractMatrix, sigma_mem::AbstractVector, h::Float64,
                                             basis1::SVector{3,Float64}, basis2::SVector{3,Float64};
                                             trans_mode::Symbol=:all,
                                             rot_grad_scale::Float64=0.0)
    sigma_gp = zeros(4, 3)
    @inbounds for gp in 1:4
        sigma_gp[gp, 1] = sigma_mem[1]
        sigma_gp[gp, 2] = sigma_mem[2]
        sigma_gp[gp, 3] = sigma_mem[3]
    end
    return geometric_stiffness_quad4_covariant(coords3d, sigma_gp, h, basis1, basis2;
                                               trans_mode=trans_mode,
                                               rot_grad_scale=rot_grad_scale)
end

function geometric_stiffness_quad4_covariant(coords3d::AbstractMatrix, sigma_mem_gp::AbstractMatrix, h::Float64,
                                             basis1::SVector{3,Float64}, basis2::SVector{3,Float64};
                                             trans_mode::Symbol=:all,
                                             rot_grad_scale::Float64=0.0)
    Kg = zeros(24, 24)
    if h < 1e-30; return Kg; end

    pt = 1.0 / sqrt(3.0)
    gauss_pts = (SVector(-pt,-pt), SVector(pt,-pt), SVector(pt,pt), SVector(-pt,pt))

    @inbounds @fastmath for gp in 1:4
        s_xx = sigma_mem_gp[gp, 1]
        s_yy = sigma_mem_gp[gp, 2]
        s_xy = sigma_mem_gp[gp, 3]
        r, s = gauss_pts[gp][1], gauss_pts[gp][2]
        dNr, dNs = shape_derivs_quad(r, s)

        a_r = SVector(
            dNr[1]*coords3d[1,1] + dNr[2]*coords3d[2,1] + dNr[3]*coords3d[3,1] + dNr[4]*coords3d[4,1],
            dNr[1]*coords3d[1,2] + dNr[2]*coords3d[2,2] + dNr[3]*coords3d[3,2] + dNr[4]*coords3d[4,2],
            dNr[1]*coords3d[1,3] + dNr[2]*coords3d[2,3] + dNr[3]*coords3d[3,3] + dNr[4]*coords3d[4,3],
        )
        a_s = SVector(
            dNs[1]*coords3d[1,1] + dNs[2]*coords3d[2,1] + dNs[3]*coords3d[3,1] + dNs[4]*coords3d[4,1],
            dNs[1]*coords3d[1,2] + dNs[2]*coords3d[2,2] + dNs[3]*coords3d[3,2] + dNs[4]*coords3d[4,2],
            dNs[1]*coords3d[1,3] + dNs[2]*coords3d[2,3] + dNs[3]*coords3d[3,3] + dNs[4]*coords3d[4,3],
        )

        g11 = dot(a_r, a_r)
        g12 = dot(a_r, a_s)
        g22 = dot(a_s, a_s)
        detg = g11 * g22 - g12 * g12
        if abs(detg) < 1e-14
            detg = detg < 0.0 ? -1e-14 : 1e-14
        end
        invg11 = g22 / detg
        invg12 = -g12 / detg
        invg22 = g11 / detg
        a_r_contra = invg11 * a_r + invg12 * a_s
        a_s_contra = invg12 * a_r + invg22 * a_s
        dA = norm(cross(a_r, a_s))
        dA < 1e-12 && continue

        for i in 1:4
            gradNi = dNr[i] * a_r_contra + dNs[i] * a_s_contra
            dNi_dx = dot(gradNi, basis1)
            dNi_dy = dot(gradNi, basis2)
            for j in 1:4
                gradNj = dNr[j] * a_r_contra + dNs[j] * a_s_contra
                dNj_dx = dot(gradNj, basis1)
                dNj_dy = dot(gradNj, basis2)
                val = h * dA * (s_xx * dNi_dx * dNj_dx +
                                s_yy * dNi_dy * dNj_dy +
                                s_xy * (dNi_dx * dNj_dy + dNi_dy * dNj_dx))
                if trans_mode === :normal_only
                    row = (i-1)*6 + 3
                    col = (j-1)*6 + 3
                    Kg[row, col] += val
                else
                    for d in 1:3
                        row = (i-1)*6 + d
                        col = (j-1)*6 + d
                        Kg[row, col] += val
                    end
                    if rot_grad_scale > 0.0
                        rot_val = rot_grad_scale * (h^3 / 12.0) * dA * (
                            s_xx * dNi_dx * dNj_dx +
                            s_yy * dNi_dy * dNj_dy +
                            s_xy * (dNi_dx * dNj_dy + dNi_dy * dNj_dx)
                        )
                        row_rx = (i-1)*6 + 4
                        col_rx = (j-1)*6 + 4
                        row_ry = (i-1)*6 + 5
                        col_ry = (j-1)*6 + 5
                        Kg[row_rx, col_rx] += rot_val
                        Kg[row_ry, col_ry] += rot_val
                    end
                end
            end
        end
    end

    return Kg
end

# =============================================================================
# MASS MATRICES — SOL 103 Normal Modes
# =============================================================================

"""
    consistent_mass_quad4(coords, rho, h) -> 24×24 Matrix

Consistent mass matrix for a 4-node bilinear shell element.
Integrates M = ∫ ρh NᵀN dA using 2×2 Gauss quadrature.
Translational DOFs (1,2,3) get full inertia; rotational DOFs (4,5) get
h²/12 rotary inertia; drilling DOF (6) gets zero mass.
"""
function consistent_mass_quad4(coords::AbstractMatrix, rho::Float64, h::Float64)
    Me = zeros(24, 24)
    if rho < 1e-30 || h < 1e-30; return Me; end

    pt = 1.0 / sqrt(3.0)
    gauss_pts = ((-pt,-pt), (pt,-pt), (pt,pt), (-pt,pt))

    for (r, s) in gauss_pts
        N1 = 0.25*(1-r)*(1-s); N2 = 0.25*(1+r)*(1-s)
        N3 = 0.25*(1+r)*(1+s); N4 = 0.25*(1-r)*(1+s)
        Nv = SVector(N1, N2, N3, N4)

        dNr = SVector(-0.25*(1-s), 0.25*(1-s), 0.25*(1+s), -0.25*(1+s))
        dNs = SVector(-0.25*(1-r), -0.25*(1+r), 0.25*(1+r), 0.25*(1-r))
        J11 = dNr[1]*coords[1,1]+dNr[2]*coords[2,1]+dNr[3]*coords[3,1]+dNr[4]*coords[4,1]
        J12 = dNr[1]*coords[1,2]+dNr[2]*coords[2,2]+dNr[3]*coords[3,2]+dNr[4]*coords[4,2]
        J21 = dNs[1]*coords[1,1]+dNs[2]*coords[2,1]+dNs[3]*coords[3,1]+dNs[4]*coords[4,1]
        J22 = dNs[1]*coords[1,2]+dNs[2]*coords[2,2]+dNs[3]*coords[3,2]+dNs[4]*coords[4,2]
        detJ = abs(J11*J22 - J12*J21)
        if detJ < 1e-30; continue; end

        mass_t = rho * h * detJ       # translational mass per unit area × |J|
        mass_r = rho * h^3/12 * detJ  # rotary inertia

        for j in 1:4, i in 1:4
            NiNj = Nv[i] * Nv[j]
            bi = (i-1)*6; bj = (j-1)*6
            # Translational DOFs (u, v, w)
            Me[bi+1, bj+1] += mass_t * NiNj
            Me[bi+2, bj+2] += mass_t * NiNj
            Me[bi+3, bj+3] += mass_t * NiNj
            # Rotational DOFs (rx, ry) — rotary inertia
            Me[bi+4, bj+4] += mass_r * NiNj
            Me[bi+5, bj+5] += mass_r * NiNj
            # DOF 6 (drilling): zero mass
        end
    end
    return Me
end

"""
    consistent_mass_tria3(coords, rho, h) -> 18×18 Matrix

Consistent mass matrix for a 3-node constant-strain triangle shell element.
Analytical integration (no quadrature needed for linear shape functions).
"""
function consistent_mass_tria3(coords::AbstractMatrix, rho::Float64, h::Float64)
    Me = zeros(18, 18)
    if rho < 1e-30 || h < 1e-30; return Me; end

    # Triangle area
    x1, y1 = coords[1,1], coords[1,2]
    x2, y2 = coords[2,1], coords[2,2]
    x3, y3 = coords[3,1], coords[3,2]
    A = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    if A < 1e-30; return Me; end

    mass_t = rho * h * A
    mass_r = rho * h^3/12 * A

    # Consistent mass for linear triangle: M_ij = (ρhA/12) * (1+δ_ij)
    # i.e. diagonal = ρhA/6, off-diagonal = ρhA/12
    for i in 1:3, j in 1:3
        bi = (i-1)*6; bj = (j-1)*6
        factor = (i == j) ? mass_t/6.0 : mass_t/12.0
        factor_r = (i == j) ? mass_r/6.0 : mass_r/12.0
        Me[bi+1, bj+1] += factor
        Me[bi+2, bj+2] += factor
        Me[bi+3, bj+3] += factor
        Me[bi+4, bj+4] += factor_r
        Me[bi+5, bj+5] += factor_r
    end
    return Me
end

"""
    consistent_mass_frame3d(L, rho, A, Iy, Iz) -> 12×12 Matrix

Consistent mass matrix for a 3D Euler-Bernoulli beam element.
"""
function consistent_mass_frame3d(L::Float64, rho::Float64, A::Float64,
                                 Iy::Float64, Iz::Float64)
    Me = zeros(12, 12)
    if L < 1e-12 || rho < 1e-30 || A < 1e-30; return Me; end

    m = rho * A * L  # total element mass

    # Axial (DOFs 1, 7)
    Me[1,1] = m/3;   Me[7,7] = m/3
    Me[1,7] = m/6;   Me[7,1] = m/6

    # Torsional (DOFs 4, 10) — use polar moment Ip = Iy + Iz
    Ip = Iy + Iz
    if Ip > 0
        mt = rho * Ip * L
        Me[4,4] = mt/3;   Me[10,10] = mt/3
        Me[4,10] = mt/6;  Me[10,4] = mt/6
    end

    # Bending in XZ plane (DOFs 2, 6, 8, 12)
    Me[2,2] = 156*m/420;   Me[8,8] = 156*m/420
    Me[2,8] = 54*m/420;    Me[8,2] = 54*m/420
    Me[2,6] = 22*L*m/420;  Me[6,2] = 22*L*m/420
    Me[2,12] = -13*L*m/420; Me[12,2] = -13*L*m/420
    Me[6,6] = 4*L^2*m/420;  Me[12,12] = 4*L^2*m/420
    Me[6,8] = 13*L*m/420;   Me[8,6] = 13*L*m/420
    Me[6,12] = -3*L^2*m/420; Me[12,6] = -3*L^2*m/420
    Me[8,12] = -22*L*m/420;  Me[12,8] = -22*L*m/420

    # Bending in XY plane (DOFs 3, 5, 9, 11) — same structure, sign flips on coupling
    Me[3,3] = 156*m/420;   Me[9,9] = 156*m/420
    Me[3,9] = 54*m/420;    Me[9,3] = 54*m/420
    Me[3,5] = -22*L*m/420; Me[5,3] = -22*L*m/420
    Me[3,11] = 13*L*m/420; Me[11,3] = 13*L*m/420
    Me[5,5] = 4*L^2*m/420;  Me[11,11] = 4*L^2*m/420
    Me[5,9] = -13*L*m/420;  Me[9,5] = -13*L*m/420
    Me[5,11] = -3*L^2*m/420; Me[11,5] = -3*L^2*m/420
    Me[9,11] = 22*L*m/420;   Me[11,9] = 22*L*m/420

    return Me
end

"""
    consistent_mass_rod(L, rho, A) -> 12×12 Matrix

Consistent mass matrix for a rod/truss element (axial DOFs only).
Uses 6-DOF-per-node format (12×12) with mass only on axial DOFs.
"""
function consistent_mass_rod(L::Float64, rho::Float64, A::Float64)
    Me = zeros(12, 12)
    if L < 1e-12 || rho < 1e-30 || A < 1e-30; return Me; end

    m = rho * A * L
    # Axial (DOFs 1, 7)
    Me[1,1] = m/3;   Me[7,7] = m/3
    Me[1,7] = m/6;   Me[7,1] = m/6
    # Transverse (lumped for stability)
    Me[2,2] = m/2; Me[3,3] = m/2
    Me[8,8] = m/2; Me[9,9] = m/2

    return Me
end

"""
    consistent_mass_tetra4(coords, rho) -> 12×12 Matrix

Consistent translational mass matrix for a 4-node linear tetrahedron.
Only translational DOFs are present, ordered as [u1,v1,w1, ..., u4,v4,w4].
"""
function consistent_mass_tetra4(coords::AbstractMatrix{Float64}, rho::Float64)
    Me = zeros(12, 12)
    if rho < 1e-30; return Me; end

    x1, y1, z1 = coords[1,1], coords[1,2], coords[1,3]
    x2, y2, z2 = coords[2,1], coords[2,2], coords[2,3]
    x3, y3, z3 = coords[3,1], coords[3,2], coords[3,3]
    x4, y4, z4 = coords[4,1], coords[4,2], coords[4,3]

    J = @SMatrix [x2-x1 y2-y1 z2-z1;
                   x3-x1 y3-y1 z3-z1;
                   x4-x1 y4-y1 z4-z1]
    V = abs(det(J)) / 6.0
    if V < 1e-30; return Me; end

    factor = rho * V / 20.0
    for i in 1:4, j in 1:4
        mass_ij = factor * (i == j ? 2.0 : 1.0)
        bi = (i-1) * 3
        bj = (j-1) * 3
        for d in 1:3
            Me[bi+d, bj+d] += mass_ij
        end
    end

    return Me
end

"""
    consistent_mass_hexa8(coords, rho) -> 24×24 Matrix

Consistent translational mass matrix for an 8-node trilinear hexahedron.
Only translational DOFs are present, ordered as [u1,v1,w1, ..., u8,v8,w8].
"""
function consistent_mass_hexa8(coords::AbstractMatrix{Float64}, rho::Float64)
    Me = zeros(24, 24)
    if rho < 1e-30; return Me; end

    xi_n  = @SVector [-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0]
    eta_n = @SVector [-1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0]
    zet_n = @SVector [-1.0,-1.0,-1.0,-1.0, 1.0, 1.0, 1.0, 1.0]
    g = 1.0 / sqrt(3.0)
    gp = @SVector [-g, g]

    for xi in gp, eta in gp, zet in gp
        N = zeros(8)
        dN_dxi = zeros(3, 8)
        for i in 1:8
            N[i] = 0.125 * (1.0 + xi_n[i] * xi) * (1.0 + eta_n[i] * eta) * (1.0 + zet_n[i] * zet)
            dN_dxi[1,i] = 0.125 * xi_n[i]  * (1.0 + eta_n[i] * eta) * (1.0 + zet_n[i] * zet)
            dN_dxi[2,i] = 0.125 * eta_n[i] * (1.0 + xi_n[i] * xi)   * (1.0 + zet_n[i] * zet)
            dN_dxi[3,i] = 0.125 * zet_n[i] * (1.0 + xi_n[i] * xi)   * (1.0 + eta_n[i] * eta)
        end

        detJ = det(dN_dxi * coords)
        adJ = abs(detJ)
        if adJ < 1e-30; continue; end

        for i in 1:8, j in 1:8
            mass_ij = rho * adJ * N[i] * N[j]
            bi = (i-1) * 3
            bj = (j-1) * 3
            for d in 1:3
                Me[bi+d, bj+d] += mass_ij
            end
        end
    end

    return Me
end

"""
    consistent_mass_cpenta6(coords, rho) -> 18×18 Matrix

Consistent translational mass matrix for a 6-node linear wedge element.
Only translational DOFs are present, ordered as [u1,v1,w1, ..., u6,v6,w6].
"""
function consistent_mass_cpenta6(coords::AbstractMatrix{Float64}, rho::Float64)
    Me = zeros(18, 18)
    if rho < 1e-30; return Me; end

    tri_xi  = [1.0/6.0, 2.0/3.0, 1.0/6.0]
    tri_eta = [1.0/6.0, 1.0/6.0, 2.0/3.0]
    tri_w   = [1.0/6.0, 1.0/6.0, 1.0/6.0]
    g = 1.0 / sqrt(3.0)
    zet_pts = [-g, g]

    for tg in eachindex(tri_xi), zet in zet_pts
        xi = tri_xi[tg]
        eta = tri_eta[tg]
        w = tri_w[tg]
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta
        zm = (1.0 - zet) / 2.0
        zp = (1.0 + zet) / 2.0

        N = SVector(
            L1 * zm,
            L2 * zm,
            L3 * zm,
            L1 * zp,
            L2 * zp,
            L3 * zp,
        )

        dN_dxi = zeros(3, 6)
        dN_dxi[1,1] = -zm;    dN_dxi[1,2] = zm;     dN_dxi[1,3] = 0.0
        dN_dxi[1,4] = -zp;    dN_dxi[1,5] = zp;     dN_dxi[1,6] = 0.0
        dN_dxi[2,1] = -zm;    dN_dxi[2,2] = 0.0;    dN_dxi[2,3] = zm
        dN_dxi[2,4] = -zp;    dN_dxi[2,5] = 0.0;    dN_dxi[2,6] = zp
        dN_dxi[3,1] = -L1/2;  dN_dxi[3,2] = -L2/2;  dN_dxi[3,3] = -L3/2
        dN_dxi[3,4] =  L1/2;  dN_dxi[3,5] =  L2/2;  dN_dxi[3,6] =  L3/2

        detJ = det(dN_dxi * coords)
        adJ = abs(detJ)
        if adJ < 1e-30; continue; end

        for i in 1:6, j in 1:6
            mass_ij = rho * w * adJ * N[i] * N[j]
            bi = (i-1) * 3
            bj = (j-1) * 3
            for d in 1:3
                Me[bi+d, bj+d] += mass_ij
            end
        end
    end

    return Me
end

# ============================================================================
# 3D Solid Element Kernels: TETRA4, HEXA8, CPENTA6
# ============================================================================

"""
    iso_3d_constitutive(E, nu) -> D (6×6)

Isotropic 3D elasticity constitutive matrix.
Strain ordering: {εxx, εyy, εzz, γxy, γyz, γzx}
"""
function iso_3d_constitutive(E::Float64, nu::Float64)
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    d = 1.0 - nu
    s = (1.0 - 2.0 * nu) / 2.0
    D = @SMatrix [
        c*d   c*nu  c*nu  0.0   0.0   0.0;
        c*nu  c*d   c*nu  0.0   0.0   0.0;
        c*nu  c*nu  c*d   0.0   0.0   0.0;
        0.0   0.0   0.0   c*s   0.0   0.0;
        0.0   0.0   0.0   0.0   c*s   0.0;
        0.0   0.0   0.0   0.0   0.0   c*s
    ]
    return D
end

"""
    stiffness_tetra4(coords) -> Ke (12×12)

4-node constant-strain tetrahedron.
coords: 4×3 matrix of nodal coordinates [x y z].
Returns element stiffness in global coordinates.
"""
function stiffness_tetra4(coords::AbstractMatrix{Float64}, E::Float64, nu::Float64)
    # Jacobian: J = [x2-x1 y2-y1 z2-z1; x3-x1 y3-y1 z3-z1; x4-x1 y4-y1 z4-z1]
    x1,y1,z1 = coords[1,1], coords[1,2], coords[1,3]
    x2,y2,z2 = coords[2,1], coords[2,2], coords[2,3]
    x3,y3,z3 = coords[3,1], coords[3,2], coords[3,3]
    x4,y4,z4 = coords[4,1], coords[4,2], coords[4,3]

    J = @SMatrix [x2-x1 y2-y1 z2-z1;
                   x3-x1 y3-y1 z3-z1;
                   x4-x1 y4-y1 z4-z1]
    detJ = det(J)
    V = abs(detJ) / 6.0
    if V < 1e-30
        return zeros(12, 12)
    end

    invJ = inv(J)

    # Shape function derivatives in natural coords: dN/d(ξ,η,ζ)
    # N1 = 1-ξ-η-ζ, N2=ξ, N3=η, N4=ζ
    # dN_dnat = [-1 1 0 0; -1 0 1 0; -1 0 0 1]  (3×4)
    # dN_dx = invJ * dN_dnat  (3×4)
    dN_dx = invJ * @SMatrix [-1.0 1.0 0.0 0.0;
                              -1.0 0.0 1.0 0.0;
                              -1.0 0.0 0.0 1.0]

    # B matrix (6×12): strain = B * u
    B = zeros(6, 12)
    for i in 1:4
        c = (i-1)*3
        dx = dN_dx[1,i]; dy = dN_dx[2,i]; dz = dN_dx[3,i]
        B[1, c+1] = dx                            # εxx
        B[2, c+2] = dy                            # εyy
        B[3, c+3] = dz                            # εzz
        B[4, c+1] = dy;  B[4, c+2] = dx           # γxy
        B[5, c+2] = dz;  B[5, c+3] = dy           # γyz
        B[6, c+1] = dz;  B[6, c+3] = dx           # γzx
    end

    D = iso_3d_constitutive(E, nu)
    Ke = V * (B' * D * B)
    return Ke
end

"""
    stiffness_hexa8(coords, E, nu) -> Ke (24×24)

8-node isoparametric hexahedron with:
  - B-bar (mean dilatation) for volumetric locking relief
  - Wilson-Taylor incompatible modes for shear/bending locking relief
  - 2×2×2 Gauss integration with static condensation of 9 internal DOFs

This combination matches Nastran CHEXA accuracy for both tension and bending.

coords: 8×3 matrix of nodal coordinates [x y z].
Nastran CHEXA node numbering:
  Bottom face: 1-2-3-4, Top face: 5-6-7-8  (5 above 1, etc.)
"""
function stiffness_hexa8(coords::AbstractMatrix{Float64}, E::Float64, nu::Float64)
    D = iso_3d_constitutive(E, nu)

    # Natural coordinates of 8 corner nodes
    xi_n  = @SVector [-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0]
    eta_n = @SVector [-1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0]
    zet_n = @SVector [-1.0,-1.0,-1.0,-1.0, 1.0, 1.0, 1.0, 1.0]

    # 2×2×2 Gauss points
    g = 1.0 / sqrt(3.0)
    gp = @SVector [-g, g]

    # --- Center Jacobian (for incompatible modes — Wilson-Taylor patch test fix) ---
    dN_dxi_0 = zeros(3, 8)
    for i in 1:8
        dN_dxi_0[1,i] = 0.125 * xi_n[i]
        dN_dxi_0[2,i] = 0.125 * eta_n[i]
        dN_dxi_0[3,i] = 0.125 * zet_n[i]
    end
    J_0 = dN_dxi_0 * coords
    detJ_0 = det(J_0)
    invJ_0 = abs(detJ_0) > 1e-30 ? inv(J_0) : zeros(3, 3)

    # --- Single pass: standard B + incompatible modes ---
    Ke_aa = zeros(24, 24)   # standard DOFs
    Ke_ai = zeros(24, 9)    # coupling: standard ↔ incompatible
    Ke_ii = zeros(9, 9)     # incompatible self-coupling
    B = zeros(6, 24)
    Bi = zeros(6, 9)

    igp = 0
    for gi in 1:2, gj in 1:2, gk in 1:2
        xi = gp[gi]; eta = gp[gj]; zet = gp[gk]
        igp += 1

        dN_dxi = zeros(3, 8)
        for i in 1:8
            dN_dxi[1,i] = 0.125 * xi_n[i]  * (1.0 + eta_n[i]*eta) * (1.0 + zet_n[i]*zet)
            dN_dxi[2,i] = 0.125 * eta_n[i] * (1.0 + xi_n[i]*xi)   * (1.0 + zet_n[i]*zet)
            dN_dxi[3,i] = 0.125 * zet_n[i] * (1.0 + xi_n[i]*xi)   * (1.0 + eta_n[i]*eta)
        end

        J = dN_dxi * coords
        adJ = abs(det(J))
        if adJ < 1e-30; continue; end
        dN_dx_local = inv(J) * dN_dxi

        # --- Standard B (no B-bar) ---
        fill!(B, 0.0)
        for i in 1:8
            c = (i-1)*3
            dx = dN_dx_local[1,i]; dy = dN_dx_local[2,i]; dz = dN_dx_local[3,i]
            B[1,c+1] = dx; B[2,c+2] = dy; B[3,c+3] = dz
            B[4,c+1] = dy; B[4,c+2] = dx
            B[5,c+2] = dz; B[5,c+3] = dy
            B[6,c+1] = dz; B[6,c+3] = dx
        end

        # --- Incompatible mode B-matrix (Bi) ---
        # 3 bubble functions: φ₁=1-ξ², φ₂=1-η², φ₃=1-ζ²
        # Natural derivatives: dφ₁/dξ=-2ξ, dφ₂/dη=-2η, dφ₃/dζ=-2ζ (all others 0)
        # Physical derivatives use center Jacobian J₀ (Wilson-Taylor patch test fix)
        # Scaling by det(J₀)/det(J) ensures ∫Bi dV = 0 (orthogonality for patch test)
        dphi_dnat = zeros(3, 3)
        dphi_dnat[1, 1] = -2.0 * xi
        dphi_dnat[2, 2] = -2.0 * eta
        dphi_dnat[3, 3] = -2.0 * zet
        scale = abs(detJ_0) / adJ  # det(J₀)/det(J) scaling
        dphi_dx = (scale .* invJ_0) * dphi_dnat

        # Build Bi (6×9): 9 internal DOFs = 3 bubbles × 3 directions
        fill!(Bi, 0.0)
        for m in 1:3  # bubble function index
            gx = dphi_dx[1,m]; gy = dphi_dx[2,m]; gz = dphi_dx[3,m]
            # Direction x: col = (m-1)*3 + 1
            cx = (m-1)*3 + 1
            Bi[1, cx] = gx; Bi[4, cx] = gy; Bi[6, cx] = gz
            # Direction y: col = (m-1)*3 + 2
            cy = (m-1)*3 + 2
            Bi[2, cy] = gy; Bi[4, cy] = gx; Bi[5, cy] = gz
            # Direction z: col = (m-1)*3 + 3
            cz = (m-1)*3 + 3
            Bi[3, cz] = gz; Bi[5, cz] = gy; Bi[6, cz] = gx
        end

        # Accumulate stiffness sub-matrices
        DB = D * B
        DBi = D * Bi
        Ke_aa .+= adJ .* (B' * DB)
        Ke_ai .+= adJ .* (B' * DBi)
        Ke_ii .+= adJ .* (Bi' * DBi)
    end

    # --- Static condensation: K = Ke_aa - Ke_ai * inv(Ke_ii) * Ke_ai' ---
    if abs(det(Ke_ii)) > 1e-30
        Ke_ii_inv = inv(Ke_ii)
        Ke = Ke_aa - Ke_ai * Ke_ii_inv * Ke_ai'
    else
        Ke = Ke_aa  # fallback: no condensation if singular
    end

    return Ke
end

"""
    stiffness_cpenta6(coords) -> Ke (18×18)

6-node pentahedral (wedge) element with 2-point Gauss in ζ × 1-point in triangle.
coords: 6×3 matrix of nodal coordinates.
Nastran CPENTA node numbering:
  Bottom triangle: 1-2-3, Top triangle: 4-5-6  (4 above 1, etc.)
"""
function stiffness_cpenta6(coords::AbstractMatrix{Float64}, E::Float64, nu::Float64)
    D = iso_3d_constitutive(E, nu)
    Ke = zeros(18, 18)
    B  = zeros(6, 18)
    DB = zeros(6, 18)

    # Gauss integration: 3-point triangle × 2-point through thickness
    # Triangle: 3-point rule (midpoints of edges), weight = 1/6 each (total area = 1/2)
    # Through thickness: ζ = ±1/√3, weight = 1.0
    tri_xi  = [0.5, 0.5, 0.0]
    tri_eta = [0.0, 0.5, 0.5]
    tri_w   = [1.0/6.0, 1.0/6.0, 1.0/6.0]
    g = 1.0 / sqrt(3.0)
    zet_pts = [-g, g]
    zet_w   = [1.0, 1.0]

    for tg in 1:length(tri_xi), zg in 1:2
        xi  = tri_xi[tg]
        eta = tri_eta[tg]
        zet = zet_pts[zg]
        w = tri_w[tg] * zet_w[zg]

        # Shape functions for CPENTA6:
        # N1 = (1-xi-eta)*(1-zet)/2, N2 = xi*(1-zet)/2, N3 = eta*(1-zet)/2
        # N4 = (1-xi-eta)*(1+zet)/2, N5 = xi*(1+zet)/2, N6 = eta*(1+zet)/2
        L1 = 1.0 - xi - eta; L2 = xi; L3 = eta
        zm = (1.0 - zet) / 2.0; zp = (1.0 + zet) / 2.0

        # dN/d(xi, eta, zet) — 3×6
        dN_dxi = zeros(3, 6)
        # d/dxi
        dN_dxi[1,1] = -zm;  dN_dxi[1,2] = zm;  dN_dxi[1,3] = 0.0
        dN_dxi[1,4] = -zp;  dN_dxi[1,5] = zp;  dN_dxi[1,6] = 0.0
        # d/deta
        dN_dxi[2,1] = -zm;  dN_dxi[2,2] = 0.0; dN_dxi[2,3] = zm
        dN_dxi[2,4] = -zp;  dN_dxi[2,5] = 0.0; dN_dxi[2,6] = zp
        # d/dzet
        dN_dxi[3,1] = -L1/2; dN_dxi[3,2] = -L2/2; dN_dxi[3,3] = -L3/2
        dN_dxi[3,4] =  L1/2; dN_dxi[3,5] =  L2/2; dN_dxi[3,6] =  L3/2

        J = dN_dxi * coords
        detJ = det(J)
        if abs(detJ) < 1e-30; continue; end
        invJ = inv(J)
        dN_dx = invJ * dN_dxi

        fill!(B, 0.0)
        for i in 1:6
            c = (i-1)*3
            dx = dN_dx[1,i]; dy = dN_dx[2,i]; dz = dN_dx[3,i]
            B[1, c+1] = dx
            B[2, c+2] = dy
            B[3, c+3] = dz
            B[4, c+1] = dy;  B[4, c+2] = dx
            B[5, c+2] = dz;  B[5, c+3] = dy
            B[6, c+1] = dz;  B[6, c+3] = dx
        end

        mul!(DB, D, B)
        Ke .+= (w * abs(detJ)) .* (B' * DB)
    end

    return Ke
end

"""
    stress_solid_3d(B, D, u_el) -> (stress, strain, von_mises)

Compute stress and strain for a solid element at a given point.
B: 6×ndof strain-displacement matrix
D: 6×6 constitutive matrix
u_el: element displacement vector (translational DOFs only)
Returns stress vector {σxx,σyy,σzz,τxy,τyz,τzx}, strain vector, and von Mises stress.
"""
function stress_solid_3d(B::AbstractMatrix{Float64}, D, u_el::AbstractVector{Float64})
    strain = B * u_el
    stress = D * strain
    # Von Mises: σvm = √(σxx² + σyy² + σzz² - σxx·σyy - σyy·σzz - σzz·σxx + 3(τxy² + τyz² + τzx²))
    sxx, syy, szz, txy, tyz, tzx = stress[1], stress[2], stress[3], stress[4], stress[5], stress[6]
    vm = sqrt(max(0.0, sxx^2 + syy^2 + szz^2 - sxx*syy - syy*szz - szz*sxx + 3.0*(txy^2 + tyz^2 + tzx^2)))
    return stress, strain, vm
end

"""
    solid_centroid_B_tetra4(coords) -> B (6×12)

Compute the B matrix at the centroid of a TETRA4 element (constant strain).
"""
function solid_centroid_B_tetra4(coords::AbstractMatrix{Float64})
    J = @SMatrix [coords[2,1]-coords[1,1] coords[2,2]-coords[1,2] coords[2,3]-coords[1,3];
                   coords[3,1]-coords[1,1] coords[3,2]-coords[1,2] coords[3,3]-coords[1,3];
                   coords[4,1]-coords[1,1] coords[4,2]-coords[1,2] coords[4,3]-coords[1,3]]
    invJ = inv(J)
    dN_dx = invJ * @SMatrix [-1.0 1.0 0.0 0.0;
                              -1.0 0.0 1.0 0.0;
                              -1.0 0.0 0.0 1.0]
    B = zeros(6, 12)
    for i in 1:4
        c = (i-1)*3
        dx = dN_dx[1,i]; dy = dN_dx[2,i]; dz = dN_dx[3,i]
        B[1, c+1] = dx
        B[2, c+2] = dy
        B[3, c+3] = dz
        B[4, c+1] = dy;  B[4, c+2] = dx
        B[5, c+2] = dz;  B[5, c+3] = dy
        B[6, c+1] = dz;  B[6, c+3] = dx
    end
    return B
end

"""
    solid_centroid_B_hexa8(coords) -> B (6×24)

Compute the B matrix at the centroid (ξ=η=ζ=0) of a HEXA8 element.
"""
function solid_centroid_B_hexa8(coords::AbstractMatrix{Float64})
    xi_n  = @SVector [-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0]
    eta_n = @SVector [-1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0]
    zet_n = @SVector [-1.0,-1.0,-1.0,-1.0, 1.0, 1.0, 1.0, 1.0]

    dN_dxi = zeros(3, 8)
    for i in 1:8
        dN_dxi[1,i] = 0.125 * xi_n[i]
        dN_dxi[2,i] = 0.125 * eta_n[i]
        dN_dxi[3,i] = 0.125 * zet_n[i]
    end
    J = dN_dxi * coords
    invJ = inv(J)
    dN_dx = invJ * dN_dxi

    B = zeros(6, 24)
    for i in 1:8
        c = (i-1)*3
        dx = dN_dx[1,i]; dy = dN_dx[2,i]; dz = dN_dx[3,i]
        B[1, c+1] = dx
        B[2, c+2] = dy
        B[3, c+3] = dz
        B[4, c+1] = dy;  B[4, c+2] = dx
        B[5, c+2] = dz;  B[5, c+3] = dy
        B[6, c+1] = dz;  B[6, c+3] = dx
    end
    return B
end

"""
    solid_centroid_B_cpenta6(coords) -> B (6×18)

Compute the B matrix at the centroid of a CPENTA6 element.
"""
function solid_centroid_B_cpenta6(coords::AbstractMatrix{Float64})
    xi = 1.0/3.0; eta = 1.0/3.0; zet = 0.0
    L1 = 1.0 - xi - eta; L2 = xi; L3 = eta
    zm = 0.5; zp = 0.5

    dN_dxi = zeros(3, 6)
    dN_dxi[1,1] = -zm;  dN_dxi[1,2] = zm;  dN_dxi[1,3] = 0.0
    dN_dxi[1,4] = -zp;  dN_dxi[1,5] = zp;  dN_dxi[1,6] = 0.0
    dN_dxi[2,1] = -zm;  dN_dxi[2,2] = 0.0; dN_dxi[2,3] = zm
    dN_dxi[2,4] = -zp;  dN_dxi[2,5] = 0.0; dN_dxi[2,6] = zp
    dN_dxi[3,1] = -L1/2; dN_dxi[3,2] = -L2/2; dN_dxi[3,3] = -L3/2
    dN_dxi[3,4] =  L1/2; dN_dxi[3,5] =  L2/2; dN_dxi[3,6] =  L3/2

    J = dN_dxi * coords
    invJ = inv(J)
    dN_dx = invJ * dN_dxi

    B = zeros(6, 18)
    for i in 1:6
        c = (i-1)*3
        dx = dN_dx[1,i]; dy = dN_dx[2,i]; dz = dN_dx[3,i]
        B[1, c+1] = dx
        B[2, c+2] = dy
        B[3, c+3] = dz
        B[4, c+1] = dy;  B[4, c+2] = dx
        B[5, c+2] = dz;  B[5, c+3] = dy
        B[6, c+1] = dz;  B[6, c+3] = dx
    end
    return B
end

# ============================================================================
# Geometric Stiffness for Solid Elements (SOL105 Buckling)
# ============================================================================

"""
    geometric_stiffness_hexa8(coords, stress) -> Kg (24×24)

Geometric stiffness matrix for 8-node hexahedron under initial stress.
stress: 6-vector {σxx, σyy, σzz, τxy, τyz, τzx} at element centroid.
Uses 2×2×2 Gauss integration.
"""
function geometric_stiffness_hexa8(coords::AbstractMatrix{Float64}, stress::AbstractVector{Float64})
    Kg = zeros(24, 24)
    sxx, syy, szz, txy, tyz, tzx = stress[1], stress[2], stress[3], stress[4], stress[5], stress[6]

    # Initial stress matrix S (3×3 symmetric)
    S = @SMatrix [sxx txy tzx;
                   txy syy tyz;
                   tzx tyz szz]

    xi_n  = @SVector [-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0,-1.0]
    eta_n = @SVector [-1.0,-1.0, 1.0, 1.0,-1.0,-1.0, 1.0, 1.0]
    zet_n = @SVector [-1.0,-1.0,-1.0,-1.0, 1.0, 1.0, 1.0, 1.0]
    g = 1.0 / sqrt(3.0)
    gp = @SVector [-g, g]

    for gi in 1:2, gj in 1:2, gk in 1:2
        xi = gp[gi]; eta = gp[gj]; zet = gp[gk]

        dN_dxi = zeros(3, 8)
        for i in 1:8
            dN_dxi[1,i] = 0.125 * xi_n[i]  * (1.0 + eta_n[i]*eta) * (1.0 + zet_n[i]*zet)
            dN_dxi[2,i] = 0.125 * eta_n[i] * (1.0 + xi_n[i]*xi)   * (1.0 + zet_n[i]*zet)
            dN_dxi[3,i] = 0.125 * zet_n[i] * (1.0 + xi_n[i]*xi)   * (1.0 + eta_n[i]*eta)
        end

        J = dN_dxi * coords
        adJ = abs(det(J))
        if adJ < 1e-30; continue; end
        dN_dx = inv(J) * dN_dxi

        # Kg contribution: Kg_IJ = ∫ (∂Ni/∂x)ᵀ S (∂Nj/∂x) * I₃  dV
        # For each node pair (I,J): Kg[(I-1)*3+a, (J-1)*3+a] += dNI·S·dNJ for a=1,2,3
        for I in 1:8, J in 1:8
            gI = SVector{3}(dN_dx[1,I], dN_dx[2,I], dN_dx[3,I])
            gJ = SVector{3}(dN_dx[1,J], dN_dx[2,J], dN_dx[3,J])
            val = dot(gI, S * gJ) * adJ
            for a in 1:3
                Kg[(I-1)*3+a, (J-1)*3+a] += val
            end
        end
    end
    return Kg
end

"""
    geometric_stiffness_tetra4(coords, stress) -> Kg (12×12)

Geometric stiffness matrix for 4-node tetrahedron under initial stress.
Constant stress → single integration point.
"""
function geometric_stiffness_tetra4(coords::AbstractMatrix{Float64}, stress::AbstractVector{Float64})
    J = @SMatrix [coords[2,1]-coords[1,1] coords[2,2]-coords[1,2] coords[2,3]-coords[1,3];
                   coords[3,1]-coords[1,1] coords[3,2]-coords[1,2] coords[3,3]-coords[1,3];
                   coords[4,1]-coords[1,1] coords[4,2]-coords[1,2] coords[4,3]-coords[1,3]]
    V = abs(det(J)) / 6.0
    if V < 1e-30; return zeros(12, 12); end

    invJ = inv(J)
    dN_dx = invJ * @SMatrix [-1.0 1.0 0.0 0.0; -1.0 0.0 1.0 0.0; -1.0 0.0 0.0 1.0]

    sxx, syy, szz, txy, tyz, tzx = stress[1], stress[2], stress[3], stress[4], stress[5], stress[6]
    S = @SMatrix [sxx txy tzx; txy syy tyz; tzx tyz szz]

    Kg = zeros(12, 12)
    for I in 1:4, J_node in 1:4
        gI = SVector{3}(dN_dx[1,I], dN_dx[2,I], dN_dx[3,I])
        gJ = SVector{3}(dN_dx[1,J_node], dN_dx[2,J_node], dN_dx[3,J_node])
        val = dot(gI, S * gJ) * V
        for a in 1:3
            Kg[(I-1)*3+a, (J_node-1)*3+a] += val
        end
    end
    return Kg
end

"""
    geometric_stiffness_cpenta6(coords, stress) -> Kg (18×18)

Geometric stiffness matrix for 6-node pentahedron under initial stress.
Uses 2-point Gauss (same as stiffness).
"""
function geometric_stiffness_cpenta6(coords::AbstractMatrix{Float64}, stress::AbstractVector{Float64})
    Kg = zeros(18, 18)
    sxx, syy, szz, txy, tyz, tzx = stress[1], stress[2], stress[3], stress[4], stress[5], stress[6]
    S = @SMatrix [sxx txy tzx; txy syy tyz; tzx tyz szz]

    g = 1.0 / sqrt(3.0)
    tri_xi = [1.0/3.0]; tri_eta = [1.0/3.0]; tri_w = [0.5]
    zet_pts = [-g, g]; zet_w = [1.0, 1.0]

    for tg in 1:1, zg in 1:2
        xi = tri_xi[tg]; eta = tri_eta[tg]; zet = zet_pts[zg]
        w = tri_w[tg] * zet_w[zg]
        L1 = 1.0 - xi - eta; L2 = xi; L3 = eta
        zm = (1.0 - zet)/2.0; zp = (1.0 + zet)/2.0

        dN_dxi = zeros(3, 6)
        dN_dxi[1,1] = -zm;  dN_dxi[1,2] = zm;  dN_dxi[1,3] = 0.0
        dN_dxi[1,4] = -zp;  dN_dxi[1,5] = zp;  dN_dxi[1,6] = 0.0
        dN_dxi[2,1] = -zm;  dN_dxi[2,2] = 0.0; dN_dxi[2,3] = zm
        dN_dxi[2,4] = -zp;  dN_dxi[2,5] = 0.0; dN_dxi[2,6] = zp
        dN_dxi[3,1] = -L1/2; dN_dxi[3,2] = -L2/2; dN_dxi[3,3] = -L3/2
        dN_dxi[3,4] =  L1/2; dN_dxi[3,5] =  L2/2; dN_dxi[3,6] =  L3/2

        J = dN_dxi * coords
        adJ = abs(det(J))
        if adJ < 1e-30; continue; end
        dN_dx = inv(J) * dN_dxi

        for I in 1:6, J_node in 1:6
            gI = SVector{3}(dN_dx[1,I], dN_dx[2,I], dN_dx[3,I])
            gJ = SVector{3}(dN_dx[1,J_node], dN_dx[2,J_node], dN_dx[3,J_node])
            val = dot(gI, S * gJ) * w * adJ
            for a in 1:3
                Kg[(I-1)*3+a, (J_node-1)*3+a] += val
            end
        end
    end
    return Kg
end

end
