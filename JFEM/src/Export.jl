# ============================================================================
# Export.jl — Export functions for JFEM
#
# This file is included at the top level (not a module) and has access to
# WriteVTK, JSON, and HDF5 packages from the parent scope.
#
# Contains:
#   export_nastran_hdf5           - compact MSC/Nastran-like HDF5 export
#   sanitize!(d)                  — NaN/Inf cleaner for JSON export
#   build_jfem_element_tables     — builds sorted element tables for JFEM binary
#   build_jfem_constraint_tables  — builds CELAS/RBE2/RBE3 tables for JFEM v3
#   collect_spc_data              — collects SPC constraints for a subcase
#   collect_point_loads           — collects FORCE/MOMENT cards for a subcase
#   collect_jfem_subcase_data     — collects per-subcase JFEM binary data
#   export_vtk_subcase            — exports a single subcase to VTK format
#   export_json                   — exports aggregated results to JSON
#   export_jfem_binary            — exports mesh + results to JFEM binary format (v4)
#   export_card_inventory         — exports card inventory to JSON
# ============================================================================

using Dates

function sanitize!(d)
    if d isa Dict
        for (k,v) in d; d[k] = sanitize!(v); end
    elseif d isa Vector
        for i in eachindex(d); d[i] = sanitize!(d[i]); end
    elseif d isa Float64
        if isnan(d) || isinf(d); return 0.0; end
    end
    return d
end

@inline _export_base_name(filename) = replace(basename(filename), r"(?i)\.bdf$" => "")

@inline function _export_entry_public_id(key, entry)
    if entry isa AbstractDict && haskey(entry, "ID")
        value = entry["ID"]
        parsed = tryparse(Int, string(value))
        parsed !== nothing && return parsed
    end
    parsed = tryparse(Int, string(key))
    parsed !== nothing && return parsed
    m = match(r"^-?\d+", string(key))
    m !== nothing && return parse(Int, m.match)
    return 0
end

# Compact MSC/Nastran-like HDF5 export.  The former recursive Dict-to-HDF5
# writer was intentionally removed because it produced very large files.

struct _MSCInputDomainRec
    ID::Int64
    SE::Int64
    AFPM::Int64
    TRMC::Int64
    MODULE::Int64
end

struct _MSCIndexRec
    DOMAIN_ID::Int64
    POSITION::Int64
    LENGTH::Int64
end

struct _MSCResultDomainRec
    ID::Int64
    SUBCASE::Int64
    STEP::Int64
    ANALYSIS::Int64
    TIME_FREQ_EIGR::Float64
    EIGI::Float64
    MODE::Int64
    DESIGN_CYCLE::Int64
    RANDOM::Int64
    SE::Int64
    AFPM::Int64
    TRMC::Int64
    INSTANCE::Int64
    MODULE::Int64
    SUBSTEP::Int64
    IMPFID::Int64
end

struct _MSCNodalResultRec
    ID::Int64
    X::Float64
    Y::Float64
    Z::Float64
    RX::Float64
    RY::Float64
    RZ::Float64
    DOMAIN_ID::Int64
end

struct _MSCGridWeightRec
    ID::Int64
    MO::NTuple{36,Float64}
    S::NTuple{9,Float64}
    MX::Float64
    XX::Float64
    YX::Float64
    ZX::Float64
    MY::Float64
    XY::Float64
    YY::Float64
    ZY::Float64
    MZ::Float64
    XZ::Float64
    YZ::Float64
    ZZ::Float64
    I::NTuple{9,Float64}
    PIX::Float64
    PIY::Float64
    PIZ::Float64
    Q::NTuple{9,Float64}
    DOMAIN_ID::Int64
end

struct _MSCEigenvalueRec
    MODE::Int64
    ORDER::Int64
    EIGEN::Float64
    OMEGA::Float64
    FREQ::Float64
    MASS::Float64
    STIFF::Float64
    RESFLG::Int64
    FLDFLG::Int64
    DOMAIN_ID::Int64
end

struct _MSCGridRec
    ID::Int64
    CP::Int64
    X::NTuple{3,Float64}
    CD::Int64
    PS::Int64
    SEID::Int64
    DOMAIN_ID::Int64
end

struct _MSCCquad4Rec
    EID::Int64
    PID::Int64
    G::NTuple{4,Int64}
    THETA::Float64
    ZOFFS::Float64
    TFLAG::Int64
    T::NTuple{4,Float64}
    MCID::Int64
    DOMAIN_ID::Int64
end

struct _MSCCTRIA3Rec
    EID::Int64
    PID::Int64
    G::NTuple{3,Int64}
    THETA::Float64
    ZOFFS::Float64
    TFLAG::Int64
    T::NTuple{3,Float64}
    MCID::Int64
    DOMAIN_ID::Int64
end

struct _MSCCrodRec
    EID::Int64
    PID::Int64
    G::NTuple{2,Int64}
    DOMAIN_ID::Int64
end

struct _MSCIdRec
    ID::Int64
end

struct _MSCCbarRec
    EID::Int64
    PID::Int64
    GA::Int64
    GB::Int64
    FLAG::Int64
    X1::Float64
    X2::Float64
    X3::Float64
    GO::Int64
    PA::Int64
    PB::Int64
    W1A::Float64
    W2A::Float64
    W3A::Float64
    W1B::Float64
    W2B::Float64
    W3B::Float64
    DOMAIN_ID::Int64
end

struct _MSCCbeamRec
    EID::Int64
    PID::Int64
    GA::Int64
    GB::Int64
    SA::Int64
    SB::Int64
    X::NTuple{3,Float64}
    G0::Int64
    F::Int64
    PA::Int64
    PB::Int64
    WA::NTuple{3,Float64}
    WB::NTuple{3,Float64}
    DOMAIN_ID::Int64
end

struct _MSCCelas1Rec
    EID::Int64
    PID::Int64
    G1::Int64
    G2::Int64
    C1::Int64
    C2::Int64
    DOMAIN_ID::Int64
end

struct _MSCCelas2Rec
    EID::Int64
    K::Float64
    G1::Int64
    G2::Int64
    C1::Int64
    C2::Int64
    GE::Float64
    S::Float64
    DOMAIN_ID::Int64
end

struct _MSCCmass1Rec
    EID::Int64
    PID::Int64
    G1::Int64
    G2::Int64
    C1::Int64
    C2::Int64
    DOMAIN_ID::Int64
end

struct _MSCCmass2Rec
    EID::Int64
    M::Float64
    G1::Int64
    G2::Int64
    C1::Int64
    C2::Int64
    DOMAIN_ID::Int64
end

struct _MSCConm2Rec
    EID::Int64
    G::Int64
    CID::Int64
    M::Float64
    X1::Float64
    X2::Float64
    X3::Float64
    I1::Float64
    I2::NTuple{2,Float64}
    I3::NTuple{3,Float64}
    DOMAIN_ID::Int64
end

struct _MSCConrodRec
    EID::Int64
    G1::Int64
    G2::Int64
    MID::Int64
    A::Float64
    J::Float64
    C::Float64
    NSM::Float64
    DOMAIN_ID::Int64
end

struct _MSCChexaRec
    EID::Int64
    PID::Int64
    G::NTuple{20,Int64}
    DOMAIN_ID::Int64
end

struct _MSCCpentaRec
    EID::Int64
    PID::Int64
    G::NTuple{15,Int64}
    DOMAIN_ID::Int64
end

struct _MSCCTetraRec
    EID::Int64
    PID::Int64
    G::NTuple{10,Int64}
    DOMAIN_ID::Int64
end

struct _MSCForceRec
    SID::Int64
    G::Int64
    CID::Int64
    F::Float64
    N::NTuple{3,Float64}
    DOMAIN_ID::Int64
end

struct _MSCMomentRec
    SID::Int64
    G::Int64
    CID::Int64
    M::Float64
    N::NTuple{3,Float64}
    DOMAIN_ID::Int64
end

struct _MSCPload4Rec
    SID::Int64
    EID::Int64
    P::NTuple{4,Float64}
    G1::Int64
    G34::Int64
    CID::Int64
    N::NTuple{3,Float64}
    SORL::NTuple{8,UInt8}
    LDIR::NTuple{8,UInt8}
    DOMAIN_ID::Int64
end

struct _MSCMat1Rec
    MID::Int64
    E::Float64
    G::Float64
    NU::Float64
    RHO::Float64
    A::Float64
    TREF::Float64
    GE::Float64
    ST::Float64
    SC::Float64
    SS::Float64
    MCSID::Int64
    DOMAIN_ID::Int64
end

struct _MSCMat2Rec
    MID::Int64
    G11::Float64
    G12::Float64
    G13::Float64
    G22::Float64
    G23::Float64
    G33::Float64
    RHO::Float64
    A1::Float64
    A2::Float64
    A12::Float64
    TREF::Float64
    GE::Float64
    ST::Float64
    SC::Float64
    SS::Float64
    MCSID::Int64
    DOMAIN_ID::Int64
end

struct _MSCMat8Rec
    MID::Int64
    E1::Float64
    E2::Float64
    NU12::Float64
    G12::Float64
    G1Z::Float64
    G2Z::Float64
    RHO::Float64
    A1::Float64
    A2::Float64
    TREF::Float64
    XT::Float64
    XC::Float64
    YT::Float64
    YC::Float64
    S::Float64
    GE::Float64
    F12::Float64
    STRN::Float64
    DOMAIN_ID::Int64
end

struct _MSCPShellRec
    PID::Int64
    MID1::Int64
    T::Float64
    MID2::Int64
    BK::Float64
    MID3::Int64
    TS::Float64
    NSM::Float64
    Z1::Float64
    Z2::Float64
    MID4::Int64
    DOMAIN_ID::Int64
end

struct _MSCPBarRec
    PID::Int64
    MID::Int64
    A::Float64
    I1::Float64
    I2::Float64
    J::Float64
    NSM::Float64
    FE::Float64
    C1::Float64
    C2::Float64
    D1::Float64
    D2::Float64
    E1::Float64
    E2::Float64
    F1::Float64
    F2::Float64
    K1::Float64
    K2::Float64
    I12::Float64
    DOMAIN_ID::Int64
end

struct _MSCPBeamRec
    PID::Int64
    MID::Int64
    NSEGS::Int64
    CCF::Int64
    CWELD::Int64
    SO::NTuple{11,Float64}
    XXB::NTuple{11,Float64}
    A::NTuple{11,Float64}
    I1::NTuple{11,Float64}
    I2::NTuple{11,Float64}
    I12::NTuple{11,Float64}
    J::NTuple{11,Float64}
    NSM::NTuple{11,Float64}
    C1::NTuple{11,Float64}
    C2::NTuple{11,Float64}
    D1::NTuple{11,Float64}
    D2::NTuple{11,Float64}
    E1::NTuple{11,Float64}
    E2::NTuple{11,Float64}
    F1::NTuple{11,Float64}
    F2::NTuple{11,Float64}
    K1::Float64
    K2::Float64
    S1::Float64
    S2::Float64
    NSIA::Float64
    NSIB::Float64
    CWA::Float64
    CWB::Float64
    M1A::Float64
    M2A::Float64
    M1B::Float64
    M2B::Float64
    N1A::Float64
    N2A::Float64
    N1B::Float64
    N2B::Float64
    DOMAIN_ID::Int64
end

struct _MSCProdRec
    PID::Int64
    MID::Int64
    A::Float64
    J::Float64
    C::Float64
    NSM::Float64
    DOMAIN_ID::Int64
end

struct _MSCPElasRec
    PID::Int64
    K::Float64
    GE::Float64
    S::Float64
    DOMAIN_ID::Int64
end

struct _MSCPMassRec
    PID::Int64
    M::Float64
    DOMAIN_ID::Int64
end

struct _MSCPSolidRec
    PID::Int64
    MID::Int64
    CORDM::Int64
    IN::Int64
    STRESS::Int64
    ISOP::Int64
    FCTN::NTuple{4,UInt8}
    DOMAIN_ID::Int64
end

struct _MSCPCOMPIdentityRec
    PID::Int64
    NPLIES::Int64
    Z0::Float64
    NSM::Float64
    SB::Float64
    FT::Int64
    TREF::Float64
    GE::Float64
    PLY_POS::Int64
    PLY_LEN::Int64
    DOMAIN_ID::Int64
end

struct _MSCPCOMPPlyRec
    MID::Int64
    T::Float64
    THETA::Float64
    SOUT::Int64
end

struct _MSCSpc1IdentityRec
    SID::Int64
    C::Int64
    G_POS::Int64
    G_LEN::Int64
    DOMAIN_ID::Int64
end

struct _MSCRbe2Rec
    EID::Int64
    GN::Int64
    CM::Int64
    GM_POS::Int64
    GM_LEN::Int64
    ALPHA::Float64
    TREF::Float64
    DOMAIN_ID::Int64
end

struct _MSCRbe3IdentityRec
    EID::Int64
    REFG::Int64
    REFC::Int64
    WTCG_POS::Int64
    WTCG_LEN::Int64
    GM_POS::Int64
    GM_LEN::Int64
    ALPHA::Float64
    TREF::Float64
    DOMAIN_ID::Int64
end

struct _MSCRbe3WtcgRec
    WT1::Float64
    C::Int64
    G_POS::Int64
    G_LEN::Int64
end

struct _MSCEigrlRec
    SID::Int64
    V1::Float64
    V2::Float64
    ND::Int64
    MSGLVL::Int64
    MAXSET::Int64
    SHFSCL::Float64
    FLAG1::Int64
    FLAG2::Int64
    NORM::NTuple{8,UInt8}
    ALPH::Float64
    FREQS_POS::Int64
    FREQS_LEN::Int64
    DOMAIN_ID::Int64
end

struct _MSCParamIntRec
    NAME::NTuple{8,UInt8}
    VALUE::Int64
    DOMAIN_ID::Int64
end

struct _MSCParamCharRec
    NAME::NTuple{8,UInt8}
    VALUE::NTuple{8,UInt8}
    DOMAIN_ID::Int64
end

@inline _msc_i64(x, default::Integer=0) = try
    x === nothing ? Int64(default) : Int64(round(Float64(x)))
catch
    parsed = tryparse(Int, string(x))
    parsed === nothing ? Int64(default) : Int64(parsed)
end

@inline _msc_f64(x, default::Real=0.0) = try
    x === nothing ? Float64(default) : Float64(x)
catch
    parsed = tryparse(Float64, string(x))
    parsed === nothing ? Float64(default) : parsed
end

@inline _msc_tuple_f64(values, n::Int) =
    ntuple(i -> i <= length(values) ? _msc_f64(values[i]) : 0.0, n)

@inline _msc_tuple_i64(values, n::Int) =
    ntuple(i -> i <= length(values) ? _msc_i64(values[i]) : 0, n)

function _msc_require_group(parent, path::AbstractString)
    clean = strip(String(path), '/')
    isempty(clean) && return parent
    group = parent
    for part in split(clean, '/')
        group = haskey(group, part) ? group[part] : create_group(group, part)
    end
    return group
end

function _msc_parent_and_name(parent, path::AbstractString)
    clean = strip(String(path), '/')
    parts = split(clean, '/')
    name = parts[end]
    group_path = length(parts) == 1 ? "" : join(parts[1:end-1], "/")
    return _msc_require_group(parent, group_path), name
end

function _msc_dataset_version(path::AbstractString, version)
    version !== nothing && return version
    return startswith(String(path), "/NASTRAN/") ? 0 : nothing
end

function _msc_write_records(parent, path::AbstractString, records::AbstractVector; version=nothing)
    isempty(records) && return
    group, name = _msc_parent_and_name(parent, path)
    write(group, name, records)
    dataset_version = _msc_dataset_version(path, version)
    dataset_version === nothing || (attributes(group[name])["version"] = Int64[dataset_version])
end

@inline function _msc_array_type(::Type{T}, n::Integer) where {T}
    dims = HDF5.API.hsize_t[n]
    return HDF5.API.h5t_array_create(HDF5.hdf5_type_id(T), Cuint(1), dims)
end

@inline function _msc_fixed_string_bytes(s, n::Integer=8)
    text = rpad(string(s), n)[1:n]
    bytes = codeunits(text)
    return ntuple(i -> UInt8(bytes[i]), n)
end

function _msc_fixed_string_type(n::Integer)
    string_type = HDF5.API.h5t_copy(HDF5.API.H5T_C_S1)
    HDF5.API.h5t_set_size(string_type, n)
    HDF5.API.h5t_set_strpad(string_type, HDF5.API.H5T_STR_SPACEPAD)
    return string_type
end

@inline function _msc_insert_scalar!(dtype, ::Type{Rec}, field::Symbol, ::Type{T}) where {Rec,T}
    idx = findfirst(==(field), fieldnames(Rec))
    HDF5.API.h5t_insert(dtype, String(field), fieldoffset(Rec, idx), HDF5.hdf5_type_id(T))
end

@inline function _msc_insert_array!(dtype, ::Type{Rec}, field::Symbol, ::Type{T}, n::Integer) where {Rec,T}
    idx = findfirst(==(field), fieldnames(Rec))
    array_type = _msc_array_type(T, n)
    try
        HDF5.API.h5t_insert(dtype, String(field), fieldoffset(Rec, idx), array_type)
    finally
        HDF5.API.h5t_close(array_type)
    end
end

@inline function _msc_insert_fixed_string!(dtype, ::Type{Rec}, field::Symbol, n::Integer) where {Rec}
    idx = findfirst(==(field), fieldnames(Rec))
    string_type = _msc_fixed_string_type(n)
    try
        HDF5.API.h5t_insert(dtype, String(field), fieldoffset(Rec, idx), string_type)
    finally
        HDF5.API.h5t_close(string_type)
    end
end

function _msc_write_custom(parent, path::AbstractString, records::Vector{T}, dtype_builder; version=nothing) where {T}
    isempty(records) && return
    group, name = _msc_parent_and_name(parent, path)
    dtype = dtype_builder()
    dataset = create_dataset(group, name, dtype, (length(records),))
    try
        HDF5.write_dataset(dataset, dtype, records)
        dataset_version = _msc_dataset_version(path, version)
        dataset_version === nothing || (attributes(dataset)["version"] = Int64[dataset_version])
    finally
        close(dataset)
        close(dtype)
    end
end

function _msc_dynamic_record_size(fields)
    size = 0
    for (_, kind, n) in fields
        if kind == :i64 || kind == :f64
            size += 8
        elseif kind == :str
            size += n
        elseif kind == :i64_array || kind == :f64_array
            size += 8 * n
        else
            error("Unsupported MSC HDF5 dynamic field kind: $kind")
        end
    end
    return size
end

function _msc_dynamic_dtype(fields, record_size::Integer)
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, record_size)
    offset = 0
    for (field, kind, n) in fields
        name = String(field)
        if kind == :i64
            HDF5.API.h5t_insert(dtype, name, offset, HDF5.hdf5_type_id(Int64))
            offset += 8
        elseif kind == :f64
            HDF5.API.h5t_insert(dtype, name, offset, HDF5.hdf5_type_id(Float64))
            offset += 8
        elseif kind == :str
            string_type = _msc_fixed_string_type(n)
            try
                HDF5.API.h5t_insert(dtype, name, offset, string_type)
            finally
                HDF5.API.h5t_close(string_type)
            end
            offset += n
        elseif kind == :i64_array
            array_type = _msc_array_type(Int64, n)
            try
                HDF5.API.h5t_insert(dtype, name, offset, array_type)
            finally
                HDF5.API.h5t_close(array_type)
            end
            offset += 8 * n
        elseif kind == :f64_array
            array_type = _msc_array_type(Float64, n)
            try
                HDF5.API.h5t_insert(dtype, name, offset, array_type)
            finally
                HDF5.API.h5t_close(array_type)
            end
            offset += 8 * n
        end
    end
    return dtype
end

function _msc_put_i64!(buffer::Vector{UInt8}, offset::Integer, value)
    bytes = reinterpret(UInt8, Int64[_msc_i64(value)])
    copyto!(buffer, offset + 1, bytes, 1, 8)
end

function _msc_put_f64!(buffer::Vector{UInt8}, offset::Integer, value)
    bytes = reinterpret(UInt8, Float64[_msc_f64(value)])
    copyto!(buffer, offset + 1, bytes, 1, 8)
end

function _msc_put_string!(buffer::Vector{UInt8}, offset::Integer, value, n::Integer)
    bytes = _msc_fixed_string_bytes(value, n)
    for i in 1:n
        buffer[offset + i] = bytes[i]
    end
end

function _msc_put_i64_array!(buffer::Vector{UInt8}, offset::Integer, values, n::Integer)
    for i in 1:n
        value = i <= length(values) ? values[i] : 0
        _msc_put_i64!(buffer, offset + 8 * (i - 1), value)
    end
end

function _msc_put_f64_array!(buffer::Vector{UInt8}, offset::Integer, values, n::Integer)
    for i in 1:n
        value = i <= length(values) ? values[i] : 0.0
        _msc_put_f64!(buffer, offset + 8 * (i - 1), value)
    end
end

function _msc_default_dynamic_value(kind::Symbol, n::Integer)
    kind == :i64 && return Int64(0)
    kind == :f64 && return 0.0
    kind == :str && return ""
    kind == :i64_array && return zeros(Int64, n)
    kind == :f64_array && return zeros(Float64, n)
    error("Unsupported MSC HDF5 dynamic field kind: $kind")
end

function _msc_write_dynamic_records(parent, path::AbstractString, fields, rows::AbstractVector{<:AbstractDict}; version=nothing)
    isempty(rows) && return
    record_size = _msc_dynamic_record_size(fields)
    dtype_id = _msc_dynamic_dtype(fields, record_size)
    dtype = HDF5.Datatype(dtype_id, false)
    group, name = _msc_parent_and_name(parent, path)
    dataset = create_dataset(group, name, dtype, (length(rows),))
    buffer = zeros(UInt8, record_size * length(rows))
    try
        for (row_index, row) in enumerate(rows)
            base = (row_index - 1) * record_size
            offset = 0
            for (field, kind, n) in fields
                value = get(row, field, _msc_default_dynamic_value(kind, n))
                if kind == :i64
                    _msc_put_i64!(buffer, base + offset, value)
                    offset += 8
                elseif kind == :f64
                    _msc_put_f64!(buffer, base + offset, value)
                    offset += 8
                elseif kind == :str
                    _msc_put_string!(buffer, base + offset, value, n)
                    offset += n
                elseif kind == :i64_array
                    _msc_put_i64_array!(buffer, base + offset, value, n)
                    offset += 8 * n
                elseif kind == :f64_array
                    _msc_put_f64_array!(buffer, base + offset, value, n)
                    offset += 8 * n
                end
            end
        end
        HDF5.API.h5d_write(dataset.id, dtype_id, HDF5.API.H5S_ALL, HDF5.API.H5S_ALL, HDF5.API.H5P_DEFAULT, pointer(buffer))
        dataset_version = _msc_dataset_version(path, version)
        dataset_version === nothing || (attributes(dataset)["version"] = Int64[dataset_version])
    finally
        close(dataset)
        close(dtype)
    end
end

function _msc_grid_dtype()
    Rec = _MSCGridRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :ID, Int64)
    _msc_insert_scalar!(dtype, Rec, :CP, Int64)
    _msc_insert_array!(dtype, Rec, :X, Float64, 3)
    _msc_insert_scalar!(dtype, Rec, :CD, Int64)
    _msc_insert_scalar!(dtype, Rec, :PS, Int64)
    _msc_insert_scalar!(dtype, Rec, :SEID, Int64)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_grid_weight_dtype()
    Rec = _MSCGridWeightRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :ID, Int64)
    _msc_insert_array!(dtype, Rec, :MO, Float64, 36)
    _msc_insert_array!(dtype, Rec, :S, Float64, 9)
    for field in (:MX, :XX, :YX, :ZX, :MY, :XY, :YY, :ZY, :MZ, :XZ, :YZ, :ZZ)
        _msc_insert_scalar!(dtype, Rec, field, Float64)
    end
    _msc_insert_array!(dtype, Rec, :I, Float64, 9)
    for field in (:PIX, :PIY, :PIZ)
        _msc_insert_scalar!(dtype, Rec, field, Float64)
    end
    _msc_insert_array!(dtype, Rec, :Q, Float64, 9)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_cquad4_dtype()
    Rec = _MSCCquad4Rec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :EID, Int64)
    _msc_insert_scalar!(dtype, Rec, :PID, Int64)
    _msc_insert_array!(dtype, Rec, :G, Int64, 4)
    _msc_insert_scalar!(dtype, Rec, :THETA, Float64)
    _msc_insert_scalar!(dtype, Rec, :ZOFFS, Float64)
    _msc_insert_scalar!(dtype, Rec, :TFLAG, Int64)
    _msc_insert_array!(dtype, Rec, :T, Float64, 4)
    _msc_insert_scalar!(dtype, Rec, :MCID, Int64)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_ctria3_dtype()
    Rec = _MSCCTRIA3Rec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :EID, Int64)
    _msc_insert_scalar!(dtype, Rec, :PID, Int64)
    _msc_insert_array!(dtype, Rec, :G, Int64, 3)
    _msc_insert_scalar!(dtype, Rec, :THETA, Float64)
    _msc_insert_scalar!(dtype, Rec, :ZOFFS, Float64)
    _msc_insert_scalar!(dtype, Rec, :TFLAG, Int64)
    _msc_insert_array!(dtype, Rec, :T, Float64, 3)
    _msc_insert_scalar!(dtype, Rec, :MCID, Int64)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_crod_dtype()
    Rec = _MSCCrodRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :EID, Int64)
    _msc_insert_scalar!(dtype, Rec, :PID, Int64)
    _msc_insert_array!(dtype, Rec, :G, Int64, 2)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_cbeam_dtype()
    Rec = _MSCCbeamRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    for field in (:EID, :PID, :GA, :GB, :SA, :SB)
        _msc_insert_scalar!(dtype, Rec, field, Int64)
    end
    _msc_insert_array!(dtype, Rec, :X, Float64, 3)
    for field in (:G0, :F, :PA, :PB)
        _msc_insert_scalar!(dtype, Rec, field, Int64)
    end
    _msc_insert_array!(dtype, Rec, :WA, Float64, 3)
    _msc_insert_array!(dtype, Rec, :WB, Float64, 3)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_solid_dtype(::Type{Rec}, n::Integer) where {Rec}
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :EID, Int64)
    _msc_insert_scalar!(dtype, Rec, :PID, Int64)
    _msc_insert_array!(dtype, Rec, :G, Int64, n)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

_msc_chexa_dtype() = _msc_solid_dtype(_MSCChexaRec, 20)
_msc_cpenta_dtype() = _msc_solid_dtype(_MSCCpentaRec, 15)
_msc_ctetra_dtype() = _msc_solid_dtype(_MSCCTetraRec, 10)

function _msc_conm2_dtype()
    Rec = _MSCConm2Rec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    for field in (:EID, :G, :CID)
        _msc_insert_scalar!(dtype, Rec, field, Int64)
    end
    for field in (:M, :X1, :X2, :X3, :I1)
        _msc_insert_scalar!(dtype, Rec, field, Float64)
    end
    _msc_insert_array!(dtype, Rec, :I2, Float64, 2)
    _msc_insert_array!(dtype, Rec, :I3, Float64, 3)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_force_dtype(::Type{Rec}, force_field::Symbol) where {Rec}
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :SID, Int64)
    _msc_insert_scalar!(dtype, Rec, :G, Int64)
    _msc_insert_scalar!(dtype, Rec, :CID, Int64)
    _msc_insert_scalar!(dtype, Rec, force_field, Float64)
    _msc_insert_array!(dtype, Rec, :N, Float64, 3)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

_msc_force_card_dtype() = _msc_force_dtype(_MSCForceRec, :F)
_msc_moment_card_dtype() = _msc_force_dtype(_MSCMomentRec, :M)

function _msc_pload4_dtype()
    Rec = _MSCPload4Rec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :SID, Int64)
    _msc_insert_scalar!(dtype, Rec, :EID, Int64)
    _msc_insert_array!(dtype, Rec, :P, Float64, 4)
    _msc_insert_scalar!(dtype, Rec, :G1, Int64)
    _msc_insert_scalar!(dtype, Rec, :G34, Int64)
    _msc_insert_scalar!(dtype, Rec, :CID, Int64)
    _msc_insert_array!(dtype, Rec, :N, Float64, 3)
    _msc_insert_fixed_string!(dtype, Rec, :SORL, 8)
    _msc_insert_fixed_string!(dtype, Rec, :LDIR, 8)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_param_int_dtype()
    Rec = _MSCParamIntRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_fixed_string!(dtype, Rec, :NAME, 8)
    _msc_insert_scalar!(dtype, Rec, :VALUE, Int64)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_param_char_dtype()
    Rec = _MSCParamCharRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_fixed_string!(dtype, Rec, :NAME, 8)
    _msc_insert_fixed_string!(dtype, Rec, :VALUE, 8)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_psolid_dtype()
    Rec = _MSCPSolidRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    for field in (:PID, :MID, :CORDM, :IN, :STRESS, :ISOP)
        _msc_insert_scalar!(dtype, Rec, field, Int64)
    end
    _msc_insert_fixed_string!(dtype, Rec, :FCTN, 4)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_eigrl_dtype()
    Rec = _MSCEigrlRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    _msc_insert_scalar!(dtype, Rec, :SID, Int64)
    _msc_insert_scalar!(dtype, Rec, :V1, Float64)
    _msc_insert_scalar!(dtype, Rec, :V2, Float64)
    for field in (:ND, :MSGLVL, :MAXSET)
        _msc_insert_scalar!(dtype, Rec, field, Int64)
    end
    _msc_insert_scalar!(dtype, Rec, :SHFSCL, Float64)
    _msc_insert_scalar!(dtype, Rec, :FLAG1, Int64)
    _msc_insert_scalar!(dtype, Rec, :FLAG2, Int64)
    _msc_insert_fixed_string!(dtype, Rec, :NORM, 8)
    _msc_insert_scalar!(dtype, Rec, :ALPH, Float64)
    _msc_insert_scalar!(dtype, Rec, :FREQS_POS, Int64)
    _msc_insert_scalar!(dtype, Rec, :FREQS_LEN, Int64)
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_pbeam_dtype()
    Rec = _MSCPBeamRec
    dtype = HDF5.API.h5t_create(HDF5.API.H5T_COMPOUND, sizeof(Rec))
    for field in (:PID, :MID, :NSEGS, :CCF, :CWELD)
        _msc_insert_scalar!(dtype, Rec, field, Int64)
    end
    for field in (:SO, :XXB, :A, :I1, :I2, :I12, :J, :NSM, :C1, :C2, :D1, :D2, :E1, :E2, :F1, :F2)
        _msc_insert_array!(dtype, Rec, field, Float64, 11)
    end
    for field in (:K1, :K2, :S1, :S2, :NSIA, :NSIB, :CWA, :CWB, :M1A, :M2A, :M1B, :M2B, :N1A, :N2A, :N1B, :N2B)
        _msc_insert_scalar!(dtype, Rec, field, Float64)
    end
    _msc_insert_scalar!(dtype, Rec, :DOMAIN_ID, Int64)
    return HDF5.Datatype(dtype, true)
end

function _msc_model_value_by_id(collection, id)
    sid = string(id)
    haskey(collection, sid) && return collection[sid]
    haskey(collection, id) && return collection[id]
    return nothing
end

_msc_node_order(id_map) = sort(collect(keys(id_map)))

function _msc_input_grid_records(model)
    records = _MSCGridRec[]
    for grid in sort(collect(values(get(model, "GRIDs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        push!(records, _MSCGridRec(
            _msc_i64(get(grid, "ID", 0)), _msc_i64(get(grid, "CP", 0)),
            _msc_tuple_f64(get(grid, "X", [0.0, 0.0, 0.0]), 3),
            _msc_i64(get(grid, "CD", 0)), _msc_i64(get(grid, "PS", 0)),
            _msc_i64(get(grid, "SEID", 0)), 1,
        ))
    end
    return records
end

function _msc_input_shell_records(model)
    quads = _MSCCquad4Rec[]
    trias = _MSCCTRIA3Rec[]
    pshells = get(model, "PSHELLs", Dict())
    for element in sort(collect(values(get(model, "CSHELLs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        node_ids = get(element, "NODES", Int[])
        pid = _msc_i64(get(element, "PID", 0))
        prop = _msc_model_value_by_id(pshells, pid)
        thickness = _msc_f64(prop === nothing ? 0.0 : get(prop, "T", 0.0))
        theta = _msc_f64(get(element, "THETA", 0.0))
        mcid = _msc_i64(get(element, "MCID", 0))
        eid = _msc_i64(get(element, "ID", 0))
        if length(node_ids) == 4
            push!(quads, _MSCCquad4Rec(eid, pid, _msc_tuple_i64(node_ids, 4), theta, 0.0, 0, (thickness, thickness, thickness, thickness), mcid, 1))
        elseif length(node_ids) == 3
            push!(trias, _MSCCTRIA3Rec(eid, pid, _msc_tuple_i64(node_ids, 3), theta, 0.0, 0, (thickness, thickness, thickness), mcid, 1))
        end
    end
    return quads, trias
end

function _msc_input_rod_records(model)
    rods = _MSCCrodRec[]
    for rod in sort(collect(values(get(model, "CRODs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        push!(rods, _MSCCrodRec(_msc_i64(get(rod, "ID", 0)), _msc_i64(get(rod, "PID", 0)),
            (_msc_i64(get(rod, "GA", 0)), _msc_i64(get(rod, "GB", 0))), 1))
    end
    return rods
end

function _msc_input_bar_records(model)
    bars = _MSCCbarRec[]
    for bar in sort(collect(values(get(model, "CBARs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        v = _msc_tuple_f64(get(bar, "V", [0.0, 0.0, 1.0]), 3)
        wa = _msc_tuple_f64(get(bar, "WA", [0.0, 0.0, 0.0]), 3)
        wb = _msc_tuple_f64(get(bar, "WB", [0.0, 0.0, 0.0]), 3)
        push!(bars, _MSCCbarRec(
            _msc_i64(get(bar, "ID", 0)), _msc_i64(get(bar, "PID", 0)),
            _msc_i64(get(bar, "GA", 0)), _msc_i64(get(bar, "GB", 0)),
            0, v[1], v[2], v[3], _msc_i64(get(bar, "G0", 0)),
            _msc_i64(get(bar, "PA", 0)), _msc_i64(get(bar, "PB", 0)),
            wa[1], wa[2], wa[3], wb[1], wb[2], wb[3], 1,
        ))
    end
    return bars
end

function _msc_input_beam_records(model)
    beams = _MSCCbeamRec[]
    for beam in sort(collect(values(get(model, "CBEAMs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        push!(beams, _MSCCbeamRec(
            _msc_i64(get(beam, "ID", 0)), _msc_i64(get(beam, "PID", 0)),
            _msc_i64(get(beam, "GA", 0)), _msc_i64(get(beam, "GB", 0)),
            _msc_i64(get(beam, "SA", 0)), _msc_i64(get(beam, "SB", 0)),
            _msc_tuple_f64(get(beam, "V", [0.0, 0.0, 1.0]), 3),
            _msc_i64(get(beam, "G0", 0)), 1,
            _msc_i64(get(beam, "PA", 0)), _msc_i64(get(beam, "PB", 0)),
            _msc_tuple_f64(get(beam, "WA", [0.0, 0.0, 0.0]), 3),
            _msc_tuple_f64(get(beam, "WB", [0.0, 0.0, 0.0]), 3),
            1,
        ))
    end
    return beams
end

function _msc_input_conrod_records(model)
    rods = _MSCConrodRec[]
    for rod in sort(collect(values(get(model, "CONRODs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        push!(rods, _MSCConrodRec(
            _msc_i64(get(rod, "ID", 0)), _msc_i64(get(rod, "GA", 0)),
            _msc_i64(get(rod, "GB", 0)), _msc_i64(get(rod, "MID", 0)),
            _msc_f64(get(rod, "A", 0.0)), _msc_f64(get(rod, "J", 0.0)),
            _msc_f64(get(rod, "C", 0.0)), _msc_f64(get(rod, "NSM", 0.0)), 1,
        ))
    end
    return rods
end

function _msc_input_solid_records(model)
    hexas = _MSCChexaRec[]
    pentas = _MSCCpentaRec[]
    tetras = _MSCCTetraRec[]
    for solid in sort(collect(values(get(model, "CSOLIDs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        etype = uppercase(string(get(solid, "TYPE", "")))
        nodes = get(solid, "NODES", Int[])
        eid = _msc_i64(get(solid, "ID", 0))
        pid = _msc_i64(get(solid, "PID", 0))
        if etype == "CHEXA"
            push!(hexas, _MSCChexaRec(eid, pid, _msc_tuple_i64(nodes, 20), 1))
        elseif etype == "CPENTA"
            push!(pentas, _MSCCpentaRec(eid, pid, _msc_tuple_i64(nodes, 15), 1))
        elseif etype == "CTETRA"
            push!(tetras, _MSCCTetraRec(eid, pid, _msc_tuple_i64(nodes, 10), 1))
        end
    end
    return hexas, pentas, tetras
end

function _msc_input_spring_records(model)
    celas1 = _MSCCelas1Rec[]
    celas2 = _MSCCelas2Rec[]
    for spring in sort(collect(values(get(model, "CELASs", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        etype = uppercase(string(get(spring, "TYPE", "")))
        if etype == "CELAS1"
            push!(celas1, _MSCCelas1Rec(
                _msc_i64(get(spring, "ID", 0)), _msc_i64(get(spring, "PID", 0)),
                _msc_i64(get(spring, "G1", 0)), _msc_i64(get(spring, "G2", 0)),
                _msc_i64(get(spring, "C1", 0)), _msc_i64(get(spring, "C2", 0)), 1,
            ))
        elseif etype == "CELAS2"
            push!(celas2, _MSCCelas2Rec(
                _msc_i64(get(spring, "ID", 0)), _msc_f64(get(spring, "K", 0.0)),
                _msc_i64(get(spring, "G1", 0)), _msc_i64(get(spring, "G2", 0)),
                _msc_i64(get(spring, "C1", 0)), _msc_i64(get(spring, "C2", 0)),
                _msc_f64(get(spring, "GE", 0.0)), _msc_f64(get(spring, "S", 0.0)), 1,
            ))
        end
    end
    return celas1, celas2
end

function _msc_input_mass_records(model)
    cmass1 = _MSCCmass1Rec[]
    cmass2 = _MSCCmass2Rec[]
    conm2 = _MSCConm2Rec[]
    for mass in sort(collect(values(get(model, "CMASS1s", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        push!(cmass1, _MSCCmass1Rec(
            _msc_i64(get(mass, "ID", 0)), _msc_i64(get(mass, "PID", 0)),
            _msc_i64(get(mass, "G1", 0)), _msc_i64(get(mass, "G2", 0)),
            _msc_i64(get(mass, "C1", 0)), _msc_i64(get(mass, "C2", 0)), 1,
        ))
    end
    for mass in sort(collect(values(get(model, "CMASS2s", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        push!(cmass2, _MSCCmass2Rec(
            _msc_i64(get(mass, "ID", 0)), _msc_f64(get(mass, "M", 0.0)),
            _msc_i64(get(mass, "G1", 0)), _msc_i64(get(mass, "G2", 0)),
            _msc_i64(get(mass, "C1", 0)), _msc_i64(get(mass, "C2", 0)), 1,
        ))
    end
    for mass in sort(collect(values(get(model, "CONM2s", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        x = _msc_tuple_f64(get(mass, "X", [0.0, 0.0, 0.0]), 3)
        inertia = _msc_tuple_f64(get(mass, "I", [0.0, 0.0, 0.0]), 3)
        push!(conm2, _MSCConm2Rec(
            _msc_i64(get(mass, "ID", 0)), _msc_i64(get(mass, "GID", get(mass, "G", 0))),
            _msc_i64(get(mass, "CID", 0)), _msc_f64(get(mass, "M", 0.0)),
            x[1], x[2], x[3], inertia[1], (inertia[2], 0.0), (inertia[3], 0.0, 0.0), 1,
        ))
    end
    return cmass1, cmass2, conm2
end

function _msc_input_material_records(model)
    mat1 = _MSCMat1Rec[]
    mat2 = _MSCMat2Rec[]
    mat8 = _MSCMat8Rec[]
    for mat in sort(collect(values(get(model, "MATs", Dict()))); by=x -> _msc_i64(get(x, "MID", 0)))
        type = uppercase(string(get(mat, "TYPE", "MAT1")))
        if type == "MAT8"
            push!(mat8, _MSCMat8Rec(
                _msc_i64(get(mat, "MID", 0)), _msc_f64(get(mat, "E1", get(mat, "E", 0.0))),
                _msc_f64(get(mat, "E2", get(mat, "E", 0.0))), _msc_f64(get(mat, "NU12", get(mat, "NU", 0.0))),
                _msc_f64(get(mat, "G12", get(mat, "G", 0.0))), _msc_f64(get(mat, "G1Z", 0.0)),
                _msc_f64(get(mat, "G2Z", 0.0)), _msc_f64(get(mat, "RHO", 0.0)),
                _msc_f64(get(mat, "A1", 0.0)), _msc_f64(get(mat, "A2", 0.0)),
                _msc_f64(get(mat, "TREF", 0.0)), _msc_f64(get(mat, "XT", 0.0)),
                _msc_f64(get(mat, "XC", 0.0)), _msc_f64(get(mat, "YT", 0.0)),
                _msc_f64(get(mat, "YC", 0.0)), _msc_f64(get(mat, "S", 0.0)),
                _msc_f64(get(mat, "GE", 0.0)), _msc_f64(get(mat, "F12", 0.0)),
                _msc_f64(get(mat, "STRN", 0.0)), 1,
            ))
        elseif type != "MAT1_EQUIV"
            push!(mat1, _MSCMat1Rec(
                _msc_i64(get(mat, "MID", 0)), _msc_f64(get(mat, "E", 0.0)),
                _msc_f64(get(mat, "G", 0.0)), _msc_f64(get(mat, "NU", 0.0)),
                _msc_f64(get(mat, "RHO", 0.0)), _msc_f64(get(mat, "ALPHA", get(mat, "A", 0.0))),
                _msc_f64(get(mat, "TREF", 0.0)), _msc_f64(get(mat, "GE", 0.0)),
                _msc_f64(get(mat, "ST", 0.0)), _msc_f64(get(mat, "SC", 0.0)),
                _msc_f64(get(mat, "SS", 0.0)), _msc_i64(get(mat, "MCSID", 0)), 1,
            ))
        end
    end

    for prop in sort(collect(values(get(model, "PSHELLs", Dict()))); by=x -> _msc_i64(get(x, "PID", 0)))
        uppercase(string(get(prop, "TYPE", ""))) == "PCOMP_CLT" || continue
        pid = _msc_i64(get(prop, "PID", 0))
        rho = 0.0
        plies = get(prop, "PLY_DATA", Any[])
        if !isempty(plies)
            first_mid = _msc_i64(get(plies[1], "MID", get(plies[1], "mid", 0)))
            mat = _msc_model_value_by_id(get(model, "MATs", Dict()), first_mid)
            rho = mat === nothing ? 0.0 : _msc_f64(get(mat, "RHO", 0.0))
        end
        cm = get(prop, "Cm", zeros(3, 3))
        cb = get(prop, "Cb", zeros(3, 3))
        cs = get(prop, "Cs", zeros(2, 2))
        bmb = get(prop, "Bmb", zeros(3, 3))
        rows = (
            (100000000 + pid, cm),
            (200000000 + pid, cb),
            (300000000 + pid, cs),
            (400000000 + pid, bmb),
        )
        for (mid, m) in rows
            push!(mat2, _MSCMat2Rec(
                mid, _msc_f64(m[1, 1]), _msc_f64(size(m, 2) >= 2 ? m[1, 2] : 0.0),
                _msc_f64(size(m, 2) >= 3 ? m[1, 3] : 0.0), _msc_f64(size(m, 1) >= 2 && size(m, 2) >= 2 ? m[2, 2] : 0.0),
                _msc_f64(size(m, 1) >= 2 && size(m, 2) >= 3 ? m[2, 3] : 0.0), _msc_f64(size(m, 1) >= 3 && size(m, 2) >= 3 ? m[3, 3] : 0.0),
                rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1,
            ))
        end
    end
    return mat1, mat2, mat8
end

_msc_beam_vec(value) = ntuple(i -> (i == 1 || i == 11) ? _msc_f64(value) : 0.0, 11)
_msc_beam_so_vec() = ntuple(i -> (i == 1 || i == 11) ? 1.0 : 0.0, 11)
_msc_beam_xxb_vec() = ntuple(i -> i == 11 ? 1.0 : 0.0, 11)

function _msc_input_property_records(model)
    pshell = _MSCPShellRec[]
    pbar = _MSCPBarRec[]
    pbeam = _MSCPBeamRec[]
    prod = _MSCProdRec[]
    pelas = _MSCPElasRec[]
    pmass = _MSCPMassRec[]
    psolid = _MSCPSolidRec[]
    pcomp_identity = _MSCPCOMPIdentityRec[]
    pcomp_ply = _MSCPCOMPPlyRec[]
    ply_pos = 0

    for prop in sort(collect(values(get(model, "PSHELLs", Dict()))); by=x -> _msc_i64(get(x, "PID", 0)))
        pid = _msc_i64(get(prop, "PID", 0))
        type = uppercase(string(get(prop, "TYPE", "PSHELL")))
        if type == "PCOMP_CLT"
            plies = get(prop, "PLY_DATA", Any[])
            push!(pcomp_identity, _MSCPCOMPIdentityRec(
                pid, length(plies), -0.5 * _msc_f64(get(prop, "T", 0.0)),
                _msc_f64(get(prop, "NSM", 0.0)), 0.0, 0, 0.0, 0.0, ply_pos, length(plies), 1,
            ))
            for ply in plies
                thickness = haskey(ply, "T") ? get(ply, "T", 0.0) : _msc_f64(get(ply, "z_top", 0.0)) - _msc_f64(get(ply, "z_bot", 0.0))
                sout = uppercase(strip(string(get(ply, "SOUT", get(ply, "sout", ""))))) == "YES" ? 1 : 0
                push!(pcomp_ply, _MSCPCOMPPlyRec(
                    _msc_i64(get(ply, "MID", get(ply, "mid", 0))), _msc_f64(thickness),
                    _msc_f64(get(ply, "THETA", get(ply, "theta", 0.0))), sout,
                ))
            end
            ply_pos += length(plies)
            push!(pshell, _MSCPShellRec(
                pid, 100000000 + pid, _msc_f64(get(prop, "T", 0.0)),
                200000000 + pid, 1.0, 300000000 + pid, 1.0,
                _msc_f64(get(prop, "NSM", 0.0)), -0.5 * _msc_f64(get(prop, "T", 0.0)),
                0.5 * _msc_f64(get(prop, "T", 0.0)), 400000000 + pid, 1,
            ))
        else
            mid = _msc_i64(get(prop, "MID", get(prop, "MID1", 0)))
            push!(pshell, _MSCPShellRec(
                pid, mid, _msc_f64(get(prop, "T", 0.0)),
                _msc_i64(get(prop, "MID2", mid)), _msc_f64(get(prop, "BEND_RATIO", get(prop, "BK", 1.0))),
                _msc_i64(get(prop, "MID3", mid)), _msc_f64(get(prop, "TS_T", get(prop, "TS", 0.8333333333333334))),
                _msc_f64(get(prop, "NSM", 0.0)), _msc_f64(get(prop, "Z1", 0.0)),
                _msc_f64(get(prop, "Z2", 0.0)), _msc_i64(get(prop, "MID4", 0)), 1,
            ))
        end
    end

    for prop in sort(collect(values(get(model, "PBARLs", Dict()))); by=x -> _msc_i64(get(x, "PID", 0)))
        type = uppercase(string(get(prop, "TYPE", "PBAR")))
        if type == "PBEAM"
            push!(pbeam, _MSCPBeamRec(
                _msc_i64(get(prop, "PID", 0)), _msc_i64(get(prop, "MID", 0)),
                1, 1, 0, _msc_beam_so_vec(), _msc_beam_xxb_vec(),
                _msc_beam_vec(get(prop, "A", 0.0)), _msc_beam_vec(get(prop, "I1", 0.0)),
                _msc_beam_vec(get(prop, "I2", 0.0)), _msc_beam_vec(get(prop, "I12", 0.0)),
                _msc_beam_vec(get(prop, "J", 0.0)), _msc_beam_vec(get(prop, "NSM", 0.0)),
                _msc_beam_vec(get(prop, "C1", 0.0)), _msc_beam_vec(get(prop, "C2", 0.0)),
                _msc_beam_vec(get(prop, "D1", 0.0)), _msc_beam_vec(get(prop, "D2", 0.0)),
                _msc_beam_vec(get(prop, "E1", 0.0)), _msc_beam_vec(get(prop, "E2", 0.0)),
                _msc_beam_vec(get(prop, "F1", 0.0)), _msc_beam_vec(get(prop, "F2", 0.0)),
                _msc_f64(get(prop, "K1", 1.0)), _msc_f64(get(prop, "K2", 1.0)),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1,
            ))
        else
            push!(pbar, _MSCPBarRec(
                _msc_i64(get(prop, "PID", 0)), _msc_i64(get(prop, "MID", 0)),
                _msc_f64(get(prop, "A", 0.0)), _msc_f64(get(prop, "I1", 0.0)),
                _msc_f64(get(prop, "I2", 0.0)), _msc_f64(get(prop, "J", 0.0)),
                _msc_f64(get(prop, "NSM", 0.0)), 0.0,
                _msc_f64(get(prop, "C1", 0.0)), _msc_f64(get(prop, "C2", 0.0)),
                _msc_f64(get(prop, "D1", 0.0)), _msc_f64(get(prop, "D2", 0.0)),
                _msc_f64(get(prop, "E1", 0.0)), _msc_f64(get(prop, "E2", 0.0)),
                _msc_f64(get(prop, "F1", 0.0)), _msc_f64(get(prop, "F2", 0.0)),
                _msc_f64(get(prop, "K1", 1.0)), _msc_f64(get(prop, "K2", 1.0)),
                _msc_f64(get(prop, "I12", 0.0)), 1,
            ))
        end
    end

    for prop in sort(collect(values(get(model, "PRODs", Dict()))); by=x -> _msc_i64(get(x, "PID", 0)))
        push!(prod, _MSCProdRec(_msc_i64(get(prop, "PID", 0)), _msc_i64(get(prop, "MID", 0)),
            _msc_f64(get(prop, "A", 0.0)), _msc_f64(get(prop, "J", 0.0)),
            _msc_f64(get(prop, "C", 0.0)), _msc_f64(get(prop, "NSM", 0.0)), 1))
    end
    for prop in sort(collect(values(get(model, "PELASs", Dict()))); by=x -> _msc_i64(get(x, "PID", 0)))
        push!(pelas, _MSCPElasRec(_msc_i64(get(prop, "PID", 0)), _msc_f64(get(prop, "K", 0.0)),
            _msc_f64(get(prop, "GE", 0.0)), _msc_f64(get(prop, "S", 0.0)), 1))
    end
    for prop in sort(collect(values(get(model, "PMASSs", Dict()))); by=x -> _msc_i64(get(x, "PID", 0)))
        push!(pmass, _MSCPMassRec(_msc_i64(get(prop, "PID", 0)), _msc_f64(get(prop, "M", 0.0)), 1))
    end
    for prop in sort(collect(values(get(model, "PSOLIDs", Dict()))); by=x -> _msc_i64(get(x, "PID", 0)))
        push!(psolid, _MSCPSolidRec(
            _msc_i64(get(prop, "PID", 0)), _msc_i64(get(prop, "MID", 0)),
            _msc_i64(get(prop, "CORDM", 0)), _msc_i64(get(prop, "IN", 0)),
            _msc_i64(get(prop, "STRESS", 0)), _msc_i64(get(prop, "ISOP", 0)),
            _msc_fixed_string_bytes(get(prop, "FCTN", "SMEC"), 4), 1,
        ))
    end
    return pshell, pbar, pbeam, prod, pelas, pmass, psolid, pcomp_identity, pcomp_ply
end

function _msc_input_load_records(model)
    forces = _MSCForceRec[]
    moments = _MSCMomentRec[]
    pload4s = _MSCPload4Rec[]
    for force in get(model, "FORCEs", Any[])
        push!(forces, _MSCForceRec(
            _msc_i64(get(force, "SID", 0)), _msc_i64(get(force, "GID", get(force, "G", 0))),
            _msc_i64(get(force, "CID", 0)), _msc_f64(get(force, "Mag", get(force, "F", 0.0))),
            _msc_tuple_f64(get(force, "Dir", get(force, "N", [0.0, 0.0, 0.0])), 3), 1,
        ))
    end
    for moment in get(model, "MOMENTs", Any[])
        push!(moments, _MSCMomentRec(
            _msc_i64(get(moment, "SID", 0)), _msc_i64(get(moment, "GID", get(moment, "G", 0))),
            _msc_i64(get(moment, "CID", 0)), _msc_f64(get(moment, "Mag", get(moment, "M", 0.0))),
            _msc_tuple_f64(get(moment, "Dir", get(moment, "N", [0.0, 0.0, 0.0])), 3), 1,
        ))
    end
    for load in get(model, "PLOAD4s", Any[])
        p = _msc_f64(get(load, "P", 0.0))
        pressures = get(load, "PVALS", [p, p, p, p])
        push!(pload4s, _MSCPload4Rec(
            _msc_i64(get(load, "SID", 0)), _msc_i64(get(load, "EID", 0)),
            _msc_tuple_f64(pressures, 4), _msc_i64(get(load, "G1", 0)),
            _msc_i64(get(load, "G34", 0)), _msc_i64(get(load, "CID", 0)),
            _msc_tuple_f64(get(load, "N", [0.0, 0.0, 0.0]), 3),
            _msc_fixed_string_bytes(get(load, "SORL", "SURF"), 8),
            _msc_fixed_string_bytes(get(load, "LDIR", "NORM"), 8), 1,
        ))
    end
    return sort!(forces; by=x -> (x.SID, x.G)),
        sort!(moments; by=x -> (x.SID, x.G)),
        sort!(pload4s; by=x -> (x.SID, x.EID))
end

function _msc_input_spc1_records(model)
    groups = Dict{Tuple{Int64,Int64}, Vector{Int64}}()
    for spc in get(model, "SPC1s", Any[])
        key = (_msc_i64(get(spc, "SID", 0)), _msc_i64(get(spc, "C", 0)))
        append!(get!(groups, key, Int64[]), _msc_i64.(get(spc, "NODES", Int[])))
    end
    identities = _MSCSpc1IdentityRec[]
    grids = _MSCIdRec[]
    pos = 0
    for key in sort(collect(keys(groups)))
        nodes = sort(unique(groups[key]))
        append!(grids, [_MSCIdRec(n) for n in nodes])
        push!(identities, _MSCSpc1IdentityRec(key[1], key[2], pos, length(nodes), 1))
        pos += length(nodes)
    end
    return identities, grids
end

function _msc_input_rbe_records(model)
    rbe2_rows = _MSCRbe2Rec[]
    rbe2_gm = _MSCIdRec[]
    gm_pos = 0
    for rbe in sort(collect(values(get(model, "RBE2s", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        gm = _msc_i64.(get(rbe, "GM", Int[]))
        append!(rbe2_gm, [_MSCIdRec(g) for g in gm])
        push!(rbe2_rows, _MSCRbe2Rec(_msc_i64(get(rbe, "ID", 0)), _msc_i64(get(rbe, "GN", 0)),
            _msc_i64(get(rbe, "CM", 0)), gm_pos, length(gm), 0.0, 0.0, 1))
        gm_pos += length(gm)
    end

    rbe3_identity = _MSCRbe3IdentityRec[]
    rbe3_wtcg = _MSCRbe3WtcgRec[]
    rbe3_g = _MSCIdRec[]
    wtcg_pos = 0
    g_pos = 0
    for rbe in sort(collect(values(get(model, "RBE3s", Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
        groups = get(rbe, "WT_GROUPS", Any[])
        start_wtcg = wtcg_pos
        start_g = g_pos
        for group in groups
            grids = _msc_i64.(collect(getproperty(group, :grids)))
            append!(rbe3_g, [_MSCIdRec(g) for g in grids])
            push!(rbe3_wtcg, _MSCRbe3WtcgRec(_msc_f64(getproperty(group, :wt)),
                _msc_i64(getproperty(group, :comps)), g_pos, length(grids)))
            g_pos += length(grids)
            wtcg_pos += 1
        end
        push!(rbe3_identity, _MSCRbe3IdentityRec(
            _msc_i64(get(rbe, "ID", 0)), _msc_i64(get(rbe, "REFGRID", 0)),
            _msc_i64(get(rbe, "REFC", 0)), start_wtcg, length(groups), start_g, g_pos - start_g, 0.0, 0.0, 1,
        ))
    end
    return rbe2_rows, rbe2_gm, rbe3_identity, rbe3_wtcg, rbe3_g
end

function _msc_input_eigrl_records(model)
    rows = _MSCEigrlRec[]
    for eig in sort(collect(values(get(model, "EIGRLs", Dict()))); by=x -> _msc_i64(get(x, "SID", 0)))
        push!(rows, _MSCEigrlRec(
            _msc_i64(get(eig, "SID", 0)), _msc_f64(get(eig, "V1", 0.0)),
            _msc_f64(get(eig, "V2", 0.0)), _msc_i64(get(eig, "ND", 0)),
            _msc_i64(get(eig, "MSGLVL", 0)), _msc_i64(get(eig, "MAXSET", 0)),
            _msc_f64(get(eig, "SHFSCL", 0.0)), 0, 0,
            _msc_fixed_string_bytes(get(eig, "NORM", "MASS"), 8), 0.0, 0, 0, 1,
        ))
    end
    return rows
end

function _msc_write_input_tables(file, model)
    _msc_write_records(file, "/NASTRAN/INPUT/DOMAINS", [_MSCInputDomainRec(1, 0, 0, 0, 0)])
    _msc_write_custom(file, "/NASTRAN/INPUT/NODE/GRID", _msc_input_grid_records(model), _msc_grid_dtype)
    quads, trias = _msc_input_shell_records(model)
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CQUAD4", quads, _msc_cquad4_dtype)
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CTRIA3", trias, _msc_ctria3_dtype)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/CBAR", _msc_input_bar_records(model))
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CBEAM", _msc_input_beam_records(model), _msc_cbeam_dtype)
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CROD", _msc_input_rod_records(model), _msc_crod_dtype)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/CONROD", _msc_input_conrod_records(model))
    hexas, pentas, tetras = _msc_input_solid_records(model)
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CHEXA", hexas, _msc_chexa_dtype)
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CPENTA", pentas, _msc_cpenta_dtype)
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CTETRA", tetras, _msc_ctetra_dtype)
    celas1, celas2 = _msc_input_spring_records(model)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/CELAS1", celas1)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/CELAS2", celas2)
    cmass1, cmass2, conm2 = _msc_input_mass_records(model)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/CMASS1", cmass1)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/CMASS2", cmass2)
    _msc_write_custom(file, "/NASTRAN/INPUT/ELEMENT/CONM2", conm2, _msc_conm2_dtype)

    mat1, mat2, mat8 = _msc_input_material_records(model)
    _msc_write_records(file, "/NASTRAN/INPUT/MATERIAL/MAT1", mat1)
    _msc_write_records(file, "/NASTRAN/INPUT/MATERIAL/MAT2", mat2)
    _msc_write_records(file, "/NASTRAN/INPUT/MATERIAL/MAT8", mat8)
    pshell, pbar, pbeam, prod, pelas, pmass, psolid, pcomp_identity, pcomp_ply = _msc_input_property_records(model)
    _msc_write_records(file, "/NASTRAN/INPUT/PROPERTY/PSHELL", pshell)
    _msc_write_records(file, "/NASTRAN/INPUT/PROPERTY/PBAR", pbar)
    _msc_write_custom(file, "/NASTRAN/INPUT/PROPERTY/PBEAM", pbeam, _msc_pbeam_dtype)
    _msc_write_records(file, "/NASTRAN/INPUT/PROPERTY/PROD", prod)
    _msc_write_records(file, "/NASTRAN/INPUT/PROPERTY/PELAS", pelas)
    _msc_write_records(file, "/NASTRAN/INPUT/PROPERTY/PMASS", pmass)
    _msc_write_custom(file, "/NASTRAN/INPUT/PROPERTY/PSOLID", psolid, _msc_psolid_dtype)
    _msc_write_records(file, "/NASTRAN/INPUT/PROPERTY/PCOMP/IDENTITY", pcomp_identity)
    _msc_write_records(file, "/NASTRAN/INPUT/PROPERTY/PCOMP/PLY", pcomp_ply)

    forces, moments, pload4s = _msc_input_load_records(model)
    _msc_write_custom(file, "/NASTRAN/INPUT/LOAD/FORCE", forces, _msc_force_card_dtype)
    _msc_write_custom(file, "/NASTRAN/INPUT/LOAD/MOMENT", moments, _msc_moment_card_dtype)
    _msc_write_custom(file, "/NASTRAN/INPUT/LOAD/PLOAD4", pload4s, _msc_pload4_dtype)
    spc1_identity, spc1_grids = _msc_input_spc1_records(model)
    _msc_write_records(file, "/NASTRAN/INPUT/CONSTRAINT/SPC1/SPC1_G/IDENTITY", spc1_identity)
    _msc_write_records(file, "/NASTRAN/INPUT/CONSTRAINT/SPC1/SPC1_G/G", spc1_grids)
    rbe2_rows, rbe2_gm, rbe3_identity, rbe3_wtcg, rbe3_g = _msc_input_rbe_records(model)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/RBE2/RB", rbe2_rows; version=20210)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/RBE2/GM", rbe2_gm)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/RBE3/IDENTITY", rbe3_identity; version=20210)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/RBE3/WTCG", rbe3_wtcg)
    _msc_write_records(file, "/NASTRAN/INPUT/ELEMENT/RBE3/G", rbe3_g)
    _msc_write_custom(file, "/NASTRAN/INPUT/DYNAMIC/EIGRL/IDENTITY", _msc_input_eigrl_records(model), _msc_eigrl_dtype)

    _msc_write_custom(file, "/NASTRAN/INPUT/PARAMETER/MDLPRM",
        [_MSCParamIntRec(_msc_fixed_string_bytes("HDF5", 8), 1, 1)], _msc_param_int_dtype)
    _msc_write_custom(file, "/NASTRAN/INPUT/PARAMETER/PVT/CHAR",
        [_MSCParamCharRec(_msc_fixed_string_bytes("AUTOSPC", 8), _msc_fixed_string_bytes(get(model, "PARAM_AUTOSPC", "YES"), 8), 1)], _msc_param_char_dtype)
    pvt_int = _MSCParamIntRec[]
    haskey(model, "PARAM_GRDPNT") && push!(pvt_int, _MSCParamIntRec(_msc_fixed_string_bytes("GRDPNT", 8), _msc_i64(get(model, "PARAM_GRDPNT", 0)), 1))
    haskey(model, "PARAM_POST") && push!(pvt_int, _MSCParamIntRec(_msc_fixed_string_bytes("POST", 8), _msc_i64(get(model, "PARAM_POST", 0)), 1))
    _msc_write_custom(file, "/NASTRAN/INPUT/PARAMETER/PVT/INT", pvt_int, _msc_param_int_dtype)
    _msc_write_casecc_subcase(file, model)
end

function _msc_nodal_rows_from_vector(id_map, vector, domain_id::Integer)
    rows = _MSCNodalResultRec[]
    for nid in _msc_node_order(id_map)
        idx = id_map[nid]
        base = (idx - 1) * 6
        vals = ntuple(i -> base + i <= length(vector) ? _msc_f64(vector[base + i]) : 0.0, 6)
        push!(rows, _MSCNodalResultRec(_msc_i64(nid), vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], _msc_i64(domain_id)))
    end
    return rows
end

function _msc_eigenvector_rows(id_map, mode_shapes, domain_start::Integer)
    rows = _MSCNodalResultRec[]
    index = _MSCIndexRec[]
    node_count = length(id_map)
    position = 0
    for mode in 1:size(mode_shapes, 2)
        domain_id = _msc_i64(domain_start + mode - 1)
        push!(index, _MSCIndexRec(domain_id, position, node_count))
        append!(rows, _msc_nodal_rows_from_vector(id_map, view(mode_shapes, :, mode), domain_id))
        position += node_count
    end
    return rows, index
end

function _msc_eigenvalue_rows(eigenvalues, domain_start::Integer; frequencies=nothing)
    rows = _MSCEigenvalueRec[]
    for i in eachindex(eigenvalues)
        eig = _msc_f64(eigenvalues[i])
        omega = sqrt(abs(eig))
        freq = frequencies === nothing || i > length(frequencies) ? omega / (2pi) : _msc_f64(frequencies[i])
        push!(rows, _MSCEigenvalueRec(_msc_i64(i), _msc_i64(i), eig, omega, freq, 1.0, eig, 0, 0, _msc_i64(domain_start + i - 1)))
    end
    return rows
end

function _msc_result_domains_static(subcases)
    rows = _MSCResultDomainRec[_MSCResultDomainRec(1, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for (i, subcase) in enumerate(subcases)
        sid = _msc_i64(get(subcase, "sid", i))
        push!(rows, _MSCResultDomainRec(i + 1, sid, 0, 1, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    end
    return rows
end

function _msc_result_domains_modal(eigenvalues; subcase::Integer=1, analysis_code::Integer=2, summary_start::Integer=2)
    count = length(eigenvalues)
    rows = _MSCResultDomainRec[_MSCResultDomainRec(1, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for i in 1:count
        push!(rows, _MSCResultDomainRec(summary_start + i - 1, subcase, 0, 0, 0.0, 0.0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    end
    vector_start = summary_start + count
    for i in 1:count
        eig = _msc_f64(eigenvalues[i])
        push!(rows, _MSCResultDomainRec(vector_start + i - 1, subcase, 0, analysis_code, eig, 0.0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    end
    return rows, vector_start
end

function _msc_result_domains_buckling(eigenvalues)
    count = length(eigenvalues)
    rows = _MSCResultDomainRec[
        _MSCResultDomainRec(1, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        _MSCResultDomainRec(2, 1, 0, 1, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    summary_start = 3
    for i in 1:count
        push!(rows, _MSCResultDomainRec(summary_start + i - 1, 2, 0, 0, 0.0, 0.0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    end
    vector_start = summary_start + count
    for i in 1:count
        eig = _msc_f64(eigenvalues[i])
        push!(rows, _MSCResultDomainRec(vector_start + i - 1, 2, 0, 8, eig, 0.0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    end
    return rows, vector_start
end

function _msc_write_common_attrs(file, results, filename)
    sol_type = _msc_i64(get(results, "sol_type", 0))
    attributes(file)["SCHEMA"] = Int64[20214]
    nastran = _msc_require_group(file, "/NASTRAN")
    attrs = attributes(nastran)
    attrs["ARCH"] = "openjfem"
    attrs["HOSTNAME"] = get(ENV, "COMPUTERNAME", get(ENV, "HOSTNAME", "openjfem"))
    attrs["IFPSTAR"] = "YES"
    attrs["INPUT"] = basename(filename)
    attrs["SOL"] = Int64[sol_type]
    attrs["TIME"] = string(Dates.now())
    attrs["VERSION"] = "openjfem"
end

function _msc_grid_weight_records(results)
    mass_summary = get(results, "mass_summary", Dict{String,Any}())
    mx = _msc_f64(get(mass_summary, "total_mass_x", 0.0))
    my = _msc_f64(get(mass_summary, "total_mass_y", mx))
    mz = _msc_f64(get(mass_summary, "total_mass_z", mx))
    mo = ntuple(i -> i == 1 ? mx : i == 8 ? my : i == 15 ? mz : 0.0, 36)
    identity3 = ntuple(i -> (i == 1 || i == 5 || i == 9) ? 1.0 : 0.0, 9)
    inertia = ntuple(_ -> 0.0, 9)
    return [_MSCGridWeightRec(
        0, mo, identity3,
        mx, 0.0, 0.0, 0.0,
        my, 0.0, 0.0, 0.0,
        mz, 0.0, 0.0, 0.0,
        inertia, 0.0, 0.0, 0.0, identity3, 1,
    )]
end

function _msc_write_grid_weight(file, results)
    rows = _msc_grid_weight_records(results)
    _msc_write_custom(file, "/NASTRAN/RESULT/NODAL/GRID_WEIGHT",
        rows, _msc_grid_weight_dtype)
    _msc_write_records(file, "/INDEX/NASTRAN/RESULT/NODAL/GRID_WEIGHT",
        [_MSCIndexRec(1, 0, length(rows))])
end

const _MSC_EFORCE_BAR_FIELDS = ((:EID, :i64, 1), (:BM1A, :f64, 1), (:BM2A, :f64, 1), (:BM1B, :f64, 1), (:BM2B, :f64, 1), (:TS1, :f64, 1), (:TS2, :f64, 1), (:AF, :f64, 1), (:TRQ, :f64, 1), (:DOMAIN_ID, :i64, 1))
const _MSC_EFORCE_BEAM_FIELDS = ((:EID, :i64, 1), (:GRID, :i64_array, 11), (:SD, :f64_array, 11), (:BM1, :f64_array, 11), (:BM2, :f64_array, 11), (:TS1, :f64_array, 11), (:TS2, :f64_array, 11), (:AF, :f64_array, 11), (:TTRQ, :f64_array, 11), (:WTRQ, :f64_array, 11), (:DOMAIN_ID, :i64, 1))
const _MSC_EFORCE_ROD_FIELDS = ((:EID, :i64, 1), (:AF, :f64, 1), (:TRQ, :f64, 1), (:DOMAIN_ID, :i64, 1))
const _MSC_EFORCE_ELAS_FIELDS = ((:EID, :i64, 1), (:F, :f64, 1), (:DOMAIN_ID, :i64, 1))
const _MSC_EFORCE_QUAD_FIELDS = ((:EID, :i64, 1), (:TERM, :str, 4), (:GRID, :i64_array, 5), (:MX, :f64_array, 5), (:MY, :f64_array, 5), (:MXY, :f64_array, 5), (:BMX, :f64_array, 5), (:BMY, :f64_array, 5), (:BMXY, :f64_array, 5), (:TX, :f64_array, 5), (:TY, :f64_array, 5), (:DOMAIN_ID, :i64, 1))
const _MSC_EFORCE_TRIA_FIELDS = ((:EID, :i64, 1), (:MX, :f64, 1), (:MY, :f64, 1), (:MXY, :f64, 1), (:BMX, :f64, 1), (:BMY, :f64, 1), (:BMXY, :f64, 1), (:TX, :f64, 1), (:TY, :f64, 1), (:DOMAIN_ID, :i64, 1))

const _MSC_BAR_STRESS_FIELDS = ((:EID, :i64, 1), (:X1A, :f64, 1), (:X2A, :f64, 1), (:X3A, :f64, 1), (:X4A, :f64, 1), (:AX, :f64, 1), (:MAXA, :f64, 1), (:MINA, :f64, 1), (:MST, :f64, 1), (:X1B, :f64, 1), (:X2B, :f64, 1), (:X3B, :f64, 1), (:X4B, :f64, 1), (:MAXB, :f64, 1), (:MINB, :f64, 1), (:MSC, :f64, 1), (:DOMAIN_ID, :i64, 1))
const _MSC_BEAM_STRESS_FIELDS = ((:EID, :i64, 1), (:GRID, :i64_array, 11), (:SD, :f64_array, 11), (:XC, :f64_array, 11), (:XD, :f64_array, 11), (:XE, :f64_array, 11), (:XF, :f64_array, 11), (:MAX, :f64_array, 11), (:MIN, :f64_array, 11), (:MST, :f64_array, 11), (:MSC, :f64_array, 11), (:DOMAIN_ID, :i64, 1))
const _MSC_ROD_STRESS_FIELDS = ((:EID, :i64, 1), (:A, :f64, 1), (:MSA, :f64, 1), (:T, :f64, 1), (:MST, :f64, 1), (:DOMAIN_ID, :i64, 1))
const _MSC_ELAS_STRESS_FIELDS = ((:EID, :i64, 1), (:S, :f64, 1), (:DOMAIN_ID, :i64, 1))
const _MSC_SOLID_STRESS_FIELDS_9 = ((:EID, :i64, 1), (:CID, :i64, 1), (:CTYPE, :str, 4), (:NODEF, :i64, 1), (:GRID, :i64_array, 9), (:X, :f64_array, 9), (:Y, :f64_array, 9), (:Z, :f64_array, 9), (:TXY, :f64_array, 9), (:TYZ, :f64_array, 9), (:TZX, :f64_array, 9), (:DOMAIN_ID, :i64, 1))
const _MSC_SOLID_STRESS_FIELDS_7 = ((:EID, :i64, 1), (:CID, :i64, 1), (:CTYPE, :str, 4), (:NODEF, :i64, 1), (:GRID, :i64_array, 7), (:X, :f64_array, 7), (:Y, :f64_array, 7), (:Z, :f64_array, 7), (:TXY, :f64_array, 7), (:TYZ, :f64_array, 7), (:TZX, :f64_array, 7), (:DOMAIN_ID, :i64, 1))
const _MSC_SOLID_STRESS_FIELDS_5 = ((:EID, :i64, 1), (:CID, :i64, 1), (:CTYPE, :str, 4), (:NODEF, :i64, 1), (:GRID, :i64_array, 5), (:X, :f64_array, 5), (:Y, :f64_array, 5), (:Z, :f64_array, 5), (:TXY, :f64_array, 5), (:TYZ, :f64_array, 5), (:TZX, :f64_array, 5), (:DOMAIN_ID, :i64, 1))
const _MSC_QUAD_COMP_STRESS_FIELDS = ((:EID, :i64, 1), (:PLY, :i64, 1), (:X1, :f64, 1), (:Y1, :f64, 1), (:T1, :f64, 1), (:L1, :f64, 1), (:L2, :f64, 1), (:DOMAIN_ID, :i64, 1))
const _MSC_QUAD_CN_STRESS_FIELDS = ((:EID, :i64, 1), (:TERM, :str, 4), (:GRID, :i64_array, 5), (:FD1, :f64_array, 5), (:X1, :f64_array, 5), (:Y1, :f64_array, 5), (:TXY1, :f64_array, 5), (:FD2, :f64_array, 5), (:X2, :f64_array, 5), (:Y2, :f64_array, 5), (:TXY2, :f64_array, 5), (:DOMAIN_ID, :i64, 1))
const _MSC_TRIA_STRESS_FIELDS = ((:EID, :i64, 1), (:FD1, :f64, 1), (:X1, :f64, 1), (:Y1, :f64, 1), (:TXY1, :f64, 1), (:FD2, :f64, 1), (:X2, :f64, 1), (:Y2, :f64, 1), (:TXY2, :f64, 1), (:DOMAIN_ID, :i64, 1))

_msc_element_values(model, key) = sort(collect(values(get(model, key, Dict()))); by=x -> _msc_i64(get(x, "ID", 0)))
_msc_element_id_rows(elements) = [Dict{Symbol,Any}(:EID => _msc_i64(get(e, "ID", 0))) for e in elements]

function _msc_shell_groups(model)
    pshells = get(model, "PSHELLs", Dict())
    quads = Any[]
    quad_standard = Any[]
    quad_composite = Any[]
    trias = Any[]
    for elem in _msc_element_values(model, "CSHELLs")
        nodes = get(elem, "NODES", Int[])
        pid = _msc_i64(get(elem, "PID", 0))
        prop = _msc_model_value_by_id(pshells, pid)
        is_composite = prop !== nothing && uppercase(string(get(prop, "TYPE", ""))) == "PCOMP_CLT"
        if length(nodes) == 4
            push!(quads, elem)
            push!(is_composite ? quad_composite : quad_standard, elem)
        elseif length(nodes) == 3
            push!(trias, elem)
        end
    end
    return quads, quad_standard, quad_composite, trias
end

function _msc_shell_grid(elem, n::Integer)
    nodes = _msc_i64.(get(elem, "NODES", Int[]))
    if n == 5 && length(nodes) >= 4
        return [_msc_i64(length(nodes)); nodes[1:min(length(nodes), 4)]]
    end
    return vcat([_msc_i64(length(nodes))], nodes)[1:min(n, length(nodes) + 1)]
end

function _msc_solid_grid(elem, n::Integer)
    nodes = _msc_i64.(get(elem, "NODES", Int[]))
    return vcat([0], nodes)[1:min(n, length(nodes) + 1)]
end

function _msc_pcomp_ply_count(model, elem)
    prop = _msc_model_value_by_id(get(model, "PSHELLs", Dict()), _msc_i64(get(elem, "PID", 0)))
    prop === nothing && return 0
    return length(get(prop, "PLY_DATA", Any[]))
end

function _msc_with_domain(rows, domain_id)
    out = Vector{Dict{Symbol,Any}}(undef, length(rows))
    for (i, row) in enumerate(rows)
        copy_row = copy(row)
        copy_row[:DOMAIN_ID] = _msc_i64(domain_id)
        out[i] = copy_row
    end
    return out
end

function _msc_write_element_table(file, path, fields, rows, domain_ids)
    isempty(rows) && return
    all_rows = Dict{Symbol,Any}[]
    index = _MSCIndexRec[]
    position = 0
    for domain_id in domain_ids
        domain_rows = _msc_with_domain(rows, domain_id)
        append!(all_rows, domain_rows)
        push!(index, _MSCIndexRec(_msc_i64(domain_id), position, length(domain_rows)))
        position += length(domain_rows)
    end
    _msc_write_dynamic_records(file, path, fields, all_rows)
    _msc_write_records(file, "/INDEX" * path, index)
end

function _msc_write_static_element_results(file, results, domain_ids)
    isempty(domain_ids) && return
    model = results["model"]
    bars = _msc_element_id_rows(_msc_element_values(model, "CBARs"))
    beams = _msc_element_id_rows(_msc_element_values(model, "CBEAMs"))
    rods = _msc_element_id_rows(_msc_element_values(model, "CRODs"))
    conrods = _msc_element_id_rows(_msc_element_values(model, "CONRODs"))
    celas1, celas2 = _msc_input_spring_records(model)
    elas1 = [Dict{Symbol,Any}(:EID => e.EID) for e in celas1]
    elas2 = [Dict{Symbol,Any}(:EID => e.EID) for e in celas2]
    quads, quad_standard, quad_composite, trias = _msc_shell_groups(model)
    quad_force = [Dict{Symbol,Any}(:EID => _msc_i64(get(e, "ID", 0)), :TERM => "CEN/", :GRID => _msc_shell_grid(e, 5)) for e in quads]
    quad_standard_rows = [Dict{Symbol,Any}(:EID => _msc_i64(get(e, "ID", 0)), :TERM => "CEN/", :GRID => _msc_shell_grid(e, 5)) for e in quad_standard]
    quad_comp_rows = Dict{Symbol,Any}[]
    for e in quad_composite
        for ply in 1:max(_msc_pcomp_ply_count(model, e), 1)
            push!(quad_comp_rows, Dict{Symbol,Any}(:EID => _msc_i64(get(e, "ID", 0)), :PLY => ply))
        end
    end
    tria_rows = _msc_element_id_rows(trias)

    solids = _msc_element_values(model, "CSOLIDs")
    hexa_rows = Dict{Symbol,Any}[]
    penta_rows = Dict{Symbol,Any}[]
    tetra_rows = Dict{Symbol,Any}[]
    for solid in solids
        etype = uppercase(string(get(solid, "TYPE", "")))
        eid = _msc_i64(get(solid, "ID", 0))
        if etype == "CHEXA"
            push!(hexa_rows, Dict{Symbol,Any}(:EID => eid, :CTYPE => "GRID", :NODEF => 8, :GRID => _msc_solid_grid(solid, 9)))
        elseif etype == "CPENTA"
            push!(penta_rows, Dict{Symbol,Any}(:EID => eid, :CTYPE => "GRID", :NODEF => 6, :GRID => _msc_solid_grid(solid, 7)))
        elseif etype == "CTETRA"
            push!(tetra_rows, Dict{Symbol,Any}(:EID => eid, :CTYPE => "GRID", :NODEF => 4, :GRID => _msc_solid_grid(solid, 5)))
        end
    end

    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/BAR", _MSC_EFORCE_BAR_FIELDS, bars, domain_ids)
    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/BEAM", _MSC_EFORCE_BEAM_FIELDS, beams, domain_ids)
    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/CONROD", _MSC_EFORCE_ROD_FIELDS, conrods, domain_ids)
    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/ELAS1", _MSC_EFORCE_ELAS_FIELDS, elas1, domain_ids)
    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/ELAS2", _MSC_EFORCE_ELAS_FIELDS, elas2, domain_ids)
    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/QUAD4_CN", _MSC_EFORCE_QUAD_FIELDS, quad_force, domain_ids)
    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/ROD", _MSC_EFORCE_ROD_FIELDS, rods, domain_ids)
    _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/ELEMENT_FORCE/TRIA3", _MSC_EFORCE_TRIA_FIELDS, tria_rows, domain_ids)

    for family in ("STRESS", "STRAIN")
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/BAR", _MSC_BAR_STRESS_FIELDS, bars, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/BEAM", _MSC_BEAM_STRESS_FIELDS, beams, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/CONROD", _MSC_ROD_STRESS_FIELDS, conrods, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/ELAS1", _MSC_ELAS_STRESS_FIELDS, elas1, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/ELAS2", _MSC_ELAS_STRESS_FIELDS, elas2, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/HEXA", _MSC_SOLID_STRESS_FIELDS_9, hexa_rows, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/PENTA", _MSC_SOLID_STRESS_FIELDS_7, penta_rows, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/QUAD4_COMP", _MSC_QUAD_COMP_STRESS_FIELDS, quad_comp_rows, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/QUAD_CN", _MSC_QUAD_CN_STRESS_FIELDS, quad_standard_rows, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/ROD", _MSC_ROD_STRESS_FIELDS, rods, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/TETRA", _MSC_SOLID_STRESS_FIELDS_5, tetra_rows, domain_ids)
        _msc_write_element_table(file, "/NASTRAN/RESULT/ELEMENTAL/$family/TRIA3", _MSC_TRIA_STRESS_FIELDS, tria_rows, domain_ids)
    end
end

const _MSC_CASECC_FIELD_NAMES_RAW = """
SID;MPCSET;SPCSET;ESLSET;REESET;ELDSET;THLDSET;THMATSET;TIC;NONPTSET;NONMEDIA;NONFMT;DYMLDSET;FEQRESET;TFSET;SYMFLG;LDSPTSET;LDSMEDIA;LDSFMT;DPLPTSET;DPLMEDIA;DPLFMT;STSPTSET;STSMEDIA;STSFMT;FCEPTSET;FCEMEDIA;FCEFMT;ACCPTSET;ACCMEDIA;ACCFMT;VELPTSET;VELMEDIA;VELFMT;FOCPTSET;FOCMEDIA;FOCFMT;TSTEPTRN;TITLE;SUBTITLE;LABEL;STPLTFLG;AXSYMSET;NOHARMON;TSTRU;SETKP;FLAGKP;K2PP;SETMP;FLAGMP;M2PP;SETBP;FLAGBP;B2PP;OUTRESPV;SEDR;FLDBNDY;CEESET;DAMPTBL;DYNRED;SSDSET;SSDMEDIA;SSDFMT;SSVSET;SSVMEDIA;SSVFMT;SSASET;SSAMEDIA;SSAFMT;NONLINLD;PARTIT;CYCLIC;RANDOM;NONPARAM;FLUTTER;LCC;GPFSET;GPFMEDIA;GPFFMT;ESESET;ESEMEDIA;ESEFMT;ARFPTSET;ARFMEDIA;ARFFMT;SEID;LCN;GUST;SEFINAL;SEMG;SEKR;SELG;SELR;SEEX;SETKG;FLAGKG;K2GG;SETMG;FLAGMG;M2GG;SETBG;FLAGBG;B2GG;SVSET;SVMEDIA;SVFMT;FLUPTSET;FLUMEDIA;FLUFMT;HOUT;NOUT;SETPG;FLAGPG;P2G;LOADSET;SEMR;VONMISES;SECMDFLG;GPSPTSET;GPSMEDIA;GPSFMT;STFSET;STFMEDIA;STFFMT;CLOAD;SET2ID;DSAPRT;DSASTORE;DSAOUTPT;STNSET;STNMEDIA;STNFMT;APRESS;TRIM;MODLIST;REESETF;ESDPTSET;ESDMEDIA;ESDFMT;GSDPTSET;GSDMEDIA;GSDFMT;SEDV;SERE;SERS;CNTSET;CNTMEDIA;CNTFMT;DIVERG;OUTRCV;STATSUBP;DFT1;DFT2;ADAPT;DESOBJ;DESSUB;SUBSPAN;DESGLB;ANALYSIS;GPQSTRS;GPQFORC;GPQSTRN;SUPORT1;STATSUBB;BCID;AUXMODEL;ADACT;DATSET;DATMEDIA;DATFMT;VUGSET;VUGMEDIA;VUGFMT;MPCFSET;MPCMEDIA;MPCFFMT;REUESET;DAMPTBLF;ITERMETH;NLSSET;NLSMEDIA;NLSFMT;MODTRKID;DSAFORM;DSAEXPO;DSABEGIN;DSAINTVL;DSAFINAL;DSASETID;SORTFLG;RANDBIT;AECONFIG;AESYMXY;AESYMXZ;OCIDREQ;GPEPTSET;GPEMEDIA;GPEFMT;TEMPMAT;AECSSSET;EKEPTSET;EKEMEDIA;EKEFMT;EKETHRSH;EDEPTSET;EDEMEDIA;EDEFMT;EDETHRSH;DFT3;DFR1;DFR2;DFR3;SETK4G;FLAGK4G;K42GG;A2GG;NK42GG;NA2GG;EFFMASET;EFFMAGID;EFFMATHR;EQUILMED;EQUILGRD;RCRSET;RCRFMT;AEUXREF;GCHK;GCHKOUT;GCHKSET;GCHKGID;GCHKTHR;GCHKRTHR;GCHKDREC;ASPCMED;ASPCEPS;ASPCPRT;ASPCPCH;NK2PP;NM2PP;NB2PP;NK2GG;NM2GG;NB2GG;NP2G;GEODSET;GEODMXMN;GEODOCID;GEODNUMB;GEOLSET;GEOLMXMN;GEOLOCID;GEOLNUMB;GEOSSET;GEOSMXMN;GEOSOCID;GEOSNUMB;GEOMSET;GEOMMXMN;GEOMOCID;GEOMNUMB;GEOASET;GEOAMXMN;GEOAOCID;GEOANUMB;GEOVSET;GEOVMXMN;GEOVOCID;GEOVNUMB;NTFL;GPKESET;GPKEMEDI;GPKEFMT;SEDAMP;WCHK;WCHKOUT;WCHKSET;WCHKGID;WCHKCGI;WCHKWM;EXSEOUT;EXSEMED;EXSEUNIT;EXSERES1;EXSERES2;FK2PP;FM2PP;FB2PP;FK2GG;FM2GG;FB2GG;TICTYPE;FK42GG;FA2GG;SUBSTEP;STEPID;NSMID;ROUTDISP;ROUTVELO;ROUTACCE;ROUTLOAD;ROUTSPCF;ROUTSTRS;ROUTFORC;ROUTSTRN;ROUTMSCF;MDLSSET;MDLSMEDIA;MDLSFMT;MDLSESRT;MDLSTHRE;MDLSTFVL;MDLKSET;MDLKMEDIA;MDLKFMT;MDLKESRT;MDLKTHRE;MDLKTFVL;ACPOWSET;ACPOWMED;ACPOWFMT;ACPOWCSV;NLOUT;SEEFMNO;SEEFMHV;SEEFMDLF;SEEFMBND;SEEFMSDP;CONNECTOR;ESETHRSH;POSTUNIT;POSTOPT1;POSTOPT2;TICDIFF;DSAESEID;MXMNGSET;MXMNGMDA;MXMNGFMT;MXMNESET;MXMNEMDA;MXMNEFMT;MCFRSET;MCFRSOLN;MCFRFILT;MCFROPT;ELSUMID;ELSUMOPT;ELSUMDUM;RGYRO;CMSESET;CMSEMDIA;CMSEOPTS;CMSETHRE;CMSETOPN;GPRSORT;MASSSET;AESOLN;POSTO2NM;RANDVAR;RSVCRQTS;RSVCOPTS;RSVCSTBS;RSVCRQTC;RSVCOPTC;RSVCSTBC;DESVAR;BCONTACTI;BCONTACTC;MODSELS1;MODSELS2;MODSELS3;MODSELS4;MODSELS5;MODSELS6;MODSELS7;MODSELS8;MODSELS9;MODSELF1;MODSELF2;MODSELF3;MODSELF4;MODSELF5;MODSELF6;MODSELF7;MODSELF8;MODSELF9;FTNURN;SUFNAM1;SUFNAM2;ENVELOP1;ENVELOP2;GPFLXSET;GPFLXMED;CAMPBELL;SPLINOUT;MONITOR;FBODYLD;STOCHAST;EXPTLDID;EXPTLDNM;EXPTLDSI;AERCONFIG;NLHARM;PFMSSID;PFMSMED;PFMSFMT;PFMSSSET;PFMSFLTR;PFMSOPTS;PFMSSTMP;PFMFSID;PFMFMED;PFMFFMT;PFMFSSET;PFMFPSET;PFMFFLTR;PFMFOPTS;PFMFFLMP;PFMFSTMP;PFPSID;PFPMED;PFPFMT;PFPSSET;PFPFLTR;PFPOPTS;PFPPSET;PFGSID;PFGMED;PFGFMT;PFGSSET;PFGGSET;NLICCASE;NLICSTEP;NLICLFAC;ACFPMSET;ACFPMMED;ACFPMFMT;FRFFLAG;FRFCMPID;FRFCONST;FRFUNTNO;FRFCMPNM;DESMOD;RSDAMPST;RSDAMPFL;VCCT;FP2G;FRQVAR;HADAPT;BCHANGE;BCMOVE;BSQUEAL;UNGLUE;HSUBCASE;HSTEP;HTIME;ERPSID;ERPMED;ERPFMT;ERPSSET;ERPFLTR;ERPOPTS;ERPCSV;TESTTHRR;TESTTHRI;ASMOUTFL;ASMOUTNM;ICFUNTNO;TFOSET;TFUNIT;TFLSET;TFOPTS;NLOOPH;NLSTEP;NMODES;RCPARM;NLOPCTRL;NLOPDBG;NLOPPOST;NLOPMPCH;NLICSSTP;DEACTEL;ACTIVAT;INTENSET;INTENMED;INTENFMT;NLICTOLR;PFPSSID;PFPSMED;PFPSFMT;PFPSSSET;PFPSFLTR;PFPSOPTS;PFPSPSET;ACTISET;ELSOSET;ELSRSET;ELSSSET;ELSTHRS;ELSBITS;WTSOSET;WTSRSET;WTSSSET;WTSTHRS;WTSBITS;PACOSET;PACSSET;PACBITS;IRLOAD;ICFSET;ICFMED;ICFFMT;ICFGENST;ICFGENNM;ICFUSEST;ICFUSENM;HISTSET;HISTTYPE;HISTFMT;FATIGUE;FTGMED;FTGFMT;NLOPDELI;NLOPGRID;DASAVE;GVSET;GVMEDIA;GVFMT;ERMPF;NVELOSET;NVELFDEF;NVELFDTA;NVELTHRS;NVELBITS;VITSOSET;VITSRSET;VITSSSET;VITSTHRS;VITSBITS;THLDVER;THMATVER;NPEAK;NEAR;LFREQ;HFREQ;RTYPE;PSCALE;NSAMP;MONSET;SEED;OFFD;FSORT2;MFREQ;BCONTACT;SOLNID;ExtDROut;ExtDROmd;ExtDROun;ExtDRIn;ExtDRImd;ExtDRIun;FemCheck;LSEM;SYM_LEN;SYM_POS;DOMAIN_ID
"""
const _MSC_CASECC_FLOAT_FIELD_NAMES = Set(Symbol.(split("DFT1;DFT2;EKETHRSH;EDETHRSH;DFT3;DFR1;DFR2;DFR3;EFFMATHR;GCHKTHR;GCHKRTHR;ASPCEPS;ASPCPRT;MDLSTHRE;MDLKTHRE;ESETHRSH;MCFRFILT;CMSETHRE;MODSELS4;MODSELS5;MODSELS6;MODSELS7;MODSELS8;MODSELS9;MODSELF4;MODSELF5;MODSELF6;MODSELF7;MODSELF8;MODSELF9;PFMSFLTR;PFMFFLTR;PFPFLTR;NLICLFAC;HTIME;ERPFLTR;TESTTHRR;NLICTOLR;PFPSFLTR;ELSTHRS;WTSTHRS;NVELTHRS;VITSTHRS;NEAR;LFREQ;HFREQ;OFFD", ";")))
const _MSC_CASECC_STRING_FIELD_SIZES = let entries = split("TITLE:128;SUBTITLE:128;LABEL:128;K2PP:8;M2PP:8;B2PP:8;K2GG:8;M2GG:8;B2GG:8;P2G:8;ANALYSIS:4;AECONFIG:8;K42GG:8;A2GG:8;TICTYPE:4;MDLSESRT:4;MDLKESRT:4;SEEFMHV:4;TICDIFF:4;AESOLN:8;POSTO2NM:8;BCONTACTC:4;ENVELOP1:4;ENVELOP2:4;EXPTLDNM:8;AERCONFIG:8;DESMOD:8;BCONTACT:4", ";")
    Dict{Symbol,Int}(Symbol(split(e, ":")[1]) => parse(Int, split(e, ":")[2]) for e in entries)
end
const _MSC_CASECC_ARRAY_FIELD_DIMS = let entries = split("HOUT:3;NOUT:3;SPLINOUT:2;FBODYLD:2;FRFCMPNM:2;ASMOUTNM:2;ICFGENNM:2;ICFUSENM:2", ";")
    Dict{Symbol,Int}(Symbol(split(e, ":")[1]) => parse(Int, split(e, ":")[2]) for e in entries)
end
const _MSC_CASECC_FIELD_NAMES = Symbol.(split(replace(strip(_MSC_CASECC_FIELD_NAMES_RAW), r"\s+" => ""), ";"))

function _msc_casecc_fields()
    return [(field,
        haskey(_MSC_CASECC_STRING_FIELD_SIZES, field) ? :str :
        haskey(_MSC_CASECC_ARRAY_FIELD_DIMS, field) ? :i64_array :
        field in _MSC_CASECC_FLOAT_FIELD_NAMES ? :f64 : :i64,
        get(_MSC_CASECC_STRING_FIELD_SIZES, field, get(_MSC_CASECC_ARRAY_FIELD_DIMS, field, 1)))
        for field in _MSC_CASECC_FIELD_NAMES]
end

function _msc_casecc_subcase_rows(model)
    case_control = get(model, "CASE_CONTROL", Dict{String,Any}())
    subcases = get(case_control, "SUBCASES", Dict{Int,Dict{String,Any}}())
    if isempty(subcases)
        subcases = Dict(1 => Dict{String,Any}())
    end
    sol_type = _msc_i64(get(model, "SOL", get(model, "sol_type", 0)))
    title = strip(string(get(case_control, "TITLE", get(model, "TITLE", "OpenJFEM"))))
    title = isempty(title) ? "OpenJFEM" : title
    rows = Dict{Symbol,Any}[]
    for sid in sort(collect(keys(subcases)))
        sub = subcases[sid]
        row = Dict{Symbol,Any}(
            :SID => sid, :SPCSET => _msc_i64(get(sub, "SPC", get(case_control, "SPC", 0))),
            :MPCSET => _msc_i64(get(sub, "MPC", get(case_control, "MPC", 0))),
            :TITLE => title,
            :SUBTITLE => string(get(sub, "SUBTITLE", sol_type == 105 && haskey(sub, "STATSUB") ? "BUCKLING MODES" : "")),
            :LABEL => string(get(sub, "LABEL", "SUBCASE $sid")),
            :NOHARMON => 1, :SEDR => -1, :LCC => 1000,
            :SEID => -1, :SEMG => -1, :SEKR => -1, :SELG => -1, :SELR => -1, :SEMR => -1,
            :SECMDFLG => -1, :SEDV => -1, :SERE => -1,
            :DFT1 => -1.0, :DFT2 => -1.0, :DFT3 => -1.0, :DFR1 => -1.0, :DFR2 => -1.0, :DFR3 => -1.0,
            :EKETHRSH => -1.0, :EDETHRSH => -1.0, :ESETHRSH => -1.0,
            :SORTFLG => 1, :AECONFIG => "AEROSG2D", :MONITOR => -2, :VCCT => -99,
            :HSUBCASE => -1, :HSTEP => -1, :HTIME => -1.0, :DOMAIN_ID => 1,
        )
        if haskey(sub, "LOAD") && !isnothing(sub["LOAD"])
            row[:ESLSET] = _msc_i64(sub["LOAD"])
            row[:LDSPTSET] = -1; row[:LDSMEDIA] = 7; row[:LDSFMT] = 1
            row[:ROUTLOAD] = 1
        end
        if haskey(sub, "METHOD") && !isnothing(sub["METHOD"])
            row[:REESET] = _msc_i64(sub["METHOD"])
            row[:REESETF] = _msc_i64(sub["METHOD"])
        end
        if haskey(sub, "STATSUB") && !isnothing(sub["STATSUB"])
            row[:STATSUBB] = _msc_i64(sub["STATSUB"])
        end
        row[:DPLPTSET] = -1; row[:DPLMEDIA] = 7; row[:DPLFMT] = 1; row[:ROUTDISP] = 1
        row[:FOCPTSET] = -1; row[:FOCMEDIA] = 7; row[:FOCFMT] = 1
        row[:ROUTSPCF] = 1
        if sol_type == 101 || sol_type == 105
            row[:STSPTSET] = -1; row[:STSMEDIA] = 7; row[:STSFMT] = 1
            row[:FCEPTSET] = -1; row[:FCEMEDIA] = 7; row[:FCEFMT] = 1
            row[:STNSET] = -1; row[:STNMEDIA] = 7; row[:STNFMT] = 1
            row[:MPCFSET] = -1; row[:MPCMEDIA] = 7; row[:MPCFFMT] = 1
            row[:NLSSET] = -1; row[:NLSMEDIA] = 7; row[:NLSFMT] = 1
            row[:VONMISES] = 3
            row[:GPQSTRS] = 3; row[:GPQFORC] = 3; row[:GPQSTRN] = 3
            row[:ROUTSTRS] = 1; row[:ROUTFORC] = 1; row[:ROUTSTRN] = 1; row[:ROUTMSCF] = 1
        end
        push!(rows, row)
    end
    return rows
end

function _msc_write_casecc_subcase(file, model)
    rows = _msc_casecc_subcase_rows(model)
    _msc_write_dynamic_records(file, "/NASTRAN/INPUT/PARAMETER/CASECC/SUBCASE", _msc_casecc_fields(), rows; version=1)
end

function _msc_vector_from_nodal_entries(id_map, entries)
    vector = zeros(Float64, 6 * length(id_map))
    for entry in entries
        nid = _msc_i64(get(entry, "grid_id", get(entry, "ID", 0)))
        haskey(id_map, nid) || continue
        base = (id_map[nid] - 1) * 6
        vector[base + 1] = _msc_f64(get(entry, "t1", get(entry, "X", 0.0)))
        vector[base + 2] = _msc_f64(get(entry, "t2", get(entry, "Y", 0.0)))
        vector[base + 3] = _msc_f64(get(entry, "t3", get(entry, "Z", 0.0)))
        vector[base + 4] = _msc_f64(get(entry, "r1", get(entry, "RX", 0.0)))
        vector[base + 5] = _msc_f64(get(entry, "r2", get(entry, "RY", 0.0)))
        vector[base + 6] = _msc_f64(get(entry, "r3", get(entry, "RZ", 0.0)))
    end
    return vector
end

function _msc_applied_load_vector(results, load_id)
    id_map = results["id_map"]
    vector = zeros(Float64, 6 * length(id_map))
    isnothing(load_id) && return vector
    forces, moments = collect_point_loads(results["model"], load_id)
    for (nid, force) in forces
        haskey(id_map, nid) || continue
        base = (id_map[nid] - 1) * 6
        for i in 1:3
            vector[base + i] = _msc_f64(force[i])
        end
    end
    for (nid, moment) in moments
        haskey(id_map, nid) || continue
        base = (id_map[nid] - 1) * 6
        for i in 1:3
            vector[base + 3 + i] = _msc_f64(moment[i])
        end
    end
    return vector
end

function _msc_static_load_id(results, subcase)
    model = results["model"]
    sid = _msc_i64(get(subcase, "sid", 0))
    case_control = get(model, "CASE_CONTROL", Dict{String,Any}())
    subcases = get(case_control, "SUBCASES", Dict{Int,Dict{String,Any}}())
    ctrl = haskey(subcases, sid) ? subcases[sid] : Dict{String,Any}()
    return get(ctrl, "LOAD", nothing)
end

function _msc_sol105_static_load_id(results)
    diagnostics = get(results, "solver_diagnostics", Any[])
    isempty(diagnostics) && return nothing
    first_diag = diagnostics[1]
    first_diag isa AbstractDict || return nothing
    static_sid = get(first_diag, "static_subcase", nothing)
    static_sid === nothing && return nothing
    case_control = get(results["model"], "CASE_CONTROL", Dict{String,Any}())
    subcases = get(case_control, "SUBCASES", Dict{Int,Dict{String,Any}}())
    ctrl = get(subcases, _msc_i64(static_sid), Dict{String,Any}())
    return get(ctrl, "LOAD", nothing)
end

function _msc_write_nodal_table(file, path, id_map, vectors_with_domains)
    rows = _MSCNodalResultRec[]
    index = _MSCIndexRec[]
    position = 0
    node_count = length(id_map)
    for (domain_id, vector) in vectors_with_domains
        append!(rows, _msc_nodal_rows_from_vector(id_map, vector, domain_id))
        push!(index, _MSCIndexRec(_msc_i64(domain_id), position, node_count))
        position += node_count
    end
    _msc_write_records(file, path, rows; version=1)
    _msc_write_records(file, "/INDEX" * path, index)
end

function _msc_write_sol101_results(file, results)
    subcases = get(results, "subcases", Any[])
    _msc_write_records(file, "/NASTRAN/RESULT/DOMAINS", _msc_result_domains_static(subcases); version=20200)
    _msc_write_grid_weight(file, results)
    _msc_write_static_element_results(file, results, [_msc_i64(i + 1) for i in eachindex(subcases)])
    displacements = _MSCNodalResultRec[]
    index = _MSCIndexRec[]
    position = 0
    node_count = length(get(results, "id_map", Dict()))
    for (i, subcase) in enumerate(subcases)
        domain_id = i + 1
        rows = _msc_nodal_rows_from_vector(results["id_map"], get(subcase, "raw_displacement", Float64[]), domain_id)
        append!(displacements, rows)
        push!(index, _MSCIndexRec(domain_id, position, node_count))
        position += node_count
    end
    _msc_write_records(file, "/NASTRAN/RESULT/NODAL/DISPLACEMENT", displacements; version=1)
    _msc_write_records(file, "/INDEX/NASTRAN/RESULT/NODAL/DISPLACEMENT", index)

    applied = Tuple{Int64,Vector{Float64}}[]
    spc = Tuple{Int64,Vector{Float64}}[]
    mpc = Tuple{Int64,Vector{Float64}}[]
    for (i, subcase) in enumerate(subcases)
        domain_id = _msc_i64(i + 1)
        push!(applied, (domain_id, _msc_applied_load_vector(results, _msc_static_load_id(results, subcase))))
        push!(spc, (domain_id, _msc_vector_from_nodal_entries(results["id_map"], get(subcase, "spc_forces", Any[]))))
        push!(mpc, (domain_id, zeros(Float64, 6 * length(results["id_map"]))))
    end
    _msc_write_nodal_table(file, "/NASTRAN/RESULT/NODAL/APPLIED_LOAD", results["id_map"], applied)
    _msc_write_nodal_table(file, "/NASTRAN/RESULT/NODAL/SPC_FORCE", results["id_map"], spc)
    _msc_write_nodal_table(file, "/NASTRAN/RESULT/NODAL/MPC_FORCE", results["id_map"], mpc)
end

function _msc_write_sol103_results(file, results)
    eigenvalues = collect(get(results, "eigenvalues", Float64[]))
    mode_shapes = get(results, "_raw_mode_shapes", zeros(0, 0))
    domains, vector_start = _msc_result_domains_modal(eigenvalues; subcase=1, analysis_code=2, summary_start=2)
    _msc_write_records(file, "/NASTRAN/RESULT/DOMAINS", domains; version=20200)
    _msc_write_grid_weight(file, results)
    eigen_rows = _msc_eigenvalue_rows(eigenvalues, 2; frequencies=get(results, "frequencies", nothing))
    _msc_write_records(file, "/NASTRAN/RESULT/SUMMARY/EIGENVALUE", eigen_rows)
    _msc_write_records(file, "/INDEX/NASTRAN/RESULT/SUMMARY/EIGENVALUE", [_MSCIndexRec(2, 0, length(eigen_rows))])
    evec_rows, evec_index = _msc_eigenvector_rows(results["id_map"], mode_shapes, vector_start)
    _msc_write_records(file, "/NASTRAN/RESULT/NODAL/EIGENVECTOR", evec_rows; version=1)
    _msc_write_records(file, "/INDEX/NASTRAN/RESULT/NODAL/EIGENVECTOR", evec_index)

    zero = zeros(Float64, 6 * length(results["id_map"]))
    spc_vectors = [(Int64(vector_start + i - 1), zero) for i in 1:length(eigenvalues)]
    _msc_write_nodal_table(file, "/NASTRAN/RESULT/NODAL/SPC_FORCE", results["id_map"], spc_vectors)
end

function _msc_write_sol105_results(file, results)
    eigenvalues = collect(get(results, "eigenvalues", Float64[]))
    mode_shapes = get(results, "_raw_mode_shapes", zeros(0, 0))
    domains, vector_start = _msc_result_domains_buckling(eigenvalues)
    _msc_write_records(file, "/NASTRAN/RESULT/DOMAINS", domains; version=20200)
    _msc_write_grid_weight(file, results)
    haskey(results, "u_static") && _msc_write_static_element_results(file, results, [2])
    eigen_rows = _msc_eigenvalue_rows(eigenvalues, 3)
    _msc_write_records(file, "/NASTRAN/RESULT/SUMMARY/EIGENVALUE", eigen_rows)
    _msc_write_records(file, "/INDEX/NASTRAN/RESULT/SUMMARY/EIGENVALUE", [_MSCIndexRec(3, 0, length(eigen_rows))])
    if haskey(results, "u_static")
        disp_rows = _msc_nodal_rows_from_vector(results["id_map"], results["u_static"], 2)
        _msc_write_records(file, "/NASTRAN/RESULT/NODAL/DISPLACEMENT", disp_rows; version=1)
        _msc_write_records(file, "/INDEX/NASTRAN/RESULT/NODAL/DISPLACEMENT", [_MSCIndexRec(2, 0, length(disp_rows))])

        static_load_id = _msc_sol105_static_load_id(results)
        static_domain = Int64(2)
        zero_static = zeros(Float64, 6 * length(results["id_map"]))
        _msc_write_nodal_table(file, "/NASTRAN/RESULT/NODAL/APPLIED_LOAD", results["id_map"],
            [(static_domain, _msc_applied_load_vector(results, static_load_id))])
        _msc_write_nodal_table(file, "/NASTRAN/RESULT/NODAL/MPC_FORCE", results["id_map"],
            [(static_domain, zero_static)])
    end
    evec_rows, evec_index = _msc_eigenvector_rows(results["id_map"], mode_shapes, vector_start)
    _msc_write_records(file, "/NASTRAN/RESULT/NODAL/EIGENVECTOR", evec_rows; version=1)
    _msc_write_records(file, "/INDEX/NASTRAN/RESULT/NODAL/EIGENVECTOR", evec_index)

    zero = zeros(Float64, 6 * length(results["id_map"]))
    spc_vectors = Tuple{Int64,Vector{Float64}}[]
    haskey(results, "u_static") && push!(spc_vectors, (Int64(2), copy(zero)))
    for i in 1:length(eigenvalues)
        push!(spc_vectors, (Int64(vector_start + i - 1), copy(zero)))
    end
    _msc_write_nodal_table(file, "/NASTRAN/RESULT/NODAL/SPC_FORCE", results["id_map"], spc_vectors)
end

function export_nastran_hdf5(filename, output_dir, results; suffix=".h5")
    h5_path = joinpath(output_dir, _export_base_name(filename) * suffix)
    println("\n>>> Exporting MSC/Nastran-like compact HDF5: $h5_path")
    h5open(h5_path, "w") do file
        _msc_write_common_attrs(file, results, filename)
        _msc_write_input_tables(file, results["model"])
        sol_type = _msc_i64(get(results, "sol_type", 0))
        if sol_type == 101 || sol_type == 106
            _msc_write_sol101_results(file, results)
        elseif sol_type == 103
            _msc_write_sol103_results(file, results)
        elseif sol_type == 105
            _msc_write_sol105_results(file, results)
        else
            error("MSC/Nastran-like HDF5 export is not implemented for SOL $sol_type")
        end
    end
    println("  HDF5 exported: $h5_path")
    return h5_path
end

@inline function _internal_node_order(id_map)
    return [nid for (nid, _) in sort(collect(id_map), by=x -> x[2])]
end

function _export_clean_subcases(subcases)
    cleaned = Any[]
    for sc in subcases
        if sc isa AbstractDict
            sc_copy = deepcopy(sc)
            for key in collect(keys(sc_copy))
                startswith(string(key), "_") && delete!(sc_copy, key)
            end
            push!(cleaned, sc_copy)
        else
            push!(cleaned, deepcopy(sc))
        end
    end
    return cleaned
end

function _export_strip_private_keys!(value)
    if value isa AbstractDict
        for key in collect(keys(value))
            startswith(string(key), "_") && delete!(value, key)
        end
        for key in collect(keys(value))
            _export_strip_private_keys!(value[key])
        end
    elseif value isa AbstractVector
        for item in value
            _export_strip_private_keys!(item)
        end
    end
    return value
end

function build_optimization_export_payload(results)
    opt_payload = deepcopy(results["optimization"])
    if opt_payload isa AbstractDict && haskey(opt_payload, "model")
        delete!(opt_payload, "model")
    end
    _export_strip_private_keys!(opt_payload)

    return Dict(
        "analysis_type" => "SOL200_LITE_OPTIMIZATION",
        "forward_sol_type" => get(results, "forward_sol_type", nothing),
        "route_summary" => deepcopy(get(results, "route_summary", Dict{String,Any}())),
        "optimization" => opt_payload,
    )
end

function build_nonlinear_export_payload(subcases;
                                        diagnostics=nothing,
                                        analysis_type="SOL106_NONLINEAR_STATIC")
    exported_subcases = Any[]
    for sc in subcases
        push!(exported_subcases, Dict(
            "sid" => sc["sid"],
            "linear_solver_diagnostics" => deepcopy(get(sc, "solver_diagnostics", Dict{String,Any}())),
            "nonlinear_diagnostics" => deepcopy(get(sc, "nonlinear_diagnostics", Dict{String,Any}())),
        ))
    end

    payload = Dict(
        "analysis_type" => analysis_type,
        "subcases" => exported_subcases,
    )
    if diagnostics !== nothing
        payload["nonlinear_solver_summary"] = deepcopy(diagnostics)
    end
    return payload
end

@inline function _result_request_aliases(key::String)
    key_up = uppercase(strip(key))
    if key_up == "SPCFORCES"
        return ("SPCFORCES", "SPCFORCE")
    end
    return (key_up,)
end

@inline function _result_request_enabled_value(value)
    value === nothing && return false
    text = uppercase(strip(string(value)))
    return !(isempty(text) || text in ("NONE", "NO"))
end

@inline function subcase_result_request_enabled(sub_ctrl::AbstractDict, key::String; default_all_if_unspecified::Bool=true)
    for alias in _result_request_aliases(key)
        if haskey(sub_ctrl, alias)
            return _result_request_enabled_value(sub_ctrl[alias])
        end
    end

    if default_all_if_unspecified
        any_explicit = false
        for req in ("DISPLACEMENT", "FORCE", "STRESS", "STRAIN", "SPCFORCES")
            for alias in _result_request_aliases(req)
                if haskey(sub_ctrl, alias)
                    any_explicit = true
                    break
                end
            end
            any_explicit && break
        end
        return !any_explicit
    end

    return false
end

function append_requested_subcase_results!(global_results::AbstractDict, sc::AbstractDict, sub_ctrl::AbstractDict)
    if subcase_result_request_enabled(sub_ctrl, "DISPLACEMENT")
        append!(global_results["displacements"], sc["displacements"])
    end
    if subcase_result_request_enabled(sub_ctrl, "SPCFORCES")
        append!(global_results["spc_forces"], sc["spc_forces"])
    end
    if subcase_result_request_enabled(sub_ctrl, "FORCE")
        for k in keys(global_results["forces"])
            append!(global_results["forces"][k], get(sc["forces"], k, Any[]))
        end
        for k in keys(global_results["forces_bilin"])
            append!(global_results["forces_bilin"][k], get(sc["forces_bilin"], k, Any[]))
        end
    end
    if subcase_result_request_enabled(sub_ctrl, "STRESS")
        for k in keys(global_results["stresses"])
            append!(global_results["stresses"][k], get(sc["stresses"], k, Any[]))
        end
    end
    if subcase_result_request_enabled(sub_ctrl, "STRAIN")
        for k in keys(global_results["strains"])
            append!(global_results["strains"][k], get(sc["strains"], k, Any[]))
        end
    end
    return global_results
end

function filtered_subcase_result_payload(sc::AbstractDict, sub_ctrl::AbstractDict)
    force_enabled = subcase_result_request_enabled(sub_ctrl, "FORCE")
    stress_enabled = subcase_result_request_enabled(sub_ctrl, "STRESS")
    strain_enabled = subcase_result_request_enabled(sub_ctrl, "STRAIN")

    forces = Dict{String,Any}()
    for (k, v) in sc["forces"]
        forces[k] = force_enabled ? deepcopy(v) : Any[]
    end

    forces_bilin = Dict{String,Any}()
    for (k, v) in get(sc, "forces_bilin", Dict{String,Any}())
        forces_bilin[k] = force_enabled ? deepcopy(v) : Any[]
    end

    stresses = Dict{String,Any}()
    for (k, v) in sc["stresses"]
        stresses[k] = stress_enabled ? deepcopy(v) : Any[]
    end

    strains = Dict{String,Any}()
    for (k, v) in sc["strains"]
        strains[k] = strain_enabled ? deepcopy(v) : Any[]
    end

    return Dict(
        "displacements" => subcase_result_request_enabled(sub_ctrl, "DISPLACEMENT") ? deepcopy(sc["displacements"]) : Any[],
        "spc_forces" => subcase_result_request_enabled(sub_ctrl, "SPCFORCES") ? deepcopy(sc["spc_forces"]) : Any[],
        "forces" => forces,
        "forces_bilin" => forces_bilin,
        "stresses" => stresses,
        "strains" => strains,
    )
end

function build_buckling_export_payload(eigenvalues, mode_shapes, id_map;
                                       frequencies=nothing,
                                       mass_summary=nothing,
                                       modal_effective_mass=nothing,
                                       buckling_subcases=nothing,
                                       analysis_type="SOL105_BUCKLING",
                                       diagnostics=nothing)
    sorted_nodes = sort(collect(keys(id_map)))
    mode_count = size(mode_shapes, 2)
    modes = Any[]
    for i in 1:mode_count
        mode_data = Any[]
        for nid in sorted_nodes
            idx = id_map[nid]
            base = (idx - 1) * 6
            push!(mode_data, Dict(
                "grid_id" => nid,
                "t1" => mode_shapes[base + 1, i],
                "t2" => mode_shapes[base + 2, i],
                "t3" => mode_shapes[base + 3, i],
                "r1" => mode_shapes[base + 4, i],
                "r2" => mode_shapes[base + 5, i],
                "r3" => mode_shapes[base + 6, i],
            ))
        end

        mode_entry = Dict(
            "mode_number" => i,
            "mode_shape" => mode_data,
        )
        if frequencies !== nothing
            mode_entry["frequency_hz"] = frequencies[i]
        end
        if eigenvalues !== nothing
            mode_entry["eigenvalue"] = eigenvalues[i]
        end
        push!(modes, mode_entry)
    end

    payload = Dict(
        "analysis_type" => analysis_type,
        "grid_id_order" => sorted_nodes,
        "modes" => modes,
    )
    if eigenvalues !== nothing
        payload["eigenvalues"] = collect(eigenvalues)
    end
    if frequencies !== nothing
        payload["frequencies"] = collect(frequencies)
    end
    if mass_summary !== nothing
        payload["mass_summary"] = deepcopy(mass_summary)
    end
    if modal_effective_mass !== nothing
        payload["modal_effective_mass"] = deepcopy(modal_effective_mass)
    end
    if buckling_subcases !== nothing
        payload["subcases"] = _export_clean_subcases(buckling_subcases)
    end
    if diagnostics !== nothing
        payload["solver_diagnostics"] = deepcopy(diagnostics)
    end
    return payload
end

function build_jfem_element_tables(model, id_map)
    jfem_node_ids = sort(collect(keys(id_map)))
    pshells = model["PSHELLs"]
    pbarls  = model["PBARLs"]
    prods_m = model["PRODs"]

    # Helper to look up property by PID (handles both string and int keys)
    function find_prop(pdict, pid)
        p = get(pdict, string(pid), nothing)
        if p === nothing; p = get(pdict, pid, nothing); end
        return p
    end

    jfem_quads = Tuple{Int,Int,Vector{Int},Float32}[]   # eid, pid, nodes, thickness
    jfem_trias = Tuple{Int,Int,Vector{Int},Float32}[]
    for (id, el) in model["CSHELLs"]
        eid = _export_entry_public_id(id, el)
        if !haskey(el, "NODES"); continue; end
        nids = el["NODES"]
        if !all(n -> haskey(id_map, n), nids); continue; end
        pid = get(el, "PID", 0)
        prop = find_prop(pshells, pid)
        t = Float32(prop !== nothing ? get(prop, "T", 0.0) : 0.0)
        if length(nids) == 4
            push!(jfem_quads, (eid, pid, nids, t))
        elseif length(nids) == 3
            push!(jfem_trias, (eid, pid, nids, t))
        end
    end
    sort!(jfem_quads, by=x->x[1])
    sort!(jfem_trias, by=x->x[1])

    jfem_bars = Tuple{Int,Int,Int,Int,Float32}[]   # eid, pid, ga, gb, area
    for (id, bar) in model["CBARs"]
        eid = _export_entry_public_id(id, bar)
        if !haskey(bar, "GA"); continue; end
        ga, gb = bar["GA"], bar["GB"]
        if !haskey(id_map, ga) || !haskey(id_map, gb); continue; end
        pid = get(bar, "PID", 0)
        prop = find_prop(pbarls, pid)
        a = Float32(prop !== nothing ? get(prop, "A", 0.0) : 0.0)
        push!(jfem_bars, (eid, pid, ga, gb, a))
    end
    for (id, bar) in get(model, "CBEAMs", Dict())
        eid = _export_entry_public_id(id, bar)
        if !haskey(bar, "GA"); continue; end
        ga, gb = bar["GA"], bar["GB"]
        if !haskey(id_map, ga) || !haskey(id_map, gb); continue; end
        pid = get(bar, "PID", 0)
        prop = find_prop(pbarls, pid)
        a = Float32(prop !== nothing ? get(prop, "A", 0.0) : 0.0)
        push!(jfem_bars, (eid, pid, ga, gb, a))
    end
    sort!(jfem_bars, by=x->x[1])

    jfem_rods = Tuple{Int,Int,Int,Int,Float32}[]   # eid, pid, ga, gb, area
    for (id, rod) in model["CRODs"]
        eid = _export_entry_public_id(id, rod)
        if !haskey(rod, "GA"); continue; end
        ga, gb = rod["GA"], rod["GB"]
        if !haskey(id_map, ga) || !haskey(id_map, gb); continue; end
        pid = get(rod, "PID", 0)
        prop = find_prop(prods_m, pid)
        a = Float32(prop !== nothing ? get(prop, "A", 0.0) : 0.0)
        push!(jfem_rods, (eid, pid, ga, gb, a))
    end
    sort!(jfem_rods, by=x->x[1])

    # Solid elements: CTETRA, CHEXA, CPENTA
    jfem_tetras  = Tuple{Int,Int,Vector{Int}}[]   # eid, pid, nodes(4)
    jfem_hexas   = Tuple{Int,Int,Vector{Int}}[]   # eid, pid, nodes(8)
    jfem_pentas  = Tuple{Int,Int,Vector{Int}}[]   # eid, pid, nodes(6)
    for (id, el) in get(model, "CSOLIDs", Dict())
        if !haskey(el, "NODES"); continue; end
        eid = _export_entry_public_id(id, el)
        nids = el["NODES"]
        if !all(n -> haskey(id_map, n), nids); continue; end
        pid = get(el, "PID", 0)
        nn = length(nids)
        if nn == 4;     push!(jfem_tetras, (eid, pid, nids))
        elseif nn == 8; push!(jfem_hexas,  (eid, pid, nids))
        elseif nn == 6; push!(jfem_pentas, (eid, pid, nids))
        end
    end
    sort!(jfem_tetras, by=x->x[1])
    sort!(jfem_hexas,  by=x->x[1])
    sort!(jfem_pentas, by=x->x[1])

    return jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas
end

# --- v3 extension: constraint tables and per-subcase load data ---

# Coordinate transform for force/moment direction vectors (duplicated from solver/helpers.jl to avoid cross-module dependency)
function _export_coord_transform(model, cid, vec)
    if cid == 0; return vec; end
    if !haskey(model["CORDs"], string(cid)); return vec; end
    cord = model["CORDs"][string(cid)]
    R = hcat(cord["U"], cord["V"], cord["W"])
    return R * vec
end

function build_jfem_constraint_tables(model, id_map)
    # --- CELAS1 springs ---
    jfem_celas = Tuple{Int,Int,Int,Int,Int,Float32}[]   # eid, g1, c1, g2, c2, stiffness
    pelases = get(model, "PELASs", Dict())
    for (id, el) in get(model, "CELASs", Dict())
        eid = _export_entry_public_id(id, el)
        g1 = get(el, "G1", 0); c1 = get(el, "C1", 0)
        g2 = get(el, "G2", 0); c2 = get(el, "C2", 0)
        pid = get(el, "PID", 0)
        pelas = get(pelases, string(pid), nothing)
        if pelas === nothing; pelas = get(pelases, pid, nothing); end
        K_stiff = Float32(pelas !== nothing ? get(pelas, "K", 0.0) : 0.0)
        push!(jfem_celas, (eid, g1, c1, g2, c2, K_stiff))
    end
    sort!(jfem_celas, by=x->x[1])

    # --- RBE2 rigid body elements ---
    jfem_rbe2s = []   # (eid, gn, cm, slave_nids::Vector{Int})
    for (id, rbe) in get(model, "RBE2s", Dict())
        eid = _export_entry_public_id(id, rbe)
        gn = rbe["GN"]; cm = Int(rbe["CM"])
        slaves = Int.(rbe["GM"])
        push!(jfem_rbe2s, (eid=eid, gn=gn, cm=cm, slaves=slaves))
    end
    sort!(jfem_rbe2s, by=x->x.eid)

    # --- RBE3 interpolation elements ---
    jfem_rbe3s = []   # (eid, refgrid, refc, dep_grids::Vector{Int})
    for (id, rbe) in get(model, "RBE3s", Dict())
        eid = _export_entry_public_id(id, rbe)
        refgrid = rbe["REFGRID"]; refc = Int(rbe["REFC"])
        # Collect all independent grids from weight groups
        wt_groups = get(rbe, "WT_GROUPS", [])
        deps = Int[]
        for group in wt_groups
            grids_raw = group isa AbstractDict ? group["grids"] : group.grids
            append!(deps, Int.(grids_raw))
        end
        push!(jfem_rbe3s, (eid=eid, refgrid=refgrid, refc=refc, deps=deps))
    end
    sort!(jfem_rbe3s, by=x->x.eid)

    return jfem_celas, jfem_rbe2s, jfem_rbe3s
end

function collect_spc_data(model, spc_id)
    # Returns Dict{Int, Int} mapping nid → dof_mask (e.g., 123456)
    spc_nodes = Dict{Int, Set{Int}}()
    if isnothing(spc_id); return Dict{Int,Int}(); end

    # Resolve SPCADD
    sets = Set{Int}()
    sid = Int(spc_id)
    if haskey(model["SPCADDs"], sid)
        union!(sets, model["SPCADDs"][sid])
    else
        push!(sets, sid)
    end

    # Collect SPC1 entries matching the set
    for spc in model["SPC1s"]
        if Int(spc["SID"]) in sets
            for n in spc["NODES"]
                if !haskey(spc_nodes, n); spc_nodes[n] = Set{Int}(); end
                for ch in spc["C"]
                    if isdigit(ch); push!(spc_nodes[n], parse(Int, string(ch))); end
                end
            end
        end
    end

    # Convert to integer mask: Set{1,2,3} → 123
    result = Dict{Int,Int}()
    for (nid, dofs) in spc_nodes
        mask = 0
        for d in sort(collect(dofs))
            mask = mask * 10 + d
        end
        result[nid] = mask
    end
    return result
end

function _collect_point_loads_recursive(model, sid, scale, forces_acc, moments_acc)
    # Collect FORCE cards
    for frc in get(model, "FORCEs", [])
        if Int(frc["SID"]) == sid
            gid = frc["GID"]
            global_dir = _export_coord_transform(model, Int(frc["CID"]), frc["Dir"])
            fvec = global_dir .* (frc["Mag"] * scale)
            if !haskey(forces_acc, gid); forces_acc[gid] = zeros(3); end
            forces_acc[gid] .+= fvec
        end
    end

    # Collect MOMENT cards
    for mom in get(model, "MOMENTs", [])
        if Int(mom["SID"]) == sid
            gid = mom["GID"]
            global_dir = _export_coord_transform(model, Int(mom["CID"]), mom["Dir"])
            mvec = global_dir .* (mom["Mag"] * scale)
            if !haskey(moments_acc, gid); moments_acc[gid] = zeros(3); end
            moments_acc[gid] .+= mvec
        end
    end

    # Recurse through LOAD combos
    for c in get(model, "LOAD_COMBOS", [])
        if Int(c["SID"]) == sid
            for sub in c["COMPS"]
                _collect_point_loads_recursive(model, Int(sub["LID"]), scale * c["S"] * sub["S"], forces_acc, moments_acc)
            end
        end
    end
end

function collect_point_loads(model, load_id)
    forces_acc = Dict{Int, Vector{Float64}}()
    moments_acc = Dict{Int, Vector{Float64}}()
    if isnothing(load_id); return forces_acc, moments_acc; end
    _collect_point_loads_recursive(model, Int(load_id), 1.0, forces_acc, moments_acc)
    return forces_acc, moments_acc
end

function collect_jfem_subcase_data(u, sub_res, id_map, jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas; model=nothing, spc_id=nothing, load_id=nothing)
    safe_f32(x) = (v = Float32(x); isnan(v) || isinf(v) ? Float32(0) : v)
    nNodes_jfem = length(jfem_node_ids)
    nQuads_jfem = length(jfem_quads)
    nTrias_jfem = length(jfem_trias)
    nBars_jfem  = length(jfem_bars)
    nRods_jfem  = length(jfem_rods)

    # Displacements: 6 per node in jfem_node_ids order
    disp_jfem = Vector{Float32}(undef, nNodes_jfem * 6)
    for (i, nid) in enumerate(jfem_node_ids)
        idx = id_map[nid]
        for k in 1:6
            disp_jfem[(i-1)*6+k] = safe_f32(u[(idx-1)*6+k])
        end
    end

    # Shell results: 7 per shell (fx, fy, fxy, mx, my, mxy, vonmises)
    nShells = nQuads_jfem + nTrias_jfem
    shell_jfem = zeros(Float32, nShells * 7)
    shell_force_map = Dict{Int,Any}()
    for f in sub_res["forces"]["quad4"]; shell_force_map[f["eid"]] = f; end
    for f in sub_res["forces"]["tria3"]; shell_force_map[f["eid"]] = f; end
    shell_stress_map = Dict{Int,Any}()
    for s in sub_res["stresses"]["quad4"]; shell_stress_map[s["eid"]] = s; end
    for s in sub_res["stresses"]["tria3"]; shell_stress_map[s["eid"]] = s; end

    for (i, (eid, _, _, _)) in enumerate(jfem_quads)
        base = (i-1) * 7
        f = get(shell_force_map, eid, nothing)
        s = get(shell_stress_map, eid, nothing)
        if f !== nothing
            shell_jfem[base+1] = safe_f32(f["fx"]);  shell_jfem[base+2] = safe_f32(f["fy"])
            shell_jfem[base+3] = safe_f32(f["fxy"]); shell_jfem[base+4] = safe_f32(f["mx"])
            shell_jfem[base+5] = safe_f32(f["my"]);  shell_jfem[base+6] = safe_f32(f["mxy"])
        end
        if s !== nothing
            vm = max(s["z1"]["von_mises"], s["z2"]["von_mises"])
            shell_jfem[base+7] = safe_f32(vm)
        end
    end
    for (i, (eid, _, _, _)) in enumerate(jfem_trias)
        base = (nQuads_jfem + i - 1) * 7
        f = get(shell_force_map, eid, nothing)
        s = get(shell_stress_map, eid, nothing)
        if f !== nothing
            shell_jfem[base+1] = safe_f32(f["fx"]);  shell_jfem[base+2] = safe_f32(f["fy"])
            shell_jfem[base+3] = safe_f32(f["fxy"]); shell_jfem[base+4] = safe_f32(f["mx"])
            shell_jfem[base+5] = safe_f32(f["my"]);  shell_jfem[base+6] = safe_f32(f["mxy"])
        end
        if s !== nothing
            vm = max(s["z1"]["von_mises"], s["z2"]["von_mises"])
            shell_jfem[base+7] = safe_f32(vm)
        end
    end

    # Bar results: 7 per bar (axial, shear_1, shear_2, torque, moment_a1, moment_a2, bar_vonmises)
    bar_jfem = zeros(Float32, nBars_jfem * 7)
    bar_force_map = Dict{Int,Any}()
    for f in sub_res["forces"]["cbar"]; bar_force_map[f["eid"]] = f; end
    bar_stress_map = Dict{Int,Any}()
    for s in sub_res["stresses"]["cbar"]; bar_stress_map[s["eid"]] = s; end

    for (i, (eid, _, _, _, _)) in enumerate(jfem_bars)
        base = (i-1) * 7
        f = get(bar_force_map, eid, nothing)
        s = get(bar_stress_map, eid, nothing)
        if f !== nothing
            bar_jfem[base+1] = safe_f32(f["axial"]);    bar_jfem[base+2] = safe_f32(f["shear_1"])
            bar_jfem[base+3] = safe_f32(f["shear_2"]);  bar_jfem[base+4] = safe_f32(f["torque"])
            bar_jfem[base+5] = safe_f32(f["moment_a1"]); bar_jfem[base+6] = safe_f32(f["moment_a2"])
        end
        if s !== nothing
            vm = abs(s["axial"])
            for pk in ["p1","p2","p3","p4"]
                vm = max(vm, abs(get(s["end_a"], pk, 0.0)), abs(get(s["end_b"], pk, 0.0)))
            end
            bar_jfem[base+7] = safe_f32(vm)
        end
    end

    # Rod results: 2 per rod (axial, torque)
    rod_jfem = zeros(Float32, nRods_jfem * 2)
    rod_force_map = Dict{Int,Any}()
    for f in sub_res["forces"]["crod"]; rod_force_map[f["eid"]] = f; end

    for (i, (eid, _, _, _, _)) in enumerate(jfem_rods)
        base = (i-1) * 2
        f = get(rod_force_map, eid, nothing)
        if f !== nothing
            rod_jfem[base+1] = safe_f32(f["axial"]); rod_jfem[base+2] = safe_f32(f["torque"])
        end
    end

    # Solid results: 1 value per solid element (von_mises)
    nSolids = length(jfem_tetras) + length(jfem_hexas) + length(jfem_pentas)
    solid_jfem = zeros(Float32, nSolids)
    solid_stress_map = Dict{Int,Any}()
    for key in ["ctetra", "chexa", "cpenta"]
        for s in get(get(sub_res, "stresses", Dict()), key, [])
            solid_stress_map[s["eid"]] = s
        end
    end
    solid_idx = 0
    for (eid, _, _) in jfem_tetras
        solid_idx += 1
        s = get(solid_stress_map, eid, nothing)
        if s !== nothing; solid_jfem[solid_idx] = safe_f32(s["von_mises"]); end
    end
    for (eid, _, _) in jfem_hexas
        solid_idx += 1
        s = get(solid_stress_map, eid, nothing)
        if s !== nothing; solid_jfem[solid_idx] = safe_f32(s["von_mises"]); end
    end
    for (eid, _, _) in jfem_pentas
        solid_idx += 1
        s = get(solid_stress_map, eid, nothing)
        if s !== nothing; solid_jfem[solid_idx] = safe_f32(s["von_mises"]); end
    end

    # --- v3: SPC, forces, moments per subcase ---
    spc_data = Tuple{Int32, UInt32}[]
    force_data = Tuple{Int32, Float32, Float32, Float32}[]
    moment_data = Tuple{Int32, Float32, Float32, Float32}[]

    if model !== nothing
        # Collect SPC constraints
        spc_map = collect_spc_data(model, spc_id)
        for (nid, mask) in sort(collect(spc_map), by=x->x[1])
            push!(spc_data, (Int32(nid), UInt32(mask)))
        end

        # Collect point forces and moments
        forces_dict, moments_dict = collect_point_loads(model, load_id)
        for (nid, fvec) in sort(collect(forces_dict), by=x->x[1])
            if norm(fvec) > 1e-30
                push!(force_data, (Int32(nid), safe_f32(fvec[1]), safe_f32(fvec[2]), safe_f32(fvec[3])))
            end
        end
        for (nid, mvec) in sort(collect(moments_dict), by=x->x[1])
            if norm(mvec) > 1e-30
                push!(moment_data, (Int32(nid), safe_f32(mvec[1]), safe_f32(mvec[2]), safe_f32(mvec[3])))
            end
        end
    end

    return (disp=disp_jfem, shell=shell_jfem, bar=bar_jfem, rod=rod_jfem, solid=solid_jfem,
            spc=spc_data, forces=force_data, moments=moment_data)
end

function export_vtk_subcase(filename, output_dir, sid, model, id_map, X, u, stresses)
    base_name = basename(filename)
    vtk_base = replace(base_name, ".bdf" => "") * "_Subcase_$sid"
    vtk_path = joinpath(output_dir, vtk_base)
    points = zeros(3, length(id_map))
    disp = zeros(3, length(id_map))
    for (nid, idx) in id_map
         points[:, idx] = X[idx, :]
         disp[:, idx] = u[(idx-1)*6+1:(idx-1)*6+3]
    end
    cells = MeshCell[]
    data_vonmises = Float64[]
    for (id, el) in model["CSHELLs"]
        if !haskey(el, "NODES"); continue; end
        eid = _export_entry_public_id(id, el); nids = [get(id_map, n, 0) for n in el["NODES"]]; if 0 in nids; continue; end
        if length(nids) == 3
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
        elseif length(nids) == 4
            push!(cells, MeshCell(VTKCellTypes.VTK_QUAD, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
        end
    end
    for (id, bar) in model["CBARs"]
         if !haskey(bar, "GA"); continue; end
         eid = _export_entry_public_id(id, bar); nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
         push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
    end
    for (id, bar) in get(model, "CBEAMs", Dict())
         if !haskey(bar, "GA"); continue; end
         eid = _export_entry_public_id(id, bar); nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
         push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
    end
    for (id, rod) in model["CRODs"]
         if !haskey(rod, "GA"); continue; end
         eid = _export_entry_public_id(id, rod); nids = [get(id_map, rod["GA"], 0), get(id_map, rod["GB"], 0)]; if 0 in nids; continue; end
         push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids)); push!(data_vonmises, get(stresses, eid, 0.0))
    end
    # Solid elements
    for (id, el) in get(model, "CSOLIDs", Dict())
        if !haskey(el, "NODES"); continue; end
        eid = _export_entry_public_id(id, el); enids = el["NODES"]
        nids = [get(id_map, n, 0) for n in enids]; if 0 in nids; continue; end
        nn = length(nids)
        if nn == 4
            push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, nids))
        elseif nn == 8
            push!(cells, MeshCell(VTKCellTypes.VTK_HEXAHEDRON, nids))
        elseif nn == 6
            push!(cells, MeshCell(VTKCellTypes.VTK_WEDGE, nids))
        else
            continue
        end
        push!(data_vonmises, get(stresses, eid, 0.0))
    end
    if !isempty(cells)
        vtk = vtk_grid(vtk_path, points, cells)
        vtk["Displacement", VTKPointData()] = disp
        vtk["VonMises_Stress", VTKCellData()] = data_vonmises
        vtk_save(vtk)
        println("  VTK saved: $vtk_path.vtu")
    end
end

function export_json(filename, output_dir, global_results)
    json_name = _export_base_name(filename) * ".JU.JSON"
    json_path = joinpath(output_dir, json_name)
    println("\n>>> Exporting AGGREGATED JSON: $json_path")
    sanitize!(global_results)
    open(json_path, "w") do f; JSON.print(f, global_results, 4); end
end

function export_optimization_json(filename, output_dir, results)
    json_path = joinpath(output_dir, _export_base_name(filename) * ".OPTIMIZATION.JSON")
    payload = build_optimization_export_payload(results)
    sanitize!(payload)
    open(json_path, "w") do f
        JSON.print(f, payload, 2)
    end
    println("  Optimization JSON saved: $json_path")
end

function export_nonlinear_json(filename, output_dir, subcases;
                               diagnostics=nothing, analysis_type="SOL106_NONLINEAR_STATIC")
    json_name = _export_base_name(filename) * ".NONLINEAR.JSON"
    json_path = joinpath(output_dir, json_name)
    results = build_nonlinear_export_payload(subcases;
        diagnostics=diagnostics, analysis_type=analysis_type)

    println("\n>>> Exporting NONLINEAR JSON: $json_path")
    sanitize!(results)
    open(json_path, "w") do f; JSON.print(f, results, 4); end
end

function export_jfem_binary(filename, output_dir, id_map, X, jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas, jfem_subcases_data; jfem_celas=[], jfem_rbe2s=[], jfem_rbe3s=[])
    base_name = basename(filename)
    jfem_name = replace(base_name, ".bdf" => "") * ".jfem"
    jfem_path = joinpath(output_dir, jfem_name)
    nNodes_jfem = length(jfem_node_ids)
    nQuads_jfem = length(jfem_quads)
    nTrias_jfem = length(jfem_trias)
    nBars_jfem  = length(jfem_bars)
    nRods_jfem  = length(jfem_rods)
    nCelas_jfem = length(jfem_celas)
    nRBE2_jfem  = length(jfem_rbe2s)
    nRBE3_jfem  = length(jfem_rbe3s)
    println("\n>>> Exporting JFEM binary (v4): $jfem_path")
    open(jfem_path, "w") do io
        # Magic: 'JFEM'
        write(io, UInt8('J')); write(io, UInt8('F')); write(io, UInt8('E')); write(io, UInt8('M'))
        # Header (v3: extended with constraint counts)
        write(io, UInt32(4))                          # version 4
        write(io, UInt32(nNodes_jfem))
        write(io, UInt32(nQuads_jfem))
        write(io, UInt32(nTrias_jfem))
        write(io, UInt32(nBars_jfem))
        write(io, UInt32(nRods_jfem))
        write(io, UInt32(length(jfem_subcases_data)))  # nSubcases
        write(io, UInt32(nCelas_jfem))                 # v3: nCelas
        write(io, UInt32(nRBE2_jfem))                  # v3: nRBE2
        write(io, UInt32(nRBE3_jfem))                  # v3: nRBE3
        write(io, UInt32(length(jfem_tetras)))         # v4: nTetras
        write(io, UInt32(length(jfem_hexas)))          # v4: nHexas
        write(io, UInt32(length(jfem_pentas)))         # v4: nPentas

        # Node table: nid(i32), x(f32), y(f32), z(f32)
        for nid in jfem_node_ids
            idx = id_map[nid]
            write(io, Int32(nid))
            write(io, Float32(X[idx, 1])); write(io, Float32(X[idx, 2])); write(io, Float32(X[idx, 3]))
        end

        # CQUAD4 table: eid(i32), pid(i32), g1-g4(i32), thickness(f32)
        for (eid, pid, nodes, t) in jfem_quads
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end

        # CTRIA3 table: eid(i32), pid(i32), g1-g3(i32), thickness(f32)
        for (eid, pid, nodes, t) in jfem_trias
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end

        # CBAR table: eid(i32), pid(i32), ga(i32), gb(i32), area(f32)
        for (eid, pid, ga, gb, a) in jfem_bars
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end

        # CROD table: eid(i32), pid(i32), ga(i32), gb(i32), area(f32)
        for (eid, pid, ga, gb, a) in jfem_rods
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end

        # v3: CELAS table: eid(i32), g1(i32), c1(i32), g2(i32), c2(i32), stiffness(f32), pad(f32)
        for (eid, g1, c1, g2, c2, K_stiff) in jfem_celas
            write(io, Int32(eid)); write(io, Int32(g1)); write(io, Int32(c1))
            write(io, Int32(g2)); write(io, Int32(c2)); write(io, K_stiff); write(io, Float32(0))
        end

        # v3: RBE2 table (variable-length): eid(i32), gn(i32), cm(i32), nSlaves(u32), [slave_nid(i32) × nSlaves]
        for rbe in jfem_rbe2s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.gn)); write(io, Int32(rbe.cm))
            write(io, UInt32(length(rbe.slaves)))
            for s in rbe.slaves; write(io, Int32(s)); end
        end

        # v3: RBE3 table (variable-length): eid(i32), refgrid(i32), refc(i32), nDep(u32), [dep_nid(i32) × nDep]
        for rbe in jfem_rbe3s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.refgrid)); write(io, Int32(rbe.refc))
            write(io, UInt32(length(rbe.deps)))
            for d in rbe.deps; write(io, Int32(d)); end
        end

        # v4: CTETRA table: eid(i32), pid(i32), g1-g4(i32)
        for (eid, pid, nodes) in jfem_tetras
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
        end

        # v4: CHEXA table: eid(i32), pid(i32), g1-g8(i32)
        for (eid, pid, nodes) in jfem_hexas
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
        end

        # v4: CPENTA table: eid(i32), pid(i32), g1-g6(i32)
        for (eid, pid, nodes) in jfem_pentas
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
        end

        # Per-subcase data
        for sc in jfem_subcases_data
            write(io, UInt32(sc.sid))
            write(io, sc.disp)    # nNodes * 6 Float32
            write(io, sc.shell)   # (nQuads + nTrias) * 7 Float32
            write(io, sc.bar)     # nBars * 7 Float32
            write(io, sc.rod)     # nRods * 2 Float32
            write(io, sc.solid)   # nSolids Float32 (von_mises per solid)

            # v3: SPC data
            write(io, UInt32(length(sc.spc)))
            for (nid, mask) in sc.spc
                write(io, nid); write(io, mask)
            end

            # v3: Applied forces
            write(io, UInt32(length(sc.forces)))
            for (nid, fx, fy, fz) in sc.forces
                write(io, nid); write(io, fx); write(io, fy); write(io, fz)
            end

            # v3: Applied moments
            write(io, UInt32(length(sc.moments)))
            for (nid, mx, my, mz) in sc.moments
                write(io, nid); write(io, mx); write(io, my); write(io, mz)
            end
        end
    end
    nTet = length(jfem_tetras); nHex = length(jfem_hexas); nPen = length(jfem_pentas)
    println("  JFEM v4: $(nNodes_jfem) nodes, $(nQuads_jfem)Q+$(nTrias_jfem)T shells, $(nBars_jfem) bars, $(nRods_jfem) rods, $(nTet)Tet+$(nHex)Hex+$(nPen)Pen solids, $(nCelas_jfem) springs, $(length(jfem_subcases_data)) subcases")
end

function export_card_inventory(cards, output_dir, filename)
    processed_card_types = Set([
        "GRID", "CORD2R", "CORD1R", "CORD2C", "CORD2S",
        "CTRIA3", "CTRIA6", "CQUAD4", "CQUAD8", "CSHEAR", "CBAR", "CBEAM", "CROD", "CONROD", "CELAS1", "CELAS2", "CBUSH",
        "CTETRA", "CHEXA", "CPENTA", "PSOLID",
        "RBE1", "RBE2", "RBE3", "RBAR", "RSPLINE",
        "PSHELL", "PSHEAR", "PBARL", "PBAR", "PBAR*", "PBEAM", "PBEAM*", "PBEAML", "PROD", "PCOMP", "PELAS", "PBUSH",
        "MAT1", "MAT2", "MAT8", "MATT1", "TABLEM1",
        "DESVAR", "DRESP1", "DVPREL1", "DVMREL1", "DCONSTR", "DOPTPRM",
        "FORCE", "MOMENT", "PLOAD4", "PLOAD2", "PLOAD1", "PLOAD", "GRAV", "RFORCE",
        "SPC1", "SPC", "SPCADD", "MPC", "MPCADD", "LOAD",
        "CONM2", "CONM1", "CMASS1", "CMASS2", "PMASS",
        "EIGRL",
        "TEMP", "TEMPD", "DMIG",
        "PARAM"
    ])
    card_counts = Dict{String,Int}()
    unprocessed_cards = Dict{String,Int}()
    for (cname, clist) in cards
        card_counts[cname] = length(clist)
        if !(cname in processed_card_types)
            unprocessed_cards[cname] = length(clist)
        end
    end
    inv_json = Dict(
        "card_counts" => Dict(card_counts),
        "processed_card_types" => sort(collect(processed_card_types)),
        "unprocessed_cards" => Dict(unprocessed_cards)
    )
    inv_path = joinpath(output_dir, replace(basename(filename), ".bdf" => "") * ".CARDS.JSON")
    open(inv_path, "w") do f; JSON.print(f, inv_json, 4); end
    println(">>> Card inventory exported: $inv_path")
    if !isempty(unprocessed_cards)
        println("    WARNING: $(length(unprocessed_cards)) unprocessed card type(s):")
        for (cname, cnt) in sort(collect(unprocessed_cards), by=x->x[1])
            println("      $cname: $cnt")
        end
    end
end

# =============================================================================
# SOL105 BUCKLING EXPORT FUNCTIONS
# =============================================================================

function export_buckling_vtk(filename, output_dir, model, id_map, X, eigenvalues, mode_shapes)
    if isempty(eigenvalues); return; end
    base_name = replace(basename(filename), ".bdf" => "")
    n_modes = length(eigenvalues)

    points = zeros(3, length(id_map))
    for (nid, idx) in id_map
        points[:, idx] = X[idx, :]
    end

    cells = MeshCell[]
    for (id, el) in model["CSHELLs"]
        if !haskey(el, "NODES"); continue; end
        nids = [get(id_map, n, 0) for n in el["NODES"]]; if 0 in nids; continue; end
        if length(nids) == 3
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, nids))
        elseif length(nids) == 4
            push!(cells, MeshCell(VTKCellTypes.VTK_QUAD, nids))
        end
    end
    for (id, bar) in model["CBARs"]
        if !haskey(bar, "GA"); continue; end
        nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, bar) in get(model, "CBEAMs", Dict())
        if !haskey(bar, "GA"); continue; end
        nids = [get(id_map, bar["GA"], 0), get(id_map, bar["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, rod) in model["CRODs"]
        if !haskey(rod, "GA"); continue; end
        nids = [get(id_map, rod["GA"], 0), get(id_map, rod["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, rod) in get(model, "CONRODs", Dict())
        if !haskey(rod, "GA"); continue; end
        nids = [get(id_map, rod["GA"], 0), get(id_map, rod["GB"], 0)]; if 0 in nids; continue; end
        push!(cells, MeshCell(VTKCellTypes.VTK_LINE, nids))
    end
    for (id, el) in get(model, "CSOLIDs", Dict())
        if !haskey(el, "NODES"); continue; end
        enids = el["NODES"]
        nids = [get(id_map, n, 0) for n in enids]; if 0 in nids; continue; end
        nn = length(nids)
        if nn == 4;     push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, nids))
        elseif nn == 8; push!(cells, MeshCell(VTKCellTypes.VTK_HEXAHEDRON, nids))
        elseif nn == 6; push!(cells, MeshCell(VTKCellTypes.VTK_WEDGE, nids))
        end
    end

    if isempty(cells); return; end

    for m in 1:n_modes
        vtk_name = base_name * "_Buckling_Mode_$m"
        vtk_path = joinpath(output_dir, vtk_name)
        disp = zeros(3, length(id_map))
        for (nid, idx) in id_map
            base = (idx-1)*6
            disp[1, idx] = mode_shapes[base+1, m]
            disp[2, idx] = mode_shapes[base+2, m]
            disp[3, idx] = mode_shapes[base+3, m]
        end
        vtk = vtk_grid(vtk_path, points, cells)
        vtk["BucklingMode_$m", VTKPointData()] = disp
        vtk_save(vtk)
        println("  VTK buckling mode $m saved: $vtk_path.vtu (lambda=$(round(eigenvalues[m], digits=4)))")
    end
end

function export_buckling_json(filename, output_dir, eigenvalues, mode_shapes, id_map;
                              frequencies=nothing, mass_summary=nothing,
                              modal_effective_mass=nothing, buckling_subcases=nothing,
                              analysis_type="SOL105_BUCKLING",
                              diagnostics=nothing)
    if isempty(eigenvalues); return; end
    json_name = _export_base_name(filename) * ".BUCKLING.JSON"
    json_path = joinpath(output_dir, json_name)
    results = build_buckling_export_payload(eigenvalues, mode_shapes, id_map;
        frequencies=frequencies,
        mass_summary=mass_summary,
        modal_effective_mass=modal_effective_mass,
        buckling_subcases=buckling_subcases,
        analysis_type=analysis_type,
        diagnostics=diagnostics)
    sanitize!(results)
    open(json_path, "w") do f; JSON.print(f, results, 4); end
    println(">>> Buckling JSON exported: $json_path")
end

function export_jfem_buckling(filename, output_dir, id_map, X, jfem_node_ids, jfem_quads, jfem_trias, jfem_bars, jfem_rods, jfem_tetras, jfem_hexas, jfem_pentas, eigenvalues, mode_shapes; jfem_celas=[], jfem_rbe2s=[], jfem_rbe3s=[], K_global=nothing, node_R=nothing)
    if isempty(eigenvalues); return; end
    base_name = replace(basename(filename), ".bdf" => "")
    jfem_name = base_name * ".jfem"
    jfem_path = joinpath(output_dir, jfem_name)

    nNodes = length(jfem_node_ids)
    nQuads = length(jfem_quads)
    nTrias = length(jfem_trias)
    nBars  = length(jfem_bars)
    nRods  = length(jfem_rods)
    nModes = length(eigenvalues)

    println("\n>>> Exporting JFEM binary (v3 buckling): $jfem_path")
    open(jfem_path, "w") do io
        write(io, UInt8('J')); write(io, UInt8('F')); write(io, UInt8('E')); write(io, UInt8('M'))
        write(io, UInt32(3))        # version
        write(io, UInt32(nNodes))
        write(io, UInt32(nQuads))
        write(io, UInt32(nTrias))
        write(io, UInt32(nBars))
        write(io, UInt32(nRods))
        write(io, UInt32(nModes))   # nSubcases = nModes
        write(io, UInt32(length(jfem_celas)))
        write(io, UInt32(length(jfem_rbe2s)))
        write(io, UInt32(length(jfem_rbe3s)))

        # Node table: nid(i32), x(f32), y(f32), z(f32)
        for nid in jfem_node_ids
            idx = id_map[nid]
            write(io, Int32(nid))
            write(io, Float32(X[idx,1])); write(io, Float32(X[idx,2])); write(io, Float32(X[idx,3]))
        end

        # Element connectivity (same format as v2 JFEM)
        for (eid, pid, nodes, t) in jfem_quads
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end
        for (eid, pid, nodes, t) in jfem_trias
            write(io, Int32(eid)); write(io, Int32(pid))
            for n in nodes; write(io, Int32(n)); end
            write(io, t)
        end
        for (eid, pid, ga, gb, a) in jfem_bars
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end
        for (eid, pid, ga, gb, a) in jfem_rods
            write(io, Int32(eid)); write(io, Int32(pid)); write(io, Int32(ga)); write(io, Int32(gb))
            write(io, a)
        end

        # Constraints (same format as v3 JFEM)
        for (eid, g1, c1, g2, c2, K_stiff) in jfem_celas
            write(io, Int32(eid)); write(io, Int32(g1)); write(io, Int32(c1))
            write(io, Int32(g2)); write(io, Int32(c2)); write(io, K_stiff); write(io, Float32(0))
        end
        for rbe in jfem_rbe2s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.gn)); write(io, Int32(rbe.cm))
            write(io, UInt32(length(rbe.slaves)))
            for s in rbe.slaves; write(io, Int32(s)); end
        end
        for rbe in jfem_rbe3s
            write(io, Int32(rbe.eid)); write(io, Int32(rbe.refgrid)); write(io, Int32(rbe.refc))
            write(io, UInt32(length(rbe.deps)))
            for d in rbe.deps; write(io, Int32(d)); end
        end

        # Build nid→jfem_index map for fast lookup
        nid_to_jidx = Dict{Int,Int}()
        for (ji, nid) in enumerate(jfem_node_ids)
            nid_to_jidx[nid] = ji
        end

        # Per-mode data (stored as subcases, matching v3 static format)
        for m in 1:nModes
            # Subcase ID
            write(io, UInt32(m))

            # Displacements: nNodes × 6 Float32
            # Also build per-node displacement magnitude array for element results
            node_disp_mag = Vector{Float64}(undef, nNodes)
            phi = mode_shapes[:, m]
            for (ji, nid) in enumerate(jfem_node_ids)
                idx = id_map[nid]; base = (idx-1)*6
                tx = phi[base+1]
                ty = phi[base+2]
                tz = phi[base+3]
                node_disp_mag[ji] = sqrt(tx*tx + ty*ty + tz*tz)
                for d in 1:6; write(io, Float32(phi[base+d])); end
            end

            # Compute nodal strain energy: NSE_i = 0.5 * phi_i . (K*phi)_i.
            # Buckling mode vectors are exported in global components, while
            # K_eig is assembled in analysis/node coordinate components. When
            # node_R is available, rotate the mode back for the energy proxy.
            node_se = zeros(Float64, nNodes)
            if !isnothing(K_global)
                phi_energy = phi
                if !isnothing(node_R)
                    phi_energy = similar(phi)
                    for (nid, idx) in id_map
                        base = (idx - 1) * 6
                        R = node_R[idx]
                        phi_energy[base+1:base+3] = R' * phi[base+1:base+3]
                        phi_energy[base+4:base+6] = R' * phi[base+4:base+6]
                    end
                end
                f = K_global * phi_energy  # sparse matrix-vector product
                for (ji, nid) in enumerate(jfem_node_ids)
                    idx = id_map[nid]; base = (idx-1)*6
                    se = 0.0
                    for d in 1:6
                        se += phi_energy[base+d] * f[base+d]
                    end
                    node_se[ji] = 0.5 * abs(se)  # abs to avoid tiny negatives from numerics
                end
            end

            # Shell results: (nQuads + nTrias) × 7 Float32
            # Slot 0: strain energy, slots 1-5: zeros, slot 6: displacement magnitude
            for (eid, pid, nodes, t) in jfem_quads
                mag_sum = 0.0; se_sum = 0.0
                for n in nodes
                    ji = get(nid_to_jidx, n, 0)
                    if ji > 0; mag_sum += node_disp_mag[ji]; se_sum += node_se[ji]; end
                end
                nn = length(nodes)
                write(io, Float32(se_sum / nn))  # slot 0: strain energy
                for _ in 1:5; write(io, Float32(0.0)); end  # slots 1-5: zeros
                write(io, Float32(mag_sum / nn))  # slot 6: displacement magnitude
            end
            for (eid, pid, nodes, t) in jfem_trias
                mag_sum = 0.0; se_sum = 0.0
                for n in nodes
                    ji = get(nid_to_jidx, n, 0)
                    if ji > 0; mag_sum += node_disp_mag[ji]; se_sum += node_se[ji]; end
                end
                nn = length(nodes)
                write(io, Float32(se_sum / nn))
                for _ in 1:5; write(io, Float32(0.0)); end
                write(io, Float32(mag_sum / nn))
            end

            # Bar results: nBars × 7 Float32
            # Slot 0: strain energy, slots 1-5: zeros, slot 6: displacement magnitude
            for (eid, pid, ga, gb, a) in jfem_bars
                ja = get(nid_to_jidx, ga, 0)
                jb = get(nid_to_jidx, gb, 0)
                avg_mag = 0.0; avg_se = 0.0; cnt = 0
                if ja > 0; avg_mag += node_disp_mag[ja]; avg_se += node_se[ja]; cnt += 1; end
                if jb > 0; avg_mag += node_disp_mag[jb]; avg_se += node_se[jb]; cnt += 1; end
                if cnt > 0; avg_mag /= cnt; avg_se /= cnt; end
                write(io, Float32(avg_se))  # slot 0: strain energy
                for _ in 1:5; write(io, Float32(0.0)); end
                write(io, Float32(avg_mag))  # slot 6: displacement magnitude
            end

            # Rod results: nRods × 2 Float32 (zeros for buckling)
            for _ in 1:nRods; for _ in 1:2; write(io, Float32(0.0)); end; end

            # v3: SPC data (empty list)
            write(io, UInt32(0))
            # v3: Applied forces (empty list)
            write(io, UInt32(0))
            # v3: Applied moments (empty list)
            write(io, UInt32(0))
        end

        # Eigenvalue footer: 'EVAL' marker + nModes(u32) + eigenvalues(f64)
        write(io, UInt8('E')); write(io, UInt8('V')); write(io, UInt8('A')); write(io, UInt8('L'))
        write(io, UInt32(nModes))
        for ev in eigenvalues
            write(io, Float64(ev))
        end
    end
    println("  JFEM binary exported: $jfem_path ($nModes buckling modes)")
end
