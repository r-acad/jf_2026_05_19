using Dates
using JSON
using SHA

function apply_jfem_flags!(flags_raw::AbstractString)
    applied = Dict{String,String}()
    flag_separator = occursin(";", flags_raw) ? ";" : ","
    for kv in split(flags_raw, flag_separator)
        isempty(strip(kv)) && continue
        parts = split(kv, "="; limit=2)
        length(parts) == 2 || error("invalid flag assignment: $kv")
        k = strip(parts[1])
        v = strip(parts[2])
        ENV[k] = v
        applied[k] = v
    end
    return applied
end

function _manifest_readchomp(cmd)
    try
        return strip(readchomp(cmd))
    catch e
        return "ERROR: $(sprint(showerror, e))"
    end
end

function _manifest_git_info(repo_root::AbstractString)
    commit = _manifest_readchomp(`git -C $repo_root rev-parse HEAD`)
    branch = _manifest_readchomp(`git -C $repo_root branch --show-current`)
    status = _manifest_readchomp(`git -C $repo_root status --short`)
    return Dict(
        "commit" => commit,
        "branch" => branch,
        "dirty" => !isempty(status),
        "status_short" => status,
    )
end

function _manifest_file_sha256(path::AbstractString)
    open(path, "r") do io
        return bytes2hex(sha256(io))
    end
end

function _manifest_jfem_environment()
    env = Dict{String,String}()
    for (k, v) in ENV
        startswith(k, "JFEM_") || continue
        env[string(k)] = string(v)
    end
    return Dict(k => env[k] for k in sort(collect(keys(env))))
end

function write_run_manifest(out_dir::AbstractString; repo_root::AbstractString,
                            bdf_path::AbstractString, script_path::AbstractString,
                            args=String[], applied_flags=Dict{String,String}(),
                            extra=Dict{String,Any}())
    mkpath(out_dir)
    manifest = Dict{String,Any}(
        "created_utc" => string(Dates.now(Dates.UTC)),
        "script" => abspath(script_path),
        "args" => collect(string.(args)),
        "repo_root" => abspath(repo_root),
        "git" => _manifest_git_info(repo_root),
        "julia" => Dict(
            "version" => string(VERSION),
            "threads" => Threads.nthreads(),
            "project" => get(ENV, "JULIA_PROJECT", ""),
        ),
        "bdf" => Dict(
            "path" => abspath(bdf_path),
            "sha256" => _manifest_file_sha256(bdf_path),
            "bytes" => filesize(bdf_path),
        ),
        "output_dir" => abspath(out_dir),
        "flags_raw" => get(extra, "flags_raw", ""),
        "applied_flags" => Dict(k => applied_flags[k] for k in sort(collect(keys(applied_flags)))),
        "jfem_environment" => _manifest_jfem_environment(),
        "extra" => extra,
    )
    path = joinpath(out_dir, "run_manifest.json")
    open(path, "w") do io
        JSON.print(io, manifest, 4)
        println(io)
    end
    println(">>> manifest=$path")
    return path
end
