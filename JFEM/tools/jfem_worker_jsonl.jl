# Persistent JSONL worker for Python-driven optimization loops.
#
# Stdout is reserved for one JSON response per line. Solver output is written
# to per-case jfem_case_stdout.log files. Human-readable diagnostics go to
# stderr so Python can safely parse stdout as JSONL.

include(joinpath(@__DIR__, "testing", "run_manifest.jl"))
include(joinpath(@__DIR__, "manifest_batch_core.jl"))

const WORKER_DEFAULT_FLAGS = copy(MANIFEST_FAST_DEFAULT_FLAGS)

function _worker_usage()
    return """
    usage:
      julia --threads=auto --startup-file=no --project=JFEM JFEM/tools/jfem_worker_jsonl.jl [FLAG=val,FLAG2=val2]

    JSONL commands on stdin:
      {"command":"status"}
      {"command":"run_batch","manifest":"C:/path/cases.json"}
      {"command":"gc"}
      {"command":"quit"}
    """
end

if length(ARGS) == 1 && lowercase(strip(ARGS[1])) in ("-h", "--help", "help")
    print(_worker_usage())
    exit(0)
end
length(ARGS) > 1 && error("usage: jfem_worker_jsonl.jl [FLAG=val,FLAG2=val2]")

startup_flags = copy(WORKER_DEFAULT_FLAGS)
if length(ARGS) == 1
    merge!(startup_flags, _manifest_string_dict(ARGS[1]))
end
manifest_apply_flags!(startup_flags)

using OpenJFEM

const WORKER_SESSION_ID = string(Dates.format(Dates.now(Dates.UTC), dateformat"yyyymmddTHHMMSS"), "-pid", getpid())
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const COUNTERS = Dict{String,Int}("commands" => 0, "completed_batches" => 0, "failed_batches" => 0)

function _json_response(obj)
    JSON.print(stdout, obj)
    println(stdout)
    flush(stdout)
end

function _worker_error_response(id_value, err)
    return Dict{String,Any}(
        "id" => id_value,
        "status" => "error",
        "error" => sprint(showerror, err),
        "worker_session_id" => WORKER_SESSION_ID,
    )
end

function _handle_worker_command(cmd::AbstractDict)
    COUNTERS["commands"] += 1
    command = lowercase(strip(string(get(cmd, "command", get(cmd, "cmd", "")))))
    id_value = get(cmd, "id", COUNTERS["commands"])
    if command in ("status", "ping")
        return Dict{String,Any}(
            "id" => id_value,
            "status" => "ok",
            "worker_session_id" => WORKER_SESSION_ID,
            "pid" => getpid(),
            "threads" => Threads.nthreads(),
            "commands" => COUNTERS["commands"],
            "completed_batches" => COUNTERS["completed_batches"],
            "failed_batches" => COUNTERS["failed_batches"],
        )
    elseif command == "gc"
        GC.gc()
        return Dict{String,Any}("id" => id_value, "status" => "ok", "worker_session_id" => WORKER_SESSION_ID)
    elseif command in ("quit", "exit", "stop")
        return Dict{String,Any}("id" => id_value, "status" => "stopping", "worker_session_id" => WORKER_SESSION_ID)
    elseif command == "run_batch"
        if haskey(cmd, "manifest")
            manifest, manifest_abs = load_batch_manifest(string(cmd["manifest"]))
        elseif haskey(cmd, "manifest_data")
            raw = cmd["manifest_data"]
            raw isa AbstractDict || error("manifest_data must be an object")
            manifest = Dict{String,Any}(string(k) => v for (k, v) in raw)
            manifest_abs = nothing
        else
            error("run_batch requires manifest or manifest_data")
        end
        summary = run_batch_manifest!(manifest;
            manifest_path=manifest_abs,
            repo_root=REPO_ROOT,
            script_path=@__FILE__,
            args=[JSON.json(cmd)],
            quiet=true,
            command_extra=Dict{String,Any}(
                "worker_session_id" => WORKER_SESSION_ID,
                "worker_command_index" => COUNTERS["commands"],
            ))
        if Int(summary["failed"]) == 0
            COUNTERS["completed_batches"] += 1
            status = "ok"
        else
            COUNTERS["failed_batches"] += 1
            status = "failed"
        end
        return Dict{String,Any}(
            "id" => id_value,
            "status" => status,
            "worker_session_id" => WORKER_SESSION_ID,
            "batch_id" => summary["batch_id"],
            "completed" => summary["completed"],
            "failed" => summary["failed"],
            "total" => summary["total"],
            "total_wall_s" => summary["total_wall_s"],
            "summary_json" => summary["summary_json"],
            "summary_csv" => summary["summary_csv"],
            "output_root" => summary["output_root"],
        )
    else
        error("unknown worker command: $command")
    end
end

println(stderr, "OpenJFEM JSONL worker ready session=$WORKER_SESSION_ID pid=$(getpid()) threads=$(Threads.nthreads())")
_json_response(Dict{String,Any}(
    "status" => "ready",
    "worker_session_id" => WORKER_SESSION_ID,
    "pid" => getpid(),
    "threads" => Threads.nthreads(),
))

for raw in eachline(stdin)
    line = strip(raw)
    isempty(line) && continue
    id_value = nothing
    try
        cmd = JSON.parse(line)
        cmd isa AbstractDict || error("JSONL command must be an object")
        id_value = get(cmd, "id", nothing)
        response = _handle_worker_command(cmd)
        _json_response(response)
        lowercase(strip(string(get(cmd, "command", get(cmd, "cmd", ""))))) in ("quit", "exit", "stop") && break
    catch err
        _json_response(_worker_error_response(id_value, err))
    end
end

println(stderr, "OpenJFEM JSONL worker stopped session=$WORKER_SESSION_ID")
