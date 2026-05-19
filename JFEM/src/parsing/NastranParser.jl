
module NastranParser

using LinearAlgebra

include("utilities.jl")
include("card_processor.jl")
include("extract_geometry.jl")
include("extract_properties.jl")
include("extract_optimization.jl")
include("extract_materials.jl")
include("extract_loads.jl")
include("extract_constraints.jl")
include("extract_elements.jl")
include("mystran_converter.jl")

end # module NastranParser
