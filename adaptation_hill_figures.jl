using CairoMakie
include("bone_adaptation_model.jl")
include("makie_functions.jl")
set_theme!(makie_theme)

ψ_vals = 0:0.01:2.5
A_OBL_vals = osteoblast_activity.(ψ_vals,A_OBL_min_default,p_optimal()[1][2])
A_OCL_vals = osteoclast_activity.(ψ_vals,A_OCL_min_default,p_optimal()[2][2])
lineheight = 8e-4
textcol = Makie.wong_colors()[3]

fig = Figure(size=(600,450))
ax = Axis(fig[1, 1],xlabel=L"$\psi_\text{tissue}$ (MPa)",ylabel="Activity (mm/day)")
lines!(ax,ψ_vals, A_OBL_vals, label=L"$A_\text{OBL}$", color=Makie.wong_colors()[1])
lines!(ax,ψ_vals, A_OCL_vals, label=L"$A_\text{OCL}$", color=Makie.wong_colors()[6])
axislegend(ax, labelsize=20)
savefig("output/bone_activities",fig)