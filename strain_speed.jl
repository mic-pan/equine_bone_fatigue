using CairoMakie, Polynomials
include("bone_adaptation_model.jl")
include("makie_functions.jl")
set_theme!(makie_theme)

# Time to peak strain (Hitchens et al., 2018)
speed = [1.4,3.6,6.0,7.5,11.0,14.0,17.0]
stresses = [30,42,54,66,78,90,102]
E_max = 5608.3
ϵ_max = stresses./E_max
time_to_peak = [0.259,0.145,0.078,0.080,0.054,0.045,0.040]
strain_rate = ϵ_max./time_to_peak

# Change to modified stresses, using stress relationship in Morrice-West et al. (2022)
new_stresses = joint_stress.(speed)
new_ϵ_max = new_stresses./E_max
new_strain_rates = new_ϵ_max./time_to_peak

# Fit a linear and quadratic model to the data
function fit_to_data(speed,strain_rate,datalabel)
    x_powers = [s^n for s in speed, n in 0:2]
    coeffs_linear = x_powers[:,1:2]\strain_rate
    coeffs_quadratic = x_powers\strain_rate
    coeffs_quadratic_zero = x_powers[:,2:3]\strain_rate

    f_linear = Polynomial(coeffs_linear)
    f_quadratic = Polynomial(coeffs_quadratic)
    f_quadratic_zero = Polynomial(coeffs_quadratic_zero)*Polynomial([0,1])

    speeds_plot = 0.1:0.1:17.0
    colors = Makie.wong_colors()
    fig = Figure(size=(500,400))
    ax = Axis(fig[1,1], xlabel="Speed (m/s)", ylabel="Strain rate (1/s)")
    sc = scatter!(ax,speed,strain_rate,color=colors[1],label=datalabel,strokecolor=:white,strokewidth=1.5)
    translate!(sc,0,0,1)
    l1 = lines!(ax,speeds_plot,f_linear.(speeds_plot),color=colors[2],label="Linear fit")
    l2 = lines!(ax,speeds_plot,f_quadratic.(speeds_plot),color=colors[3],label="Quadratic fit")
    l3 = lines!(ax,speeds_plot,f_quadratic_zero.(speeds_plot),color=colors[4],label="Quadratic fit (zero intercept)")
    axislegend(ax, position=:lt, framevisible=false)
    
    return (f_linear,f_quadratic,f_quadratic_zero,fig)
end

(f_linear,f_quadratic,f_quadratic_zero,fig) = fit_to_data(speed,strain_rate,"Hitchens et al. (2018)")
print(f_quadratic)
savefig("output/strain_rate_fit_hitchens",fig)

(f_linear_new,f_quadratic_new,f_quadratic_zero_new,fig_new) = fit_to_data(speed,new_strain_rates,"Calculated strain rates")
print(f_quadratic_new)
savefig("output/strain_rate_fit_new",fig_new)

