using CairoMakie, Colors

colors = Makie.wong_colors()
_rest_bg = convert(HSL,colors[5])
rest_bg = HSL(_rest_bg.h,_rest_bg.s,0.9)
pretraining_bg = colorant"#C1E9DE"
_progressive_bg = convert(HSL,colors[2])
progressive_bg = HSL(_progressive_bg.h,_progressive_bg.s,0.9)
_racing_bg = convert(HSL,colors[4])
racing_bg = HSL(_racing_bg.h,_racing_bg.s,0.9)
bg_palette_programs = [rest_bg,pretraining_bg,progressive_bg,racing_bg]

function add_bg!(ax,rest_periods=[],pretraining_periods=[],progressive_periods=[],racing_periods=[])
    for (a,b) in rest_periods
        vs = vspan!(ax,a,b,color=rest_bg)
        translate!(vs,0,0,-1)
    end
    for (a,b) in pretraining_periods
        vs = vspan!(ax,a,b,color=pretraining_bg)
        translate!(vs,0,0,-1)
    end
    for (a,b) in progressive_periods
        vs = vspan!(ax,a,b,color=progressive_bg)
        translate!(vs,0,0,-1)
    end
    for (a,b) in racing_periods
        vs = vspan!(ax,a,b,color=racing_bg)
        translate!(vs,0,0,-1)
    end
    return nothing
end