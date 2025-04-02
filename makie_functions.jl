using CairoMakie
makie_theme = Theme(
    fontsize=20,
    size=(400,300),
    Axis = (
        spinewidth=2,
        xtickwidth=2,
        ytickwidth=2,
        xgridvisible=false,
        ygridvisible=false,
        titlesize=18,
        titlefont=:regular,
        titlegap=10
    ),
    Colorbar = (
        ticklabelsize=16,
        labelsize=16,
    ),
    Legend = (
        labelsize=14, 
        titlesize=14
    ),
    Label = (
        font = :bold,
        halign = :left
    )
)

function savefig(filename,fig)
    save(filename*".svg",fig)
    save(filename*".pdf",fig)
    save(filename*".png",fig)
end
