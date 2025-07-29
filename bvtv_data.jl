using XLSX, DataFrames, Cleaner, OrderedCollections, Statistics
bvtv_data_location = "data/bvtv equation_dataset.xlsx"

# Load in spreadsheet
function load_bvtv_data(filename=bvtv_data_location)
    xf = XLSX.readxlsx(filename)
    sh = xf["stata"]
    labels = sh["A1:O1"]
    data = sh["A2:O262"]
    dict = OrderedDict(labels[i] => data[:,i] for i in 1:length(labels))
    dict["timeinwork"] = Vector{Union{Float64,Missing}}(dict["timeinwork"])
    df = DataFrame(dict) |> reinfer_schema |> DataFrame
    return df
end

remove_missing(df) = filter(r -> ~ismissing(r.timeinwork), df)

function extract_work_and_rest_data(df)
    df_filter = remove_missing(df)
    df_work = filter(r -> r.timeinwork >= 0, df_filter)
    df_rest = filter(r -> r.timeinwork < 0, df_filter)
    return (df_work, df_rest)
end

extract_bvtv_timeseries(df,timescale) = (timescale*df.timeinwork, df.LCBVTV)

function load_bv_tv_timeseries(df)
    (df_work, df_rest) = extract_work_and_rest_data(df)
    t_work, fBM_work = extract_bvtv_timeseries(df_work,7)
    t_rest, fBM_rest = extract_bvtv_timeseries(df_rest,-7)
    median_work = median(df_work.LCBVTV)
    median_rest = median(df_rest.LCBVTV)
    return (t_work, fBM_work, t_rest, fBM_rest),(median_work,median_rest)
end

function extract_mature_fractured(df;filterage=true)
    df_filter = remove_missing(df)
    if filterage
        df_mature_fracture = filter(r -> (r.fracture == 1) && (r.inwork1 == 1) && (r.age > 2), df_filter)
    else
        df_mature_fracture = filter(r -> (r.fracture == 1) && (r.inwork1 == 1), df_filter)
    end
    return df_mature_fracture
end

function median_fracture_time_hitchens(df;filterage=true)
    df_mature_fracture = extract_mature_fractured(df,filterage=filterage)
    dict_horse_fracture = Dict((s,id) => 7*t for (s,id,t) in zip(df_mature_fracture.study,df_mature_fracture.studyid,df_mature_fracture.timeinwork))
    fracture_times = collect(values(dict_horse_fracture))
    med_frac_time = isempty(fracture_times) ? NaN : median(fracture_times)
    return (fracture_times,med_frac_time)
end

