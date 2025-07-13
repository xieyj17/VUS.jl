module VUS

using DataFrames, Statistics, StatsBase, Plots, ROCAnalysis, ProgressMeter, CategoricalArrays

# Note: We no longer need `using Distributed`
export generate_binned_df, calculate_roc_and_auc, get_vus, plot_vus, bootstrap_vus

# ... (generate_binned_df, calculate_roc_and_auc, and get_vus functions remain the same) ...

function generate_binned_df(df::DataFrame; num_bins::Int=100)
    df_copy = copy(df)
    breaks = quantile(df_copy.normalized_time, (0:num_bins)/num_bins)
    unique_breaks = unique(breaks)
    df_copy.time_bin = levelcode.(cut(df_copy.normalized_time, unique_breaks, allowempty=true, extend=true))
    binned_df = combine(groupby(df_copy, [:game_num, :time_bin])) do sdf
        (avg_home_WP=mean(sdf.home_WP), mode_actual_result=mode(sdf.actual_result))
    end
    bin_widths = diff(unique_breaks)
    bin_width_map = Dict(zip(1:length(bin_widths), bin_widths))
    return binned_df, bin_width_map
end

function calculate_roc_and_auc(binned_df::DataFrame, bin_index::Int)
    slice_df = filter(row -> row.time_bin == bin_index, binned_df)
    if nrow(slice_df) < 2 || length(unique(slice_df.mode_actual_result)) < 2
        return (NaN, Float64[], Float64[])
    end
    targets = slice_df[slice_df.mode_actual_result .== 1, :avg_home_WP]
    nontargets = slice_df[slice_df.mode_actual_result .== 0, :avg_home_WP]
    targets_converted = convert(Vector{Union{Missing, Float64}}, targets)
    nontargets_converted = convert(Vector{Union{Missing, Float64}}, nontargets)
    roc_data_for_auc = roc(nontargets_converted, targets_converted)
    roc_auc = auc(roc_data_for_auc)
    roc_data_for_plot = roc(targets_converted, nontargets_converted)
    tpr_values = 1 .- roc_data_for_plot.pmiss
    fpr_values = roc_data_for_plot.pfa
    return roc_auc, tpr_values, fpr_values
end

function get_vus(binned_df::DataFrame, bin_width_map::Dict)
    vus = 0.0
    max_bin_index = isempty(bin_width_map) ? 0 : maximum(keys(bin_width_map))
    auc_list = fill(NaN, max_bin_index)
    for (bin_index, bin_width) in bin_width_map
        if bin_index in binned_df.time_bin
            roc_auc, _, _ = calculate_roc_and_auc(binned_df, bin_index)
            auc_list[bin_index] = roc_auc
            if !isnan(roc_auc)
                vus += roc_auc * bin_width
            end
        end
    end
    return vus, auc_list
end


"""
    _bootstrap_iteration(original_df, game_numbers, num_bins)

Helper function to perform a single bootstrap iteration.
"""
function _bootstrap_iteration(original_df::DataFrame, game_numbers::Vector, num_bins::Int)
    boot_game_nums = sample(game_numbers, length(game_numbers), replace=true)
    boot_df = filter(row -> row.game_num in boot_game_nums, original_df)
    binned_boot_df, bin_width_map = generate_binned_df(boot_df, num_bins=num_bins)
    boot_vus, _ = get_vus(binned_boot_df, bin_width_map)
    return boot_vus
end

"""
    bootstrap_vus(original_df; n_bootstraps=1000, alpha=0.05, num_bins=100)

Performs bootstrapping using native multi-threading to estimate the VUS confidence interval.
"""
function bootstrap_vus(original_df::DataFrame; n_bootstraps::Int=1000, alpha::Float64=0.05, num_bins::Int=100)
    println("Starting multi-threaded bootstrapping with $n_bootstraps samples on $(Threads.nthreads()) threads...")
    game_numbers = unique(original_df.game_num)
    
    # Pre-allocate an array to store the results from each thread
    vus_results = zeros(n_bootstraps)
    
    # Use the @threads macro for a parallel for loop
    Threads.@threads for i in 1:n_bootstraps
        vus_results[i] = _bootstrap_iteration(original_df, game_numbers, num_bins)
    end
    
    # Filter out any potential NaN results before calculating percentiles
    vus_distribution = filter(!isnan, vus_results)
    
    # Calculate confidence interval
    lower_bound = percentile(vus_distribution, (alpha / 2) * 100)
    upper_bound = percentile(vus_distribution, (1 - alpha / 2) * 100)

    println("\nBootstrap analysis complete.")
    return vus_distribution, lower_bound, upper_bound
end


function plot_vus(binned_df::DataFrame; num_bins::Int=100)
    println("Generating 3D plot...")
    p = plot3d(1, linecolor=:blue, legend=false, title="ROC Curves by Time Bin")

    time_bins = sort(unique(binned_df.time_bin))

    @showprogress "Plotting bins" for bin_index in time_bins
        _, tpr_vals, fpr_vals = calculate_roc_and_auc(binned_df, bin_index)
        if !isempty(tpr_vals) && !isempty(fpr_vals)
            x_coords = fill(bin_index, length(fpr_vals))
            color_val = (bin_index - 1) / max(1, num_bins - 1)
            plot!(p, x_coords, fpr_vals, tpr_vals, linecolor=cgrad(:viridis)[color_val])
        end
    end

    x_range = range(0, num_bins, length=2)
    y_range = range(0, 1, length=2)
    surface!(p, x_range, y_range, (x, y) -> y, c=:navy, alpha=0.2)

    xlabel!("Time Bin")
    ylabel!("False Positive Rate (FPR)")
    zlabel!("True Positive Rate (TPR)")
    plot!(xlims=(0, num_bins), ylims=(0, 1), zlims=(0, 1))
    
    display(p)
end


end