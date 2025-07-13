module VUS

using DataFrames, Statistics, StatsBase, Plots, ROCAnalysis, ProgressMeter, CategoricalArrays

export generate_binned_df, calculate_roc_and_auc, get_vus, plot_vus

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

    # --- FINAL FIX ---
    # 1. Calculate AUC with swapped arguments to get the correct score > 0.5
    roc_data_for_auc = roc(nontargets_converted, targets_converted)
    roc_auc = auc(roc_data_for_auc)
    
    # 2. Calculate ROC data again in the correct order ONLY for plotting
    roc_data_for_plot = roc(targets_converted, nontargets_converted)
    tpr_values = 1 .- roc_data_for_plot.pmiss
    fpr_values = roc_data_for_plot.pfa
    
    return roc_auc, tpr_values, fpr_values
end

function get_vus(binned_df::DataFrame, bin_width_map::Dict)
    vus = 0.0
    
    # Create a list to store AUCs. We can size it based on the number of actual bins.
    max_bin_index = isempty(bin_width_map) ? 0 : maximum(keys(bin_width_map))
    auc_list = fill(NaN, max_bin_index)

    # --- NEW AND CORRECTED LOGIC ---
    # Iterate directly over the (key, value) pairs of the bin_width_map.
    # This is the most robust way, as it only considers bins that were actually created.
    for (bin_index, bin_width) in bin_width_map
        
        # Check if this bin has any data after aggregation.
        if bin_index in binned_df.time_bin
            roc_auc, _, _ = calculate_roc_and_auc(binned_df, bin_index)
            
            # Store the calculated AUC in our list.
            auc_list[bin_index] = roc_auc

            # If the AUC is a valid number, add its weighted value to the total VUS.
            if !isnan(roc_auc)
                vus += roc_auc * bin_width
            end
        end
    end
    
    # The VUS is the sum of each slice's area (AUC * width). No further normalization is needed.
    return vus, auc_list
end

function plot_vus(binned_df::DataFrame; num_bins::Int=100)
    println("Generating 3D plot...")
    p = plot3d(1, linecolor=:blue, legend=false, title="ROC Curves by Time Bin")

    time_bins = sort(unique(binned_df.time_bin))

    # We need to get the bin_width_map to pass to get_vus
    # We can regenerate it quickly here for the plot title. This can be refactored later.
    # Note: This assumes the binning logic is consistent. A better refactor would pass this in.
    vus_for_title, _ = get_vus(binned_df, Dict()) # Pass empty dict if map not avail here for title

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

    # Add a title with the VUS score
    # To do this, we need to recalculate the VUS score here, which is inefficient
    # but works for now. We need the bin_width_map.
    # This part requires a small refactor in your main script to pass the map.
    # For now, we'll omit the score from the title to avoid complexity.
    plot!(title="ROC Curves by Time Bin")

    display(p)
end

end