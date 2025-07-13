# Debug script to identify VUS calculation issues
using VUS
using CSV
using DataFrames

# Load and process data with detailed debugging
println("=== STEP 1: Loading Data ===")
raw_data = CSV.read("nba_2019.csv", DataFrame)
println("Raw data shape: $(size(raw_data))")
println("Raw data columns: $(names(raw_data))")
println("First few rows:")
println(first(raw_data, 3))

println("\n=== STEP 2: Cleaning Data ===")
df = clean_nba_data(raw_data)
println("Cleaned data shape: $(size(df))")
println("Sample of cleaned data:")
println(first(df, 5))

println("\nData ranges:")
println("- Unique games: $(length(unique(df.game_num)))")
println("- normalized_time range: $(minimum(df.normalized_time)) to $(maximum(df.normalized_time))")
println("- home_WP range: $(minimum(df.home_WP)) to $(maximum(df.home_WP))")
println("- home_WP unique values: $(length(unique(df.home_WP)))")
println("- actual_result values: $(sort(unique(df.actual_result)))")
println("- actual_result counts: $(countmap(df.actual_result))")

println("\n=== STEP 3: Binning Data ===")
binned_df, bin_width_map = generate_binned_df(df, num_bins=10)  # Use fewer bins for debugging
println("Binned data shape: $(size(binned_df))")
println("Available time bins: $(sort(unique(binned_df.time_bin)))")
println("Bin width map: $bin_width_map")

println("\nSample of binned data:")
println(first(binned_df, 10))

println("\nBinned data ranges:")
println("- avg_home_WP range: $(minimum(binned_df.avg_home_WP)) to $(maximum(binned_df.avg_home_WP))")
println("- avg_home_WP unique values: $(length(unique(binned_df.avg_home_WP)))")
println("- mode_actual_result values: $(sort(unique(binned_df.mode_actual_result)))")
println("- mode_actual_result counts: $(countmap(binned_df.mode_actual_result))")

println("\n=== STEP 4: Testing Individual Bin ===")
# Test calculation on a specific bin
test_bin = first(sort(unique(binned_df.time_bin)))
test_slice = binned_df[binned_df.time_bin .== test_bin, :]
println("Testing bin $test_bin:")
println("- Rows in bin: $(nrow(test_slice))")
println("- y_true values: $(test_slice.mode_actual_result)")
println("- y_scores values: $(test_slice.avg_home_WP)")

if nrow(test_slice) >= 2
    y_true = test_slice.mode_actual_result
    y_scores = test_slice.avg_home_WP
    
    println("- Unique y_true: $(unique(y_true))")
    println("- Unique y_scores: $(unique(y_scores))")
    
    if length(unique(y_true)) >= 2 && length(unique(y_scores)) >= 2
        try
            # Test MLBase.roc directly
            println("- Testing MLBase.roc...")
            roc_curve = MLBase.roc(y_true, y_scores, 10)  # 10 thresholds
            println("- ROC curve length: $(length(roc_curve))")
            println("- First ROC point: $(roc_curve[1])")
            
            # Calculate simple AUC manually
            fpr = [r.fp / r.n for r in roc_curve]
            tpr = [r.tp / r.p for r in roc_curve]
            println("- FPR: $fpr")
            println("- TPR: $tpr")
            
        catch e
            println("- Error in MLBase.roc: $e")
        end
    else
        println("- Insufficient variability for ROC calculation")
    end
end

println("\n=== STEP 5: VUS Calculation ===")
vus_value, auc_list = get_vus(binned_df, bin_width_map, num_bins=10)
println("VUS: $vus_value")
println("AUC list: $auc_list")