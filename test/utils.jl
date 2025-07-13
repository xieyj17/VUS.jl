module TestUtils

using DataFrames, Statistics, Dates, Missings

export clean_nba_data, TEAM_NAME_MAP

# --- Required Columns ---
const REQUIRED_COLUMNS = [
    "game_num", "Quarter", "clock", "play_id", "home_WP", "home", "away"
]

# --- Constants ---
const QUARTER_IN_SEC = 12 * 60

const TEAM_NAME_MAP = Dict(
    "ATL" => "atlanta-hawks", "BOS" => "boston-celtics", "BRK" => "brooklyn-nets",
    "CHO" => "charlotte-hornets", "CHI" => "chicago-bulls", "CLE" => "cleveland-cavaliers",
    "DAL" => "dallas-mavericks", "DEN" => "denver-nuggets", "DET" => "detroit-pistons",
    "GSW" => "golden-state-warriors", "HOU" => "houston-rockets", "IND" => "indiana-pacers",
    "LAC" => "los-angeles-clippers", "LAL" => "los-angeles-lakers", "MEM" => "memphis-grizzlies",
    "MIA" => "miami-heat", "MIL" => "milwaukee-bucks", "MIN" => "minnesota-timberwolves",
    "NOP" => "new-orleans-pelicans", "NYK" => "new-york-knicks", "OKC" => "oklahoma-city-thunder",
    "ORL" => "orlando-magic", "PHI" => "philadelphia-76ers", "PHO" => "phoenix-suns",
    "POR" => "portland-trail-blazers", "SAC" => "sacramento-kings", "SAS" => "san-antonio-spurs",
    "TOR" => "toronto-raptors", "UTA" => "utah-jazz", "WAS" => "washington-wizards"
)


# --- Helper Functions ---
function time_to_seconds(time_str)
    # --- FIX ---
    # The `isa` function can only check against one type at a time.
    # We split the check into `isa(time_str, Int)` and `isa(time_str, Float64)`.
    if isa(time_str, Int) || isa(time_str, Float64)
        return float(time_str)
    end
    
    m = match(r"(\d+):(\d+)", string(time_str))
    if m !== nothing
        return parse(Int, m.captures[1]) * 60 + parse(Int, m.captures[2])
    end
    
    return parse(Float64, string(time_str))
end


time_left_in_quarter(second) = QUARTER_IN_SEC - second

function normalize_time(df::DataFrame)
    df_copy = copy(df)
    df_copy.current_quarter_seconds_passed = [time_left_in_quarter(time_to_seconds(x)) for x in df_copy.clock]
    df_copy.second_passed = (df_copy.Quarter .- 1) .* QUARTER_IN_SEC .+ df_copy.current_quarter_seconds_passed
    df_copy.normalized_time = df_copy.second_passed / (4 * QUARTER_IN_SEC)
    return df_copy
end

function actual_results(df::DataFrame)
    df_copy = copy(df)
    df_copy.play_id = tryparse.(Float64, string.(df_copy.play_id))
    dropmissing!(df_copy, :play_id)

    final_game_states = combine(groupby(df_copy, :game_num)) do sdf
        sdf[argmax(sdf.play_id), :]
    end

    final_game_states.actual_result = [row.home > row.away ? 1 : 0 for row in eachrow(final_game_states)]
    actual_result_map = Dict(zip(final_game_states.game_num, final_game_states.actual_result))
    df_copy.actual_result = [get(actual_result_map, gn, missing) for gn in df_copy.game_num]

    sort!(df_copy, [:game_num, :normalized_time])
    return df_copy
end

# --- Main Data Cleaning Function ---
function clean_nba_data(input_df::DataFrame)
    for col in REQUIRED_COLUMNS
        if !hasproperty(input_df, Symbol(col))
            error("Input data is missing a required column: '$col'. Please check your CSV file.")
        end
    end

    df = normalize_time(input_df)
    df = actual_results(df)
    df = filter(row -> row.Quarter <= 4, df)
    
    final_cols = ["game_num", "normalized_time", "home_WP", "actual_result", "home", "away"]
    df = select(df, final_cols)
    
    return unique(df)
end

end # End of TestUtils module