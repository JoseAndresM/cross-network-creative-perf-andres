# main.py
import streamlit as st
from analysis_functions import load_tested_creatives, extract_creative_id, categorize_creative
from cncp_logic import cncp_aggregate
from autotesting_logic import autotesting_aggregate

# Streamlit app
st.title("Creative Performance Analyzer")

# User selects analysis type
analysis_type = st.sidebar.radio(
    "Choose the Analysis Type:",
    ('Standard CNCP', 'Auto-Testing CNCP')
)

# File upload section
st.sidebar.header("Upload Files")
prev_file = st.sidebar.file_uploader("Upload Previous Tested Creatives CSV", type="csv")
new_file = st.sidebar.file_uploader("Upload New Report CSV", type="csv")

# Game code input
st.sidebar.header("Game Code")
game_code = st.sidebar.text_input("Enter the 3-letter game code (e.g., CRC)")

# Target ROAS D0 input
st.sidebar.header("Target ROAS D0")
target_roas_d0 = st.sidebar.number_input("Enter the Target ROAS D0", min_value=0.0, value=0.5, step=0.1)

# main.py (continued)

# Conditional inputs based on analysis type
if analysis_type == 'Auto-Testing CNCP':
    # Target CPI input
    st.sidebar.header("Target CPI")
    target_cpi = st.sidebar.number_input("Enter the Target CPI", min_value=0.0, value=2.0, step=0.1)

# Threshold settings
st.sidebar.header("Threshold Settings")
impressions_threshold = st.sidebar.number_input("Impressions Threshold", min_value=1000, value=2000, step=100)
cost_threshold = st.sidebar.slider("Cost Threshold Multiplier", min_value=0.0, max_value=2.0, value=1.1, step=0.1)
ipm_threshold = st.sidebar.slider("IPM Threshold Multiplier", min_value=0.0, max_value=2.0, value=1.1, step=0.1)

# First-time run toggle
first_time_run = st.sidebar.checkbox("First-time run (No Previous Tested Creatives CSV)")

if new_file and game_code:
    # Step 1: Load previous and new data
    prev_data = load_tested_creatives(prev_file) if not first_time_run else pd.DataFrame(columns=['creative_id', 'Facebook', 'Google Ads', 'Google Organic Search', 'Organic', 'Snapchat', 'TikTok for Business', 'Untrusted Devices'])
    new_data = pd.read_csv(new_file)
    
    if 'creative_network' not in new_data.columns:
        st.error("The uploaded new report CSV does not contain a 'creative_network' column.")
    else:
        # Step 2: Filter out irrelevant creatives
        exclude_creative_ids = [
            'Search SearchPartners', 'Search GoogleSearch', 'Youtube YouTubeVideos',
            'Display', 'TTCC_0021_Ship Craft - Gaming App'
        ]
        new_data = new_data[~new_data['creative_network'].isin(exclude_creative_ids)]
        new_data = new_data[~new_data['creative_network'].str.startswith('TTCC')]

        # Step 3: Extract creative IDs
        new_data['creative_id'] = new_data.apply(lambda row: extract_creative_id(row['creative_network'], game_code), axis=1)
        new_data = new_data[new_data['creative_id'] != 'unknown']

        # Step 4: Apply the appropriate aggregation logic based on the analysis type
        if analysis_type == 'Standard CNCP':
            aggregated_data = cncp_aggregate(new_data, target_roas_d0)
        elif analysis_type == 'Auto-Testing CNCP':
            aggregated_data = autotesting_aggregate(new_data, target_roas_d0, target_cpi)

        # Step 5: Categorize creatives
        average_ipm = aggregated_data['IPM'].mean()
        average_cost = aggregated_data['cost'].mean()
        aggregated_data['Category'] = aggregated_data.apply(lambda row: categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold), axis=1)
        
        # Step 6: Output the overall creative performance data as CSV
        overall_output = aggregated_data.to_csv(index=False)
        st.download_button("Download Overall Creative Performance CSV", overall_output.encode('utf-8'), "Overall_Creative_Performance.csv")


