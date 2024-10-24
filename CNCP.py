import streamlit as st
import pandas as pd
import numpy as np
import re

# Function to load previous tested creatives
def load_tested_creatives(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.DataFrame(columns=['creative_id'])

# Updated function to calculate robust z-scores with epsilon
def calculate_robust_zscore(series, epsilon=1e-6):
    median = series.median()
    mad = np.median(np.abs(series - median))
    mad = mad if mad > epsilon else epsilon  # Ensure MAD is not too small
    return (series - median) / mad

# Min-max scaling function
def min_max_scale(series, epsilon=1e-8):
    return (series - series.min()) / (series.max() - series.min() + epsilon)

# Updated function to extract the creative identifier based on the game code
def extract_creative_id(name, game_code):
    # First, try to find pattern 'game_code_[CRE]<number>_V<number>'
    pattern = rf'{game_code}_([CRE]\d+_V\d+)'
    match = re.search(pattern, name)
    if match:
        return f"{game_code}_{match.group(1)}"
    else:
        # If not found, try to find any occurrence of '[CRE]\d+_V\d+'
        pattern = r'([CRE]\d+_V\d+)'
        match = re.search(pattern, name)
        if match:
            return f"{game_code}_{match.group(1)}"
        else:
            return 'unknown'

# Function to categorize creatives
def categorize_creative(row, average_ipm, average_cost, average_roas_d0, impressions_threshold):
    if row['network_impressions'] < impressions_threshold:
        return 'Testing'
    elif row['cost'] > average_cost and row['ROAS_d0'] > average_roas_d0 and row['IPM'] > average_ipm:
        return 'High Performance'
    elif row['ROAS_d0'] > average_roas_d0 and row['IPM'] > average_ipm:
        return 'Potential Creative'
    elif row['cost'] < average_cost and row['ROAS_d0'] < average_roas_d0 and row['IPM'] < average_ipm:
        return 'Low Performance'
    else:
        return 'Average Performance'

# Streamlit app
st.title("Creative Performance Analyzer")

# File upload section
st.sidebar.header("Upload Files")
prev_file = st.sidebar.file_uploader("Upload Previous Tested Creatives CSV", type="csv")
new_file = st.sidebar.file_uploader("Upload New Report CSV", type="csv")

# Game code input
st.sidebar.header("Game Code")
game_code = st.sidebar.text_input("Enter the game code used in your creative names (e.g., DVS)")

# Target ROAS D0 input
st.sidebar.header("Target ROAS D0")
target_roas_d0 = st.sidebar.number_input("Enter the Target ROAS D0", min_value=0.0, value=0.5, step=0.1)

# Threshold settings
st.sidebar.header("Threshold Settings")
impressions_threshold = st.sidebar.number_input("Impressions Threshold", min_value=1000, value=2000, step=100)

# Weights settings
st.sidebar.header("Weights Settings")
st.sidebar.write("Adjust the weights for each metric used in the Lumina Score calculation.")

# Spend weight is fixed at +1 (positive to promote scalability)
weight_cost = 1.85  # Fixed weight

# Input fields for other weights
weight_roas_diff = st.sidebar.number_input("Weight for ROAS Difference", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
weight_roas_mat_d3 = st.sidebar.number_input("Weight for ROAS Maturation D3", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
weight_ipm = st.sidebar.number_input("Weight for IPM", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

# First-time run toggle
first_time_run = st.sidebar.checkbox("First-time run (No Previous Tested Creatives CSV)")

# Input for recently tested creatives
st.sidebar.header("Recently Tested Creatives")
recent_creatives_input = st.sidebar.text_area("Enter recently tested creatives, one per line")

if new_file and game_code:
    # Step 1: Load previous and new data
    prev_data = load_tested_creatives(prev_file) if not first_time_run else pd.DataFrame(columns=['creative_id'])
    new_data = pd.read_csv(new_file)
    
    if 'creative_network' not in new_data.columns:
        st.error("The uploaded new report CSV does not contain a 'creative_network' column.")
    else:
        # Step 2: Minimal filtering to avoid excluding desired creatives
        # Adjust filters if needed
        # exclude_creative_ids = [
        #     'Search SearchPartners', 'Search GoogleSearch', 'Youtube YouTubeVideos',
        #     'Display', 'TTCC_0021_Ship Craft - Gaming App'
        # ]
        # new_data = new_data[~new_data['creative_network'].isin(exclude_creative_ids)]
        
        # Step 3: Extract creative IDs using the updated function
        new_data['creative_id'] = new_data.apply(lambda row: extract_creative_id(str(row['creative_network']), game_code), axis=1)

        # Remove creatives with 'unknown' IDs
        new_data = new_data[new_data['creative_id'] != 'unknown']

        # Step 4: Ensure required columns exist before aggregation
        required_columns = [
            'network_impressions', 'cost', 'installs',
            'retention_rate_d1', 'retention_rate_d3', 'retention_rate_d7',
            'custom_cohorted_total_revenue_d0', 'custom_cohorted_total_revenue_d3', 'custom_cohorted_total_revenue_d7'
        ]
        missing_columns = [col for col in required_columns if col not in new_data.columns]

        if missing_columns:
            st.error(f"The uploaded CSV is missing the following columns: {', '.join(missing_columns)}")
        else:
            # Step 5: Aggregate data at the creative level
            aggregated_data = new_data.groupby('creative_id').agg({
                'network_impressions': 'sum',
                'cost': 'sum',
                'installs': 'sum',
                'retention_rate_d1': 'mean',
                'retention_rate_d3': 'mean',
                'retention_rate_d7': 'mean',
                'custom_cohorted_total_revenue_d0': 'sum',
                'custom_cohorted_total_revenue_d3': 'sum',
                'custom_cohorted_total_revenue_d7': 'sum'
            }).reset_index()

            # Step 6: Calculate LTV using custom_cohorted_total_revenue
            aggregated_data['LTV_D0'] = np.where(aggregated_data['installs'] != 0, 
                                                 aggregated_data['custom_cohorted_total_revenue_d0'] / aggregated_data['installs'], 
                                                 0)
            aggregated_data['LTV_D3'] = np.where(aggregated_data['installs'] != 0, 
                                                 (aggregated_data['custom_cohorted_total_revenue_d0'] + aggregated_data['custom_cohorted_total_revenue_d3']) / aggregated_data['installs'], 
                                                 0)
            aggregated_data['LTV_D7'] = np.where(aggregated_data['installs'] != 0, 
                                                 (aggregated_data['custom_cohorted_total_revenue_d0'] + aggregated_data['custom_cohorted_total_revenue_d3'] + aggregated_data['custom_cohorted_total_revenue_d7']) / aggregated_data['installs'], 
                                                 0)

            # Step 7: Calculate ROAS using LTV and CPI (CPI = cost / installs)
            aggregated_data['CPI'] = np.where(aggregated_data['installs'] != 0, 
                                              aggregated_data['cost'] / aggregated_data['installs'], 
                                              0)
            # Handle division by zero
            aggregated_data['ROAS_d0'] = np.where(aggregated_data['CPI'] != 0, 
                                                  aggregated_data['LTV_D0'] / aggregated_data['CPI'], 
                                                  0)
            aggregated_data['ROAS_d3'] = np.where(aggregated_data['CPI'] != 0, 
                                                  aggregated_data['LTV_D3'] / aggregated_data['CPI'], 
                                                  0)
            aggregated_data['ROAS_d7'] = np.where(aggregated_data['CPI'] != 0, 
                                                  aggregated_data['LTV_D7'] / aggregated_data['CPI'], 
                                                  0)

            # Step 8: Calculate IPM using network_impressions
            aggregated_data['IPM'] = np.where(aggregated_data['network_impressions'] != 0,
                                              (aggregated_data['installs'] / aggregated_data['network_impressions']) * 1000,
                                              0)
            aggregated_data['IPM'].replace([float('inf'), -float('inf')], 0, inplace=True)
            aggregated_data['IPM'] = aggregated_data['IPM'].round(2)

            # Step 9: Calculate ROAS diff using calculated ROAS_d0
            aggregated_data['ROAS_diff'] = aggregated_data['ROAS_d0'] - target_roas_d0

            # Step 10: Calculate ROAS Mat D3 using calculated ROAS_d3 and ROAS_d0
            aggregated_data['ROAS_Mat_D3'] = np.where(
                aggregated_data['ROAS_d0'] != 0,
                aggregated_data['ROAS_d3'] / aggregated_data['ROAS_d0'],
                0
            )
            aggregated_data['ROAS_Mat_D3'].replace([float('inf'), -float('inf'), np.nan], 0, inplace=True)
            aggregated_data['ROAS_Mat_D3'] = aggregated_data['ROAS_Mat_D3'].round(2)

            # Step 11: Handle NaN values and calculate robust z-scores
            for col in ['ROAS_Mat_D3', 'cost', 'ROAS_diff', 'IPM']:
                # Replace spaces and dots in column names
                col_name = col.replace(" ", "_").replace(".", "")
                aggregated_data[col].fillna(aggregated_data[col].median(), inplace=True)
                # Calculate and display median and MAD for debugging
                median = aggregated_data[col].median()
                mad = np.median(np.abs(aggregated_data[col] - median))
                st.write(f"Column: {col}, Median: {median}, MAD: {mad}")
                aggregated_data[f'z_{col_name}'] = calculate_robust_zscore(aggregated_data[col])

            # Step 12: Optionally cap z-scores at +/-3
            # Consider removing capping if z-scores are within a reasonable range
            # for col in ['z_ROAS_Mat_D3', 'z_cost', 'z_ROAS_diff', 'z_IPM']:
            #     aggregated_data[col] = np.clip(aggregated_data[col], -3, 3)

            # Step 13: Use weights on z-scores
            weights = {
                'z_cost': weight_cost,  # Fixed weight to promote scalability
                'z_ROAS_diff': weight_roas_diff,
                'z_ROAS_Mat_D3': weight_roas_mat_d3,
                'z_IPM': weight_ipm
            }

            # Calculate weighted sums for Lumina Score
            aggregated_data['weighted_sum'] = (
                aggregated_data['z_cost'] * weights['z_cost'] +
                aggregated_data['z_ROAS_diff'] * weights['z_ROAS_diff'] +
                aggregated_data['z_ROAS_Mat_D3'] * weights['z_ROAS_Mat_D3'] +
                aggregated_data['z_IPM'] * weights['z_IPM']
            )

            # Step 14: Normalize the weighted sum to 0-100
            min_score = aggregated_data['weighted_sum'].min()
            max_score = aggregated_data['weighted_sum'].max()
            aggregated_data['Lumina_Score'] = (aggregated_data['weighted_sum'] - min_score) / (max_score - min_score + 1e-8) * 100

            # Apply penalties
            aggregated_data.loc[aggregated_data['installs'] < 5, 'Lumina_Score'] *= 0.5
            aggregated_data.loc[aggregated_data['IPM'] < 3, 'Lumina_Score'] *= 0.85

            # Ensure Lumina_Score is between 0 and 100
            aggregated_data['Lumina_Score'] = aggregated_data['Lumina_Score'].clip(0, 100)

            # Step 15: Calculate averages
            average_ipm = aggregated_data['IPM'].mean()
            average_cost = aggregated_data['cost'].mean()
            average_roas_d0 = aggregated_data['ROAS_d0'].mean()

            # Step 16: Categorize creatives with updated function
            aggregated_data['Category'] = aggregated_data.apply(
                lambda row: categorize_creative(
                    row, average_ipm, average_cost, average_roas_d0, impressions_threshold
                ), axis=1
            )
            
            # Step 17: Sort by Lumina Score
            aggregated_data.sort_values(by='Lumina_Score', ascending=False, inplace=True)
            aggregated_data.reset_index(drop=True, inplace=True)
            aggregated_data.index += 1  # Start index from 1 for ranking

            # Step 18: Output the overall creative performance data as CSV
            overall_output = aggregated_data.to_csv(index=False)
            st.download_button("Download Overall Creative Performance CSV", overall_output.encode('utf-8'), "Overall_Creative_Performance.csv")

            # Step 19: Handle recently tested creatives if provided
            if recent_creatives_input.strip():
                recent_creatives = [line.strip() for line in recent_creatives_input.strip().split('\n') if line.strip()]
                recent_data = aggregated_data[aggregated_data['creative_id'].isin(recent_creatives)]
                if not recent_data.empty:
                    # Output the recent creatives data as CSV
                    recent_output = recent_data.to_csv(index=False)
                    st.download_button("Download Recently Tested Creatives CSV", recent_output.encode('utf-8'), "Recently_Tested_Creatives.csv")

                    # Check if any recent creatives are in the top 20 of Lumina Score
                    top_20_creatives = aggregated_data.head(20)['creative_id'].tolist()
                    top_recent_creatives = [creative for creative in recent_creatives if creative in top_20_creatives]

                    if top_recent_creatives:
                        st.write(f"**Congratulations!** The following recently tested creatives are in the top 20 Lumina Scores:")
                        for creative in top_recent_creatives:
                            st.write(f"- {creative}")
                    else:
                        st.write("None of the recently tested creatives are in the top 20 Lumina Scores.")
                else:
                    st.write("No data found for the recently tested creatives provided.")
