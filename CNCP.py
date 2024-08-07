import streamlit as st
import pandas as pd
import numpy as np

# Function to load previous tested creatives
def load_tested_creatives(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return pd.DataFrame(columns=['creative_id', 'Facebook', 'Google Ads', 'Google Organic Search', 'Organic', 'Snapchat', 'TikTok for Business', 'Untrusted Devices'])

# Function to update tested creatives
def update_tested_creatives(prev_data, new_data):
    combined_data = pd.concat([prev_data, new_data]).drop_duplicates(subset=['creative_id'], keep='last').reset_index(drop=True)
    return combined_data

# Function to extract only the creative identifier (game acronym, concept number, and version number) ignoring format and localization
def extract_creative_id(name, channel, game_code):
    parts = name.split('_')
    for i in range(len(parts) - 2):
        if parts[i] == game_code and parts[i+2].startswith('V'):
            return '_'.join(parts[i:i+3])
    return '_'.join(name.split('_')[:3])

# Function to categorize creatives
def categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold):
    if row['impressions'] < impressions_threshold:
        return 'Testing'
    elif row['cost'] >= cost_threshold * average_cost and row['IPM'] > ipm_threshold * average_ipm:
        return 'High Performance'
    elif row['IPM'] >= ipm_threshold * average_ipm:
        return 'Potential Creative'
    elif row['IPM'] < average_ipm:
        return 'Low Performance'
    else:
        return 'Testing'

# Streamlit app
st.title("Creative Performance Analyzer")

# File upload section
st.sidebar.header("Upload Files")
prev_file = st.sidebar.file_uploader("Upload Previous Tested Creatives CSV", type="csv")
new_file = st.sidebar.file_uploader("Upload New Report CSV", type="csv")

# Game code input
st.sidebar.header("Game Code")
game_code = st.sidebar.text_input("Enter the 3-letter game code (e.g., CRC)")

# Threshold settings
st.sidebar.header("Threshold Settings")
impressions_threshold = st.sidebar.number_input("Impressions Threshold", min_value=1000, value=2000, step=100)
cost_threshold = st.sidebar.slider("Cost Threshold Multiplier", min_value=0.0, max_value=2.0, value=1.1, step=0.1)
ipm_threshold = st.sidebar.slider("IPM Threshold Multiplier", min_value=0.0, max_value=2.0, value=1.1, step=0.1)

# First-time run toggle
first_time_run = st.sidebar.checkbox("First-time run (No Previous Tested Creatives CSV)")

if new_file and game_code:
    prev_data = load_tested_creatives(prev_file) if not first_time_run else pd.DataFrame(columns=['creative_id', 'Facebook', 'Google Ads', 'Google Organic Search', 'Organic', 'Snapchat', 'TikTok for Business', 'Untrusted Devices'])
    new_data = pd.read_csv(new_file)
    
    if 'creative_network' not in new_data.columns:
        st.error("The uploaded new report CSV does not contain a 'creative_network' column.")
    else:
        exclude_creative_ids = [
            'Search SearchPartners', 'Search GoogleSearch', 'Youtube YouTubeVideos',
            'Display', 'TTCC_0021_Ship Craft - Gaming App'
        ]
        new_data = new_data[~new_data['creative_network'].isin(exclude_creative_ids)]
        new_data['creative_id'] = new_data.apply(lambda row: extract_creative_id(row['creative_network'], row['channel'], game_code), axis=1)
        new_data = new_data[new_data['creative_id'] != 'unknown']
        
        simplified_creatives_by_network = new_data.pivot_table(index='creative_id', columns='channel', aggfunc='size', fill_value=0)
        simplified_creatives_by_network_bool = simplified_creatives_by_network.applymap(lambda x: x > 0)
        simplified_creatives_by_network_binary = simplified_creatives_by_network_bool.astype(int)
        simplified_creatives_by_network_binary_df = simplified_creatives_by_network_binary.reset_index()
        updated_tested_creatives = update_tested_creatives(prev_data, simplified_creatives_by_network_binary_df)
        
        aggregated_data = new_data.groupby('creative_id').agg({
            'impressions': 'sum',
            'cost': 'sum',
            'installs': 'sum',
            'roas_d0': 'mean',
            'roas_d3': 'mean',
            'roas_d7': 'mean',
            'retention_rate_d1': 'mean',
            'retention_rate_d3': 'mean',
            'retention_rate_d7': 'mean',
            'lifetime_value_d0': 'mean',
            'lifetime_value_d3': 'mean',
            'lifetime_value_d7': 'mean',
            'ecpi': 'mean'
        }).reset_index()

        # Round aggregated metrics to 2 decimal places
        for column in ['roas_d0', 'roas_d3', 'roas_d7', 'retention_rate_d1', 'retention_rate_d3', 'retention_rate_d7', 'lifetime_value_d0', 'lifetime_value_d3', 'lifetime_value_d7', 'ecpi']:
            aggregated_data[column] = aggregated_data[column].round(2)
        
        aggregated_data['IPM'] = (aggregated_data['installs'] / aggregated_data['impressions']) * 1000
        aggregated_data['IPM'].replace([float('inf'), -float('inf')], 0, inplace=True)
        aggregated_data['IPM'] = aggregated_data['IPM'].round(2)
        
        Q1 = aggregated_data['IPM'].quantile(0.25)
        Q3 = aggregated_data['IPM'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        aggregated_data = aggregated_data[(aggregated_data['IPM'] >= lower_bound) & (aggregated_data['IPM'] <= upper_bound)]
        
        average_ipm = aggregated_data['IPM'].mean()
        average_cost = aggregated_data['cost'].mean()
        
        aggregated_data['Category'] = aggregated_data.apply(lambda row: categorize_creative(row, average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold), axis=1)
        
        # Add ROAS Mat. D3 column
        aggregated_data['ROAS Mat. D3'] = (aggregated_data['roas_d3'] / aggregated_data['roas_d0']).replace([float('inf'), -float('inf'), np.nan], 0).round(2)
        
        overall_output = aggregated_data.to_csv(index=False)
        
        channel_aggregated_data = new_data.groupby(['creative_id', 'channel']).agg({
            'impressions': 'sum',
            'cost': 'sum',
            'installs': 'sum',
            'roas_d0': 'mean',
            'roas_d3': 'mean',
            'roas_d7': 'mean',
            'retention_rate_d1': 'mean',
            'retention_rate_d3': 'mean',
            'retention_rate_d7': 'mean',
            'lifetime_value_d0': 'mean',
            'lifetime_value_d3': 'mean',
            'lifetime_value_d7': 'mean',
            'ecpi': 'mean'
        }).reset_index()

        for column in ['roas_d0', 'roas_d3', 'roas_d7', 'retention_rate_d1', 'retention_rate_d3', 'retention_rate_d7', 'lifetime_value_d0', 'lifetime_value_d3', 'lifetime_value_d7', 'ecpi']:
            channel_aggregated_data[column] = channel_aggregated_data[column].round(2)
        
        channel_aggregated_data['IPM'] = (channel_aggregated_data['installs'] / channel_aggregated_data['impressions']) * 1000
        channel_aggregated_data['IPM'].replace([float('inf'), -float('inf')], 0, inplace=True)
        channel_aggregated_data['IPM'] = channel_aggregated_data['IPM'].round(2)
        
        channel_average_ipm = channel_aggregated_data['IPM'].mean()
        channel_aggregated_data['Category'] = channel_aggregated_data.apply(lambda row: categorize_creative(row, channel_average_ipm, average_cost, impressions_threshold, cost_threshold, ipm_threshold), axis=1)
        
        discrepancies = []
        for creative_id in channel_aggregated_data['creative_id'].unique():
            creative_data = channel_aggregated_data[channel_aggregated_data['creative_id'] == creative_id]
            categories = creative_data['Category'].unique()
            if len(categories) > 1:
                discrepancies.append({
                    'creative_id': creative_id,
                    'networks': creative_data['channel'].tolist(),
                    'categories': creative_data['Category'].tolist()
                })
        discrepancies_df = pd.DataFrame(discrepancies)
        
        channel_output = channel_aggregated_data.to_csv(index=False)
        discrepancies_output = discrepancies_df.to_csv(index=False)
        
        st.download_button("Download Updated Tested Creatives CSV", updated_tested_creatives.to_csv(index=False).encode('utf-8'), "updated_tested_creatives.csv")
        st.download_button("Download Overall Creative Performance CSV", overall_output.encode('utf-8'), "Overall_Creative_Performance.csv")
        st.download_button("Download Channel Creative Performance CSV", channel_output.encode('utf-8'), "Channel_Creative_Performance.csv")
        st.download_button("Download Discrepancies Report CSV", discrepancies_output.encode('utf-8'), "Discrepancies_Report.csv")
