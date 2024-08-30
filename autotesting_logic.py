# autotesting_logic.py

from analysis_functions import calculate_zscore, sigmoid

def autotesting_aggregate(new_data, target_roas_d0, target_cpi):
    # Adjusted column names specific to Auto-Testing CNCP
    new_data.rename(columns={
        'ecpi_all': 'CPI',
        'network_impressions': 'impressions',
        'installs': 'installs'
    }, inplace=True)

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
        'ecpi': 'mean',
        'CPI': 'mean'
    }).reset_index()

    aggregated_data['IPM'] = (aggregated_data['installs'] / aggregated_data['impressions']) * 1000
    aggregated_data['IPM'].replace([float('inf'), -float('inf')], 0, inplace=True)
    aggregated_data['IPM'] = aggregated_data['IPM'].round(2)

    aggregated_data['ROAS_diff'] = aggregated_data['roas_d0'] - target_roas_d0
    aggregated_data['CPI_diff'] = target_cpi - aggregated_data['CPI']

    aggregated_data['ROAS Mat. D3'] = (aggregated_data['roas_d3'] / aggregated_data['roas_d0']).replace([float('inf'), -float('inf'), np.nan], 0).round(2)
    aggregated_data['z_ROAS_Mat_D3'] = calculate_zscore(aggregated_data['ROAS Mat. D3'])
    aggregated_data['z_cost'] = calculate_zscore(aggregated_data['cost'])
    aggregated_data['z_ROAS_diff'] = calculate_zscore(aggregated_data['ROAS_diff'])
    aggregated_data['z_IPM'] = calculate_zscore(aggregated_data['IPM'])
    aggregated_data['z_CPI_diff'] = calculate_zscore(aggregated_data['CPI_diff'])

    # Lumina Score calculation
    valid_creatives = aggregated_data[aggregated_data['installs'] >= 5]
    valid_creatives['Lumina_Score'] = valid_creatives.apply(
        lambda row: sigmoid(np.log(np.exp(row['z_cost']) * np.exp(row['z_ROAS_diff']) * np.exp(row['z_ROAS_Mat_D3']) * np.exp(row['z_IPM']) * np.exp(row['z_CPI_diff']))), axis=1
    )
    return valid_creatives
