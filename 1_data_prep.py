from datetime import date
import pandas as pd


# Original data source: https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr/data#Export
    # February 10, 2020 - extracted and updated (996,513 entries) - CSV

# View columns and rows
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# Import data
df_all = pd.read_csv('app/data/chicago_all.csv')
df_all.columns = map(str.lower, df_all.columns)
df_all.columns = [c.replace(' ', '_') for c in df_all.columns]

# Convert dates to datetime format
df_all['application_requirements_complete'] = df_all['application_requirements_complete'].astype('datetime64[ns]')
df_all['payment_date'] = df_all['payment_date'].astype('datetime64[ns]')
df_all['license_term_start_date'] = df_all['license_term_start_date'].astype('datetime64[ns]')
df_all['license_term_expiration_date'] = df_all['license_term_expiration_date'].astype('datetime64[ns]')
df_all['license_approved_for_issuance'] = df_all['license_approved_for_issuance'].astype('datetime64[ns]')
df_all['date_issued'] = df_all['date_issued'].astype('datetime64[ns]')


# Drop rows based on condition

# Drop not IL
df_all = df_all[df_all['state'] == 'IL']

df_all.groupby('state')['state'].count()

# Drop ambiguous licenses
df_all = df_all[df_all['license_description'] != 'Limited Business License']
df_all = df_all[df_all['license_description'] != 'Regulated Business License']
pd.DataFrame(df_all.groupby('license_description')['license_id'].count().sort_values(ascending=False))

# Drop missing longitude and latitude values
df_missing_latlong = df_all.loc[df_all['latitude'].isnull(), :]
df_missing_latlong.info()
df_all = df_all.dropna(subset=['latitude'])

# Delete rows with empty license start and expiration dates
s = df_all.loc[df_all['license_term_start_date'].isnull(), :]
e = df_all.loc[df_all['license_term_expiration_date'].isnull(), :]
df_all = df_all.dropna(subset=['license_term_start_date'])
df_all = df_all.dropna(subset=['license_term_expiration_date'])

# Latiude and longitude columns: keep 4 decimal points
df_all['latitude'] = df_all['latitude'].astype(str).str[:7]
df_all['longitude'] = df_all['longitude'].astype(str).str[:8]

# Modified location values for API input
df_all['latlong'] = df_all['latitude'].astype(str) + "," + df_all['longitude'].astype(str)
df_all['latlong']

# Final row count is 500,995
df_all.info()
df_all.describe()

# Create new csv with dropped rows
df_all.to_csv('app/data/chicago_new.csv',index=False)

# Test new csv
df_new = pd.read_csv('app/data/chicago_new.csv')
df_new.info()
