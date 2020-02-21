from datetime import date
import pandas as pd
import re

# View columns and rows
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100


# Import data
df_final = pd.read_csv('app/data/chicago_final.csv')
# df_final.columns
# df_final.info()

# Data cleaning

# Liscense description
df_final['license_description_clean'] = [re.sub("[/(),:;'-]", '', l) for l in df_final['license_description']]
df_final['license_description_clean'] = [l.replace(" ","_").lower() for l in df_final['license_description_clean']]

# 137 unique liscence descriptions - export to revise fields
# license_list = df_final['license_description_clean'].unique()
# license_list = pd.DataFrame(license_list, columns = ['license_list'])
# license_list.to_csv('app/data/license_list.csv', header=True, index=False)

# Format date columns to datetime
df_final['license_term_start_date'] = df_final['license_term_start_date'].astype('datetime64[ns]')
df_final['license_term_expiration_date'] = df_final['license_term_expiration_date'].astype('datetime64[ns]')


# Indicate if liscense is still valid or not
in_business = []
for d in df_final['license_term_expiration_date']:
    exp_date = date(2020,2,10)
    if d > exp_date:
        d=1
        in_business.append(d)
    else:
        d=0
        in_business.append(d)

df_final['in_business'] = in_business

#  Businesses in_business by neighborhood
# df_final[df_final['in_business'] == 1].groupby(df_final['neighborhood']).size().reset_index(name='count').sort_values('count', ascending=False)


# Extract years and create new column
df_final['expire_year'] = df_final['license_term_expiration_date'].astype(str).str[0:4].astype(int)
# len(df_final['account_number'].unique())

# len(df_final[df_final['license_description_clean'] == 'tobacco_sampler'])
# df_final[df_final['license_description_clean'] == 'caterers_liquor_license']

# Drop rows

# Drop drows based on license description
df_final = df_final[df_final['license_description_clean'] != 'auctioneer']
df_final = df_final[df_final['license_description_clean'] != 'broker']
df_final = df_final[df_final['license_description_clean'] != 'music_and_dance']
df_final = df_final[df_final['license_description_clean'] != 'retail_computing_center']
df_final = df_final[df_final['license_description_clean'] != 'laboratories']
df_final = df_final[df_final['license_description_clean'] != 'taxicab_twoway_dispatch_service_license']
df_final = df_final[df_final['license_description_clean'] != 'license_manager']
df_final = df_final[df_final['license_description_clean'] != 'shared_housing_unit_operator']
df_final = df_final[df_final['license_description_clean'] != 'affiliation']
df_final = df_final[df_final['license_description_clean'] != 'riverwalk_venue_liquor_license']
df_final = df_final[df_final['license_description_clean'] != 'transportation_network_provider']
df_final = df_final[df_final['license_description_clean'] != 'emerging_business']
df_final = df_final[df_final['license_description_clean'] != 'license_broker']
df_final = df_final[df_final['license_description_clean'] != 'industrial_private_event_venue']
df_final = df_final[df_final['license_description_clean'] != 'liquor_airport_pushcart_license']
df_final = df_final[df_final['license_description_clean'] != 'sports_facilities_authority_license']
df_final = df_final[df_final['license_description_clean'] != 'retail_food__seasonal_lakefront_food_establishment']

# Drop rows not in Chicago - according to original dataset
df_final = df_final[df_final['city'] == 'CHICAGO']

# Import new liscense descriptions and merge with df_final
df_license = pd.read_csv('app/data/license_description_revised.csv', encoding = "ISO-8859-1")

df_join = pd.merge(left=df_final, right=df_license, how='left', left_on='license_description_clean', right_on='original_license_data')

df = df_join[['account_number', 'doing_business_as_name','license_general_category','license_revised','license_term_start_date','license_term_expiration_date', 'neighborhood', 'in_business']]

df.to_csv('app/data/chicago_final_v2.csv', header=True, index=False)

# Test csv
df_test = pd.read_csv('app/data/chicago_final_v2.csv')

df_test.info()
