from datetime import date
import numpy as np
import pandas as pd

import pickle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn_pandas import DataFrameMapper, CategoricalImputer

# View columns and rows
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# Import data
df_final = pd.read_csv('data/chicago_final_v2.csv')
df_license = pd.read_csv('data/df_license_neigh.csv')

# df_license.info()
# df_license.columns
# df_license.head()
# df_license.describe()
#
# # Check min and max dates
# df_license['start_license'].min()
# df_license['start_license'].max()
# df_license['end_license'].min()
# df_license['end_license'].max()

# Create expire year column to later delete columns
df_license['expire_year'] = df_license['end_license'].astype(str).str[0:4].astype(int)
# df_license.groupby('expire_year')['expire_year'].count().sort_values(ascending=True)

# Drop outlier year
df_license = df_license[df_license['expire_year'] != 1901]

# Drop guard_dog_service - few rows
df_license = df_license[df_license['license'] != 'guard_dog_service']

# Check start dates vs end dates
# sum(df_license['start_license'] > df_license['end_license'])
# df_license[df_license['start_license'] > df_license['end_license']]
# df_license[df_license['start_license'] == df_license['end_license']]

# Drop rows with end_license dates less than start_license dates
df_license = df_license[df_license['start_license'] <= df_license['end_license']]

# Calculate new in_business column based on combined licenses
df_license['start_license'] = df_license['start_license'].astype('datetime64[ns]')
df_license['end_license'] = df_license['end_license'].astype('datetime64[ns]')

# Remove 3 years of data
three_years = []
for d in df_license['start_license']:
    start_date = date(2017,2,10)
    if d < start_date:
        d=1
        three_years.append(d)
    else:
        d=0
        three_years.append(d)

df_license['three_years'] = three_years

# Drop columns that started business less than three years ago
df_license = df_license[df_license['three_years'] == 1]


in_business = []
for d in df_license['end_license']:
    exp_date = date(2020,2,10)
    if d > exp_date:
        d=1
        in_business.append(d)
    else:
        d=0
        in_business.append(d)

df_license['in_business'] = in_business

# Create column for years_in_business
df_license['years_in_business'] = (df_license['end_license'] - df_license['start_license'])
df_license['years_in_business'] = df_license['years_in_business'].apply(lambda x: x.days).astype(float).div(365)

df_license.columns = ['account', 'license', 'start_license', 'end_license','neighborhoods','expire_year', 'three_years','in_business', 'years_in_business']


# Categories clean
df_categories = pd.read_csv('data/license_categories.csv')
df_categories['category'] = df_categories['category'].astype(str).str.lower()
df_categories['category'] = df_categories['category'].str.replace(' ', '_')
df_categories['name'] = df_categories['name'].astype(str).str.lower()
df_categories['name'] = df_categories['name'].str.replace(' ', '_')

# Merge for license_general_category
df_merge = pd.merge(left=df_license, right=df_categories, how='left', left_on='license', right_on='name')
df_merge['category'] = df_merge['category'].astype(str).str.lower()
df_merge['category'] = df_merge['category'].replace(' ', '_')
df_merge['neighborhoods'] = df_merge['neighborhoods'].astype(str).str.lower()
df_merge['neighborhoods'] = df_merge['neighborhoods'].str.replace(' ', '_')
# df_merge.columns

# Merge category columns
# df_merge['license_joined'] = df_merge['category'].astype(str) + "_" + df_merge['license'].astype(str)


# TTS
df = df_merge[['license','neighborhoods', 'in_business','years_in_business']]

# # Count of neighborhoods
# neigh_density = df.groupby('neighborhoods')['neighborhoods'].count().sort_values(ascending=False)
# neigh_density.to_csv('data/neigh_density.csv')
#
# # Count of license density
# license_density = df.groupby('license')['license'].count().sort_values(ascending=False)
# license_density.to_csv('data/license_density.csv')
#
# # Count of license and neighborhoods
# neigh_lic_density = df.groupby(['neighborhoods','license'])['neighborhoods'].count().sort_values(ascending=False)
# neigh_lic_density.to_csv('data/neigh_lic.csv')
#
# # Count in_business by neighborhoods
# neigh_busn_density = df.groupby(['in_business','neighborhoods'])['neighborhoods'].count().sort_values(ascending=False)
# neigh_busn_density.to_csv('data/neigh_busn.csv')
#
# # Count in_business by neighborhoods
# lic_busn_density = df.groupby(['in_business','license'])['license'].count().sort_values(ascending=False)
# lic_busn_density.to_csv('data/lic_busn.csv')


# Survival chances: ~1 in 5
sum(df['in_business'] == 0)
sum(df['in_business'] == 1)
# Naive survival = 17.6%
18858/(18858+88053)

# Years in business - exploratory analysis
# df['test']=df['years_in_business'].apply(lambda x: int(x))
# years_busn = df.groupby('test')['test'].count().sort_values(ascending=False)
# years_busn

# Final dataset length - 106,911 rows
len(df)

target = 'in_business'
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


#  Mapper
mapper = DataFrameMapper([
    ('license', [CategoricalImputer(),LabelBinarizer()]),
    ('neighborhoods', [CategoricalImputer(),LabelBinarizer()]),
    (['years_in_business'],[SimpleImputer()])
    ])

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)


# Model 1 - Logistic Regression
logistic = LogisticRegression()
logistic.fit(Z_train,y_train)
logistic.score(Z_train, y_train)
logistic.score(Z_test, y_test)
# Scores: train = .88992; test = .88892

# Model 2 - Ridge Logistic Regression with CV
ridge = LogisticRegression(solver='lbfgs', penalty = 'l2', C = 1/10)
ridge = cross_val_score(ridge, Z_train, y_train, cv = 5)
print(ridge)
print(np.mean(ridge))
# ridge mean = .88853

# Model 3 -final model - Ridge Logistic with  GridSearch and CV
param_grid = {
    'solver': ['lbfgs'],
    'penalty': ['l2'],
    'C': [1, 5, 10, 20],
    'fit_intercept': [True, False],
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, verbose=1, n_jobs=-1)
grid.fit(Z_train, y_train)

grid.best_score_
grid.best_estimator_.score(Z_test, y_test)
grid.score(Z_test,y_test)
grid.best_params_

# Use best parameters from GridSearch to create model
model = LogisticRegression(C=10, fit_intercept=True, penalty='l2', solver='lbfgs')
model.fit(Z_train,y_train)
model.score(Z_train, y_train)
model.score(Z_test, y_test)
# Scores: train = .8897; test = .8890

list(zip(X_train.columns, np.exp(model.coef_[0])))

# Create pipeline
pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
pickle.dump(pipe, open('model/model.pkl', 'wb'))
del pipe
pipe = pickle.load(open('model/model.pkl', 'rb'))


# Sample test
X_train.sample().to_dict(orient='list')
new = pd.DataFrame({
    'license': 'retail_food_establishment',
    'neighborhoods': ['lake_view_east'],
    'years_in_business': 0})
type(pipe.predict(new)[0])
prediction = pipe.predict(new)[0]
pipe.predict_proba(new)[0][1]
prediction
