The Business Predictor

This project uses Chicago business data obtained from the City of Chicago website (updated by the website and downloaded for this project on February 10, 2020).  Each entry in the dataset is a unique license issued, renewed, cancelled, etc. for a respective business.

The project set out to measure the probability of success in three years for a particular business given its license type, number of years the business has held the license, and neighborhood.  Below is a description of how each variable was derived.

The initial dataset was 996,513 entries and the final dataset was condensed to 106,911.

Below are sources for further information about the content.
- Source data: https://data.cityofchicago.org/Community-Economic-Development/Business-Licenses/r5kz-chrr/data
- Data dictionary (see 'Data Fields'): https://www.johnsnowlabs.com/marketplace/chicago-business-licenses/
- Business License Guide: https://www.chicago.gov/city/en/depts/bacp/sbc/business_licensing.html
- Business License Exemptions: https://www.chicago.gov/city/en/depts/bacp/sbc/business_licenseexemption.html

### Model Variables

#### License

The license description was used to categorize the types of businesses.  About half of the entires were immediately dropped as there was a generic license description that was not sufficiently descriptive to categorize.  Some license types were combined if they were similar. Other entries were eliminated if there were too few or if there were data errors, such as a license expiration date being earlier than the start date.

#### Number of years in business

The entires were aggregated by each unique business and then by each unique license type with the dates reset using the earliest start date and latest expiration date for each license.

#### Neighborhood

A license was dropped if there was no longitude or latitude value. The longitude and latitude provided by the original dataset was used to obtained the specific neighborhoods with the Google Reverse Geocoder API.

#### Probability of success

Once the columns were finalized, any business with a start date later than the date the data was obtained three years prior was deleted (i.e. a start date before February 10, 2017). The probability of success was determined as any remaining license with an expiration date greater than the date the data was extracted (i.e. February 10, 2020).  This enables the model to predict the probability of being open three years from now.

### Model

The naive prediction results in a 17.6% probability of survival.

A logistic regression with L2 Regularization and GridSearch best parameter results was used. The model scored 0.889.
