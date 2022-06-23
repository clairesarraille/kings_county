## final project submission

Please fill out:
* Student name: Claire Sarraille
* Student pace: self paced
* Scheduled project review date/time:
* Instructor name:
* Blog post URL: https://clairesarraille.github.io/2021/08/08/kings_county_housing.html



```python
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import datetime as dt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

%matplotlib inline

```

## CRISP-DM -- CRoss Industry Standard Process for Data Mining
- Following this process ensures that the requirements and organization of formal hypothesis testing are met (broadly speaking, an iterative approach to modeling)
- Not every step was necessary, or was in scope for this particular exercise.
----------
- Business Understanding
    - Who will be  using the model
    - How will the model be used
    - How will using the model help our business
    - clarify requirements
    - What problems are in and out of scope

- Understand Data
    - What is target
    - What are predictors
    - Distribution of Data
    - How many observations - and is this a small, medium or large collection?
    - What is the quality? - What errors could be in the data, or inaccuracy?

- Data Preparation - Prepping to model
    - Missing values
    - Type conversions
    - Remove multicollinearity (correlated predictors)
    - Standardize numeric data
    - Convert categorical data to numeric via one-hot encoding

- Modeling:
    - Tune models to get the highest performance possible on our task
    - Considerations:
        - What kind of task? Classification task? Regression task?
        - Which models will we use
        - Will we use regularization?
        - How will we validate our model?
        - Loss functions?
        - What is the performance threshold for success?

- Evaluation:
    - Does the model solve business problem outlined in step 1?
    - At this point, we may want to start over at the business understanding step, now that we have a deeper understanding
    - Things we may learn at this stage:
        - Need different data
        - Need more data
        - Should be going in a different direction
        - Should use classificaion rather than regression, or vice versa
        - Use different approach

- Deployment:
    - Move the model into production
    - Set up ETL - how much of preprocessing and cleaning can be automated?


# BUSINESS UNDERSTANDING
- Who will be  using the model?
    - The hypothetical real estate agency I work for
    - Our clients are typically people trying to sell their family home - this business case is not for apartment buildings or hotel real estate
- How will the model be used?
    - It will help us understand which variables (number of bedrooms, square footage) contribute the most to a positive increase in home selling price
- How will using the model help our business?
    - It will provide the evidence we need to give data-driven advice to our clients
    - Helping our clients make decisions to maximize selling price will increase our commissions
    - We also have the option of charging our clients for a home-renovation consultation which would include access to the data from our model
- Clarify Requirements:
    - Coefficients for each independent variable: these coefficients represent the dollar amount impact on selling price associated with a one-unit increase of a given variable.
- What problems are in and out of scope?
    - In scope: How do home renovations impact the selling price of a home.
    - Out of scope: What selling price the model would predict given new data (data we haven't seen yet)

# UNDERSTAND DATA
- What is **target**
>'price'
- What are **predictors**
>'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'condition', 'grade', 'yr_built'
- Distribution of Data - See section "DISTRIBUTIONS"
- How many observations - and is this a small, medium or large collection?
    - 21,597 entries
    - According to Frank Harrell in his book Regression Modeling Strategies, you need at least 10-20 observations per predictor
    - We have many, many more observations than 20 * (6 - num covariates) = 120
    - Our dataset would be considered large
- What is the quality? - What errors could be in the data, or inaccuracy? - See section "Data Quality"

## Read in Data


```python
df = pd.read_csv('data/kc_house_data.csv')
```

## Remove Features
- view ( has been viewed, this wouldn't help us predict prices)
- Note: **sqft_above + sqft_basement = sqft_living**, according to meta-data:
    - sqft_above (is redundant -- repeats sqft_living)
    - sqft_basement (is redundant -- repeats sqft_living)
- sqft_living15 (is redundant - repeats sqft_living)
- sqft_lot15 (is redundant - repeats sqft_lot)
- yr_renovated (~18% of values are NULL - imputing (making educated guess) for those values is a bit out of scope for this project, so we'll drop this field)


```python
# Why reduce features?
# I narrowed my list of features to avoid over-fit of the training dataset
# Reducing redundancy in features increases the accuracy of the model
# Note: The brief for this project also recommended dropping these features.
# Dropping yr_renovated because there are 3,842 nulls in that field
# View is "has been viewed"

df = df.drop(['view', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'yr_renovated'], axis=1)
```


```python
# Note: sqft_living includes a finished basement, if present
# Grade is the construction quality of improvements, according to a King County grading system
# Condition is overall condition of house
df.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'condition', 'grade', 'yr_built',
           'zipcode', 'lat', 'long'],
          dtype='object')



## Data Quality

- There could be outliers - such as hundreds of bathrooms -- this wouldn't make sense considering a given home's square-footage
- Year could be in the future
- price could be a negative number
- datatype could be wrong - such as string for number of bedrooms

### Waterfront Missing Values


```python
# Waterfront is our only variable containing NULL values:
df.isnull().sum()
```




    id                0
    date              0
    price             0
    bedrooms          0
    bathrooms         0
    sqft_living       0
    sqft_lot          0
    floors            0
    waterfront     2376
    condition         0
    grade             0
    yr_built          0
    zipcode           0
    lat               0
    long              0
    dtype: int64



- I filled in NaN values for 'waterfront' with 0. Now, 'waterfront' has a 1 value if there is a view, and 0 value if there is not.
- I also cast this column as an integer type since there are no other options besides the integers 1 and 0.


```python
# Note: waterfront is our only categorical value.
# We don't need to use dummy coding or any other coding system because it's already dichotomous (1 or 0)
df['waterfront'] = df['waterfront'].fillna(0)
df['waterfront'] = df['waterfront'].astype(int)
df['waterfront'].unique()
```




    array([0, 1])



## Cast Features -- at a Different Scale for Visualization


```python
# Create lot size in acres:
df['acres_lot'] = df['sqft_lot'] / 43560
```


```python
# Create price column in millions:
df['price_millions'] = df['price'] / 1000000
```


```python
df[['price_millions','price', 'acres_lot']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_millions</th>
      <th>price</th>
      <th>acres_lot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.221900</td>
      <td>221900.0</td>
      <td>0.129706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.538000</td>
      <td>538000.0</td>
      <td>0.166253</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.180000</td>
      <td>180000.0</td>
      <td>0.229568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.604000</td>
      <td>604000.0</td>
      <td>0.114784</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.510000</td>
      <td>510000.0</td>
      <td>0.185491</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>0.360000</td>
      <td>360000.0</td>
      <td>0.025964</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>0.400000</td>
      <td>400000.0</td>
      <td>0.133448</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>0.402101</td>
      <td>402101.0</td>
      <td>0.030992</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>0.400000</td>
      <td>400000.0</td>
      <td>0.054821</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>0.325000</td>
      <td>325000.0</td>
      <td>0.024702</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 3 columns</p>
</div>



## Create New Columns from "date" - Date Sold

- The date column is the date home was sold
- If we convert this date to a numerical value, the origin would be year 0 -- and this would be difficult to interpret for our study
- Instead, I'll extract the year from the date column, and create a new feature using year_built to get age_sold (in years)


```python
# This means the datatype of the date column is an object
df.date.dtype
```




    dtype('O')




```python
df[['date']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/13/2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12/9/2014</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2/25/2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12/9/2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2/18/2015</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>21592</th>
      <td>5/21/2014</td>
    </tr>
    <tr>
      <th>21593</th>
      <td>2/23/2015</td>
    </tr>
    <tr>
      <th>21594</th>
      <td>6/23/2014</td>
    </tr>
    <tr>
      <th>21595</th>
      <td>1/16/2015</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>10/15/2014</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 1 columns</p>
</div>



### Convert to date type


```python
df['date_sold'] =  pd.to_datetime(df['date'], format='%m/%d/%Y')
```


```python
df[['date','date_sold']].sort_values(by='date_sold', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>date_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16580</th>
      <td>5/27/2015</td>
      <td>2015-05-27</td>
    </tr>
    <tr>
      <th>13040</th>
      <td>5/24/2015</td>
      <td>2015-05-24</td>
    </tr>
    <tr>
      <th>5632</th>
      <td>5/15/2015</td>
      <td>2015-05-15</td>
    </tr>
    <tr>
      <th>15797</th>
      <td>5/14/2015</td>
      <td>2015-05-14</td>
    </tr>
    <tr>
      <th>927</th>
      <td>5/14/2015</td>
      <td>2015-05-14</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7316</th>
      <td>5/2/2014</td>
      <td>2014-05-02</td>
    </tr>
    <tr>
      <th>19661</th>
      <td>5/2/2014</td>
      <td>2014-05-02</td>
    </tr>
    <tr>
      <th>6418</th>
      <td>5/2/2014</td>
      <td>2014-05-02</td>
    </tr>
    <tr>
      <th>10689</th>
      <td>5/2/2014</td>
      <td>2014-05-02</td>
    </tr>
    <tr>
      <th>4959</th>
      <td>5/2/2014</td>
      <td>2014-05-02</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 2 columns</p>
</div>



### Year Sold


```python
# Extract Year Sold
df['year_sold'] = df['date_sold'].dt.year

```


```python
df['year_sold']
```




    0        2014
    1        2014
    2        2015
    3        2014
    4        2015
             ...
    21592    2014
    21593    2015
    21594    2014
    21595    2015
    21596    2014
    Name: year_sold, Length: 21597, dtype: int64



### Age Sold in Years


```python
# Create new feature: age_sold
df['age_sold'] = df['year_sold'] - df['yr_built']
```


```python
df[['age_sold','year_sold', 'yr_built']].sort_values(by='age_sold', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_sold</th>
      <th>year_sold</th>
      <th>yr_built</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8996</th>
      <td>115</td>
      <td>2015</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>3973</th>
      <td>115</td>
      <td>2015</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>11239</th>
      <td>115</td>
      <td>2015</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>4434</th>
      <td>115</td>
      <td>2015</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>19370</th>
      <td>115</td>
      <td>2015</td>
      <td>1900</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20836</th>
      <td>-1</td>
      <td>2014</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>1761</th>
      <td>-1</td>
      <td>2014</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>20947</th>
      <td>-1</td>
      <td>2014</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>-1</td>
      <td>2014</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>7519</th>
      <td>-1</td>
      <td>2014</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 3 columns</p>
</div>




```python
# We can see that some homes were sold the year before they built, resulting in age_sold = -1
df.sort_values(by='age_sold', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>acres_lot</th>
      <th>price_millions</th>
      <th>date_sold</th>
      <th>year_sold</th>
      <th>age_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8996</th>
      <td>5160700035</td>
      <td>4/22/2015</td>
      <td>431000.0</td>
      <td>2</td>
      <td>1.50</td>
      <td>1300</td>
      <td>4000</td>
      <td>1.5</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>1900</td>
      <td>98144</td>
      <td>47.5937</td>
      <td>-122.301</td>
      <td>0.091827</td>
      <td>0.431000</td>
      <td>2015-04-22</td>
      <td>2015</td>
      <td>115</td>
    </tr>
    <tr>
      <th>3973</th>
      <td>2767604580</td>
      <td>2/23/2015</td>
      <td>635000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1340</td>
      <td>3900</td>
      <td>2.0</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>1900</td>
      <td>98107</td>
      <td>47.6711</td>
      <td>-122.379</td>
      <td>0.089532</td>
      <td>0.635000</td>
      <td>2015-02-23</td>
      <td>2015</td>
      <td>115</td>
    </tr>
    <tr>
      <th>11239</th>
      <td>625100004</td>
      <td>3/17/2015</td>
      <td>450000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1540</td>
      <td>67756</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1900</td>
      <td>98077</td>
      <td>47.7210</td>
      <td>-122.078</td>
      <td>1.555464</td>
      <td>0.450000</td>
      <td>2015-03-17</td>
      <td>2015</td>
      <td>115</td>
    </tr>
    <tr>
      <th>4434</th>
      <td>4232902615</td>
      <td>4/28/2015</td>
      <td>819000.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1300</td>
      <td>3600</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1900</td>
      <td>98119</td>
      <td>47.6345</td>
      <td>-122.366</td>
      <td>0.082645</td>
      <td>0.819000</td>
      <td>2015-04-28</td>
      <td>2015</td>
      <td>115</td>
    </tr>
    <tr>
      <th>19370</th>
      <td>2420069042</td>
      <td>4/24/2015</td>
      <td>240000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1553</td>
      <td>6550</td>
      <td>1.0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1900</td>
      <td>98022</td>
      <td>47.2056</td>
      <td>-121.994</td>
      <td>0.150367</td>
      <td>0.240000</td>
      <td>2015-04-24</td>
      <td>2015</td>
      <td>115</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20836</th>
      <td>1257201420</td>
      <td>7/9/2014</td>
      <td>595000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>3730</td>
      <td>4560</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>2015</td>
      <td>98103</td>
      <td>47.6725</td>
      <td>-122.330</td>
      <td>0.104683</td>
      <td>0.595000</td>
      <td>2014-07-09</td>
      <td>2014</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1761</th>
      <td>1832100030</td>
      <td>6/25/2014</td>
      <td>597326.0</td>
      <td>4</td>
      <td>4.00</td>
      <td>3570</td>
      <td>8250</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>10</td>
      <td>2015</td>
      <td>98040</td>
      <td>47.5784</td>
      <td>-122.226</td>
      <td>0.189394</td>
      <td>0.597326</td>
      <td>2014-06-25</td>
      <td>2014</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>20947</th>
      <td>6058600220</td>
      <td>7/31/2014</td>
      <td>230000.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1040</td>
      <td>1264</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>2015</td>
      <td>98144</td>
      <td>47.5951</td>
      <td>-122.301</td>
      <td>0.029017</td>
      <td>0.230000</td>
      <td>2014-07-31</td>
      <td>2014</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>3076500830</td>
      <td>10/29/2014</td>
      <td>385195.0</td>
      <td>1</td>
      <td>1.00</td>
      <td>710</td>
      <td>6000</td>
      <td>1.5</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>2015</td>
      <td>98144</td>
      <td>47.5756</td>
      <td>-122.316</td>
      <td>0.137741</td>
      <td>0.385195</td>
      <td>2014-10-29</td>
      <td>2014</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7519</th>
      <td>9520900210</td>
      <td>12/31/2014</td>
      <td>614285.0</td>
      <td>5</td>
      <td>2.75</td>
      <td>2730</td>
      <td>6401</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>2015</td>
      <td>98072</td>
      <td>47.7685</td>
      <td>-122.160</td>
      <td>0.146947</td>
      <td>0.614285</td>
      <td>2014-12-31</td>
      <td>2014</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 20 columns</p>
</div>



### Month Name


```python
df['month_sold'] =  df['date_sold'].dt.month_name()
```


```python
df['month_sold']
```




    0         October
    1        December
    2        February
    3        December
    4        February
               ...
    21592         May
    21593    February
    21594        June
    21595     January
    21596     October
    Name: month_sold, Length: 21597, dtype: object



### Season Sold


```python
df['month_index'] =  df['date_sold'].dt.month
```


```python
#seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
seasons = ['winter', 'winter', 'spring', 'spring', 'spring', 'summer', 'summer', 'summer', 'fall', 'fall', 'fall', 'winter']
season_dict = dict(zip(range(1,13), seasons))
season_dict
```




    {1: 'winter',
     2: 'winter',
     3: 'spring',
     4: 'spring',
     5: 'spring',
     6: 'summer',
     7: 'summer',
     8: 'summer',
     9: 'fall',
     10: 'fall',
     11: 'fall',
     12: 'winter'}




```python
df['season_sold'] = df['date_sold'].dt.month.map(season_dict)
```


```python
df[['date_sold','month_sold', 'season_sold']].sort_values(by='season_sold', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date_sold</th>
      <th>month_sold</th>
      <th>season_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10798</th>
      <td>2015-02-06</td>
      <td>February</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>7629</th>
      <td>2015-01-20</td>
      <td>January</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>7669</th>
      <td>2015-01-13</td>
      <td>January</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>7668</th>
      <td>2014-12-04</td>
      <td>December</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>7666</th>
      <td>2015-02-23</td>
      <td>February</td>
      <td>winter</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11677</th>
      <td>2014-10-29</td>
      <td>October</td>
      <td>fall</td>
    </tr>
    <tr>
      <th>11669</th>
      <td>2014-10-09</td>
      <td>October</td>
      <td>fall</td>
    </tr>
    <tr>
      <th>11667</th>
      <td>2014-10-27</td>
      <td>October</td>
      <td>fall</td>
    </tr>
    <tr>
      <th>11665</th>
      <td>2014-11-10</td>
      <td>November</td>
      <td>fall</td>
    </tr>
    <tr>
      <th>21596</th>
      <td>2014-10-15</td>
      <td>October</td>
      <td>fall</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 3 columns</p>
</div>



### Day of the Week Sold


```python
df['day_of_week_sold'] =  df['date_sold'].dt.day_name()
```


```python
df[['date_sold','month_sold', 'day_of_week_sold']].sort_values(by='date_sold', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date_sold</th>
      <th>month_sold</th>
      <th>day_of_week_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16580</th>
      <td>2015-05-27</td>
      <td>May</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>13040</th>
      <td>2015-05-24</td>
      <td>May</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>5632</th>
      <td>2015-05-15</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>15797</th>
      <td>2015-05-14</td>
      <td>May</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>927</th>
      <td>2015-05-14</td>
      <td>May</td>
      <td>Thursday</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7316</th>
      <td>2014-05-02</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>19661</th>
      <td>2014-05-02</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>6418</th>
      <td>2014-05-02</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>10689</th>
      <td>2014-05-02</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
    <tr>
      <th>4959</th>
      <td>2014-05-02</td>
      <td>May</td>
      <td>Friday</td>
    </tr>
  </tbody>
</table>
<p>21597 rows × 3 columns</p>
</div>



## DISTRIBUTIONS

### Clarify Grade and Condition Columns

#### King County's Grading System for Buildings:
##### Represents the construction quality of improvements. Grades run from grade 1 to 13. Generally defined as:

1. Falls short of minimum building standards. Normally cabin or inferior structure.

2. Falls short of minimum building standards. Normally cabin or inferior structure.

3. Falls short of minimum building standards. Normally cabin or inferior structure.

4. Generally older, low quality construction. Does not meet code.

5. Low construction costs and workmanship. Small, simple design.

6. Lowest grade currently meeting building code. Low quality materials and simple designs.

7. Average grade of construction and design. Commonly seen in plats and older sub-divisions.

8. Just above average in construction and design. Usually better materials in both the exterior and interior finish work.

9. Better architectural design with extra interior and exterior design and quality.

10. Homes of this quality generally have high quality features. Finish work is better and more design quality is seen in the floor plans. Generally have a larger square footage.

11. Custom design and higher quality finish work with added amenities of solid woods, bathroom fixtures and more luxurious options.

12. Custom design and excellent builders. All materials are of the highest quality and all conveniences are present.

13. Generally custom designed and built. Mansion level. Large amount of highest quality cabinet work, wood trim, marble, entry ways etc.

#### King County's Condition Scale:
##### Relative to age and grade. Coded 1-5.

1. Poor- Worn out. Repair and overhaul needed on painted surfaces, roofing, plumbing, heating and numerous functional inadequacies. Excessive deferred maintenance and abuse, limited value-in-use, approaching abandonment or major reconstruction; reuse or change in occupancy is imminent. Effective age is near the end of the scale regardless of the actual chronological age.

2. Fair- Badly worn. Much repair needed. Many items need refinishing or overhauling, deferred maintenance obvious, inadequate building utility and systems all shortening the life expectancy and increasing the effective age.

3. Average- Some evidence of deferred maintenance and normal obsolescence with age in that a few minor repairs are needed, along with some refinishing. All major components still functional and contributing toward an extended life expectancy. Effective age and utility is standard for like properties of its class and usage.

4. Good- No obvious maintenance required but neither is everything new. Appearance and utility are above the standard and the overall effective age will be lower than the typical property.

5. Very Good- All items well maintained, many having been overhauled and repaired as they have shown signs of wear, increasing the life expectancy and lowering the effective age with little deterioration or obsolescence evident with a high degree of utility.

### Describe Data - Descriptive Statistics


```python
print(f"The max price is {df['price_millions'].max()} million, min price is {df['price'].min()}")
```


```python
#Markdown Version of Below Table:
#df_markdown = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'acres_lot', 'floors', 'waterfront', 'condition', 'grade', 'yr_built']].describe()
#print(df_markdown.to_markdown())
```

Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'condition', 'grade', 'yr_built',
       'zipcode', 'lat', 'long'],
      dtype='object')


```python
format_dict = {'price': '{:20,.0f}', 'bedrooms': '{:.2f}', 'bathrooms': '{:.2f}', 'sqft_living': '{:20,.2f}', 'acres_lot': '{:.2f}', 'floors': '{:.2f}', 'waterfront': '{:.5f}',
'condition': '{:.2f}', 'grade': '{:.2f}', 'age_sold': '{:.0f}', 'yr_built': '{:.0f}', 'date': '{:.0f}'}
df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'acres_lot', 'floors',
    'waterfront', 'condition', 'grade', 'age_sold', 'date', 'yr_built']].describe().style.format(format_dict)
```

#### Interpret .describe()
- There are no negative values for any of the columns except age_sold
- The year range makes sense: homes were built between 1900 and 2015
- sqft_living average is around 2,000, which is close to the US average
- The average lot size is 15,099 square feet, or .35 acres
- The lot size mean is skewed by our outlier home, which stands on a 37.91 acre lot.
- The median sqft_lot size is 7,618 or .17 acres, which is much closer to the national average of 0.188 of an acre.
- Condition and Grade: The mean, max, and min all line-up with the scale of the system, described above.
    - Mean Grade is 7.66, so between a 7 and 8:
        - 7. Average grade of construction and design. Commonly seen in plats and older sub-divisions.
        - 8. Just above average in construction and design. Usually better materials in both the exterior and interior finish work.
    - Mean Condition is 3.41, closest to a 3:
        - 3. Average- Some evidence of deferred maintenance and normal obsolescence with age in that a few minor repairs are needed, along with some refinishing. All major components still functional and contributing toward an extended life expectancy. Effective age and utility is standard for like properties of its class and usage.

### Price Frequency Histogram


```python
# Price Frequency Histogram before removing outliers, for context:
# Here we can see that the distribution of price, our target variable, has considerable right-skew
ax = df.hist(column='price_millions',
             bins='auto',
             grid=False,
             figsize=(8,5))

ax = ax[0]
for x in ax:

    x.set_title("Distribution of Home Sale Price", size=20)

    # Set x-axis label
    x.set_xlabel("Sale Price of Home", labelpad=15, weight='bold', size=10)
    x.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}' + ' Million'))

    # Set y-axis label
    x.set_ylabel("Number of Homes Sold", labelpad=15, weight='bold', size=10)
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

```

### Price Density Histogram


```python
# Ok. Now let's draw a smooth curve, given the above data, using KDE (kernal density estimation)
# The smooth line in the visualization below is an estimate of the distributions of house prices
# The parameter bandwidth rules the smoothness of the underlying distribution

# The problem with a Probability Density Function plot, is that all "point probabilities" are 0
# We must calculate the area under the curve for an interval to get the actual probability that a home selling price is in a given range.
# Thus, it's not intuitive or easy to "read" the y-axis to get probabilities for continuous variables using a PDF like below.

plt.figure(figsize = (12,8))
sns.distplot(df.price_millions,kde=True)
plt.title('Density Histogram of Housing Prices in Kings County')
plt.show()
```

For example, if we wanted to know the probability that a house price is between 1.00 and 1.75 million dollars (inclusive), we would use the following formula to take the integral of this range (AKA area under the curve)
$$\large P(1 \leq X \leq 1.75) = \int_{1}^{1.75} f_x(x) dx \geq 0 $$


### Narrow dataset for modeling


```python
# Remember all columns in df are:
df.columns
```


```python
# Select a subset of columns to create matrix:
df_all_cols = df.drop(['acres_lot','id'], axis=1)
```


```python
df_all_cols.columns
```

# DATA PREPARATION
- Prepping to model and addressing missing values -- completed in previous section "Data Quality"
- Convert categorical data to numeric via one-hot encoding (we didn't need to do this because everything is numeric and ordinal)
- Type conversions (this was done in the previous section - price to millions and sqft to acres)
- Remove multicollinearity (correlated predictors)
- Standardize numeric data -- We're not going to standardize our independent variables using z-score normalization because it's not necessary unless we're using logistic regression.


## Explore Multicollinearity
- This a phenemonen where two variables we are using as predictors are correlated with each other
- This violates one of the assumptions of performing linear regression - that all independent variables are independent from one another
- If we left all features in the model without addressing multicollinearity, it would become very hard for the model to estimate the relationship between independent variables and the dependent variable, because rather than change independently, the features would change in pairs or groups.


```python
correlations = df_all_cols.drop(['price_millions'],axis = 1).corr()
```


```python
plt.figure(figsize=(12,12)) # Set size of figure
# Use df.corr() as your matrix for the heatmap
# I set the color scheme to green using cmap
# annot= True adds the float value on each square
# fmt='.1f' sets the number of decimal places for each float number. If you want 1.00, for example, use fmt='.2f'
# cbar_kws={"shrink": .77}  - this argument shrinks the side color bar to .77 of its original size
# square=True - this argument makes the figure square
ax = sns.heatmap(correlations, cmap = "Greens", annot=True, fmt='.2f', cbar_kws={"shrink": .77}, square=True)
# Sets the title
ax.set(title='King County House Sales Dataset Correlation Heatmap')
# Get the y-axis limit values for the size of the figure:
bottom, top = ax.get_ylim()
# Add .5 to the bottom and top of the y-axis limits to fix an error where the top and bottom squares are cut off
# This is an error for the versions of Seaborne and Matplotlib I'm using
ax = ax.set_ylim(bottom + 0.5, top - 0.5)
# Rotate the bottom labels by 30 degrees (100-70)
plt.xticks(rotation=70)
plt.show()

```


```python
correlations['price'].sort_values(ascending=False)
```


```python
correlations["sqft_living"].sort_values(ascending=False)
```

### Correlations Interpretation
- Features highly correlated with price are sqft_living, grade, and bathrooms
- Features highly correlated with sqft_living are grade, bathrooms, price_millions, and bedrooms

## Address Multicollinearity
- First, review the columns and shape of the dataframe we will be using -- "df_all_cols"
- Next, use Variance Inflation Factor to detect high multicollinearity in our set of independent variables


```python
df_all_cols.columns
```


```python
df_all_cols.shape
```


```python
# We want to write a simple loop to use statsmodel's variance_inflation_factor method on each array (row) of our dataframe, for each column
# That's why we use X.shape[1] as the range in the loop below, since the second term in the output of the .shape method is the number of columns
X = sm.add_constant(df_all_cols.drop(['price','price_millions'], axis=1))
pd.Series([variance_inflation_factor(X.values, i)
               for i in range(X.shape[1])],
              index=X.columns)
```

### sqft_living has a moderately high VIF value, followed by grade
- We can try removing sqft_living or grade to see if it improves our R-squared and Root Mean Squared Error values.

# df_all_cols - Linear Regression
- Tune models to get the highest performance possible on our task
  - Considerations:
    - What kind of task? Classification task? Regression task?
      - This will be a regression task
    - Which models will we use
      - Linear Regression
    - Will be use regularization?
      - Out of scope for this exercise
    - How will we validate our model?
      - R-squared, Mean Squared Error, and the p-values of each coefficient.
    - Loss functions?
      - Out of scope for this exercise
    - What is the performance threshold for success?
      - We want to make sure to satisfy the assumptions of linear regression, and get a respectable R-squared value (at least .5)


## df_all_cols
- First we'll use statsmodels.api to run a linear regression and examine model validation scores.
- We'll use all the independent variables (df_all_cols) for now.


```python
df_all_cols.columns
```


```python
# Add constant (AKA y-intercept):
# The constant is set to 1 as default - with means that our constant coefficient will be equal to 1*Beta(0)
# Our other variables (other Betas) will be multiplied by a particular coefficient to predict price
X = sm.add_constant(df_all_cols.drop(['price','price_millions'], axis=1))
y = df_all_cols['price']
X.head()
```


```python
# The order of the parameters is: endogenous response variable(dependent variable), exogenous variables(independent variables)
model_all_cols = sm.OLS(y,X).fit()
```


```python
model_all_cols.summary()
```


```python
# In addition to R-Squared, we'll use Root Mean Squared Error to validate our model
# Here, our RMSE value is .047821 (million dollars) -- which heuristically can be thought of...
# ...as the normalized distance between our y values and our y-predicted values
# Thus, for any given prediction we make for a home price, we can expect to be about $47,821 off.
# Our model is failing to account for some important features underlying the data.
yhat = model_all_cols.predict(X)
model_rmse = np.sqrt(mean_squared_error(y, yhat))
model_rmse
```

## Test Linearity Assumption of Linear Regression
- By virtue of the method we are using to model our data, linear regression, we must satisfy the assumption that each of our explanatory variables exhibits a linear relationship with our dependent variable.
- In other words, we are assuming the following mathematical relationship between dependent variable y, and explanatory variables X:
$$
\begin{aligned}
	y &= mx + b \\
\end{aligned}
$$



```python
df_all_cols.columns
```


```python
# The first index is the number of rows and the second index is the number of cols. The third index is the position count of the figure layout

plt.subplots(figsize=(25, 25))


plt.subplot(5, 3, 1)
plt.scatter(df_all_cols['age_sold'], df_all_cols['price'])
plt.xlabel('Age Sold')
plt.ylabel('Price')

plt.subplot(5, 3, 2)
plt.scatter(df_all_cols['long'], df_all_cols['price'])
plt.xlabel('Longitude')
plt.ylabel('Price')

plt.subplot(5, 3, 3)
plt.scatter(df_all_cols['lat'], df_all_cols['price'])
plt.xlabel('Latitude')
plt.ylabel('Price')

plt.subplot(5, 3, 4)
plt.scatter(df_all_cols['zipcode'], df_all_cols['price'])
plt.xlabel('Zipcode')
plt.ylabel('Price')

plt.subplot(5, 3, 5)
plt.scatter(df_all_cols['yr_built'], df_all_cols['price'])
plt.xlabel('Year Built')
plt.ylabel('Price')

plt.subplot(5, 3, 6)
plt.scatter(df_all_cols['grade'], df_all_cols['price'])
plt.xlabel('Grade')
plt.ylabel('Price')

plt.subplot(5, 3, 7)
plt.scatter(df_all_cols['condition'], df_all_cols['price'])
plt.xlabel('Condition')
plt.ylabel('Price')

plt.subplot(5, 3, 8)
plt.scatter(df_all_cols['waterfront'], df_all_cols['price'])
plt.xlabel('Waterfront')
plt.ylabel('Price')

plt.subplot(5, 3, 9)
plt.scatter(df_all_cols['floors'], df_all_cols['price'])
plt.xlabel('Floors')
plt.ylabel('Price')

plt.subplot(5, 3, 10)
plt.scatter(df_all_cols['sqft_lot'], df_all_cols['price'])
plt.xlabel('Sqft Lot')
plt.ylabel('Price')

plt.subplot(5, 3, 11)
plt.scatter(df_all_cols['sqft_living'], df_all_cols['price'])
plt.xlabel('Sqft Living')
plt.ylabel('Price')

plt.subplot(5, 3, 12)
plt.scatter(df_all_cols['bedrooms'], df_all_cols['price'])
plt.xlabel('Bedrooms')
plt.ylabel('Price')

plt.subplot(5, 3, 13)
plt.scatter(df_all_cols['bathrooms'], df_all_cols['price'])
plt.xlabel('Bathrooms')
plt.ylabel('Price')

plt.subplot(5, 3, 14)
plt.scatter(df_all_cols['date'], df_all_cols['price'])
plt.xlabel('Year Sold')
plt.ylabel('Price')

plt.subplot(5, 3, 15)
plt.scatter(df_all_cols['price'], df_all_cols['price'])
plt.xlabel('Price')
plt.ylabel('Price')


# space between the plots
plt.tight_layout(4)

# show plot
plt.show()

```

## Notes on Scatterplots - Why we must One Hot Encode (specifically Dummy Variable Encode)
- For all scatterplots above where our datapoints appear in columns, rather than a continuous swath of points that overlap horizontally - we can see an important phenomenon that we've overlooked in the pre-processing steps "Understand Data" and "Data Preparation."
- Zipcode, Grade, Condition, Floors, Bedrooms, Bathrooms, Year Sold, and Waterfront are all categorical variables. That is, while there may be a natural ordering, as in a numerical range (.5-8 bathrooms), the data falls into distinct buckets. One way to think about categorical values is that the number of options is a fixed set. While the price of a home could be adjusted by any increment in dollar and cents amount, the number of bedrooms in a home is a variable that describes a "type" of home -- "1 bedroom mother-in-law cottage", "3-bedroom single family home", etc.
- For the purpose of this project I define a categorical variable as one where data falls into discrete bins, and where we can observe that comparing any given category ("4 bedrooms") - would be useful to regress against its opposite ("Not 4 bedrooms")
- Thus, we are going to one-hot encode all of our categorical variables using Dummy Variable Encoding - where we use i-1 columns to represent i different values for each categorical variable.
- We only need i-1 columns because the last value is Zero for all the rest of the columns.
- A simple example of this is Waterfront, which is already one-hot encoded - it's already an "Indicator Variable" which means the only possible values are True or False (We use 0 or 1)
- Thus for the variable "Waterfront" -- if the coefficient ends up being $200,000, that means the predicted price for waterfront homes will be y = 200,000(1) + intercept.
- In the case of non-waterfront homes, then, y = 200,000(0) + intercept, or y = intercept
- Our main task will be to think about which category should be dropped when we do Dummy Variable Encoding.
- Usually this dropped variable, AKA "baseline" is the value associated with the lowest target variable, or that which is associated with the average target variable.

Note:
Ultimately, if our model has a relatively low R-squared value (close to 50%), but we have:
1. fulfilled the assumptions of linear regression
2. have normally distributed error
3. coefficients have good p-values

... then we can still draw meaningful conclusions re: the relationships between the independent and dependent variables. Our coefficients will still represent the mean flucuation in the dependent variable for every 1-unit change in a given independent variable.

## df_all_cols - Interpret
- Using all of our columns to predict price, and not subsetting the data at all:
- Our p-values are all 0.00, which is great.
- However, this model is violating necessary assumptions
    - The scatterplot of predicted price vs. price is clearly not a linear relationship
    - The plot of the residuals vs. the predicted price is trumpet-shaped, which is highly heteroscedastic.
    - What we are aiming for is linearity and homoscedasticity - a fairly even distribution of the residuals vs. predicted price


```python

```


```python

```


```python

```

# Outliers Removed - Linear Regression
- Removing outliers can be a bit of an art, but given that my business case is to offer advice to mostly average, middle-class home owners, it makes sense to eliminate the priciest and lowest cost homes.
- From the descriptive statistics we ran in the Understanding Data --> Distributions section, I surmised that the independent variables with the most unusual values were price and number of bedrooms.

## Outliers Removed - statsmodels


```python
df_rm_outliers = df_all_cols.drop(['sqft_lot'], axis=1)
```


```python
# Here we take the absolute value of the z-scores for each value in columns "price" and "bedrooms" and filter our df by...
# ...only those values which have a z-score < 3

df_rm_outliers = df_rm_outliers[(np.abs(stats.zscore(df_rm_outliers[['bedrooms']])) < 2.5) & (np.abs(stats.zscore(df_rm_outliers[['price']])) < 2.5)]
```


```python
# Sort by Bedrooms:
df_rm_outliers[['price_millions', 'bedrooms', 'bathrooms']].sort_values('bedrooms', ascending = False).head(10)
```


```python
df_rm_outliers.columns
```


```python
#Markdown Version of Below Table:
#df_markdown_rm_outliers = df_rm_outliers[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'acres_lot', 'floors', 'waterfront', 'condition', 'grade', 'yr_built']].describe()
#print(df_markdown_rm_outliers.to_markdown())
```


```python
# These values make a lot more sense
# The minimum and maximum values for price, bedrooms, and bathrooms are conceivable attributes of a middle class person's home.
# We didn't sacrifice on the size of our dataset, having only shed only about 4% of the number of rows in the original dataset.
format_dict = {'bedrooms': '{:.2f}', 'bathrooms': '{:.2f}', 'floors': '{:.2f}', 'sqft_living': '{:20,.2f}', 'sqft_lot': '{:20,.2f}',
               'price_millions': '{:,.8f}', 'yr_built': '{:.0f}', 'condition': '{:.2f}', 'grade': '{:.2f}', 'waterfront': '{:.5f}'}
df_rm_outliers[['price_millions', 'bedrooms', 'bathrooms', 'sqft_living', 'floors',
    'waterfront', 'condition', 'grade', 'yr_built']].describe().style.format(format_dict)
```


```python
# For our other variables (other Betas) will be multiplied by a particular coefficient to predict price
X = sm.add_constant(df_rm_outliers.drop(['price','price_millions'], axis=1))
y = df_rm_outliers['price']
X.head()
```


```python
model_rm_outliers = sm.OLS(y, X).fit()
```


```python
model_rm_outliers.summary()
```


```python
# For any given prediction we make for a home price, we can expect to be about $25,790 off.
yhat = model_rm_outliers.predict(X)
model_rmse = mean_squared_error(y,yhat)
model_rmse
```

## Outliers Removed  - interpret

- The R-squared value for model_log is 0.62
- Our p-values are all 0.00
- Model RMSE: For any given prediction we make for a home price, we can expect to be about $25,790 off
- Necessary assumptions for linear regression are better fulfilled than our last two iterations of linear regression modeling:
    - The scatterplot of predicted price vs. price has a mostly linear shape
    - The plot of the residuals vs. the predicted price is almost evenly spread over the horizontal axis

Dropping sqft_lot and removing outliers based on the value of the z-scores for each value in columns "price" and "bedrooms" has improved our results compared to our last two iterations. Based on our p-values, R-squared value, and RMSE, as well as our assumptions of linear regression, we can be reasonably confident that we can interpret our coefficients as giving us the price increase of our home for every unit increase of the variable.
My real estate company can therefore advise clients that they should install a roof deck if it would add a waterfront view, as well as increase the quality of any improvements in their home, using this scale as their guide to quality craftmanship:
9. Better architectural design with extra interior and exterior design and quality.

10. Homes of this quality generally have high quality features. Finish work is better and more design quality is seen in the floor plans. Generally have a larger square footage.

11. Custom design and higher quality finish work with added amenities of solid woods, bathroom fixtures and more luxurious options.

12. Custom design and excellent builders. All materials are of the highest quality and all conveniences are present.

13. Generally custom designed and built. Mansion level. Large amount of highest quality cabinet work, wood trim, marble, entry ways etc.
