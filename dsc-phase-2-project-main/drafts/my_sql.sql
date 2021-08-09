CREATE TABLE original (
    id text,
    date text,
    price text,
    bedrooms text,
    bathrooms text,
    sqft_living text,
    sqft_lot text,
    floors text,
    waterfront text,
    view text,
    condition text,
    grade text,
    sqft_above text,
    sq ft_basement text,
    yr_built text,
    yr_renovated text,
    zipcode text,
    lat text,
    long text,
    sqft_living15 text,
    sqft_lot15 text
);

\copy original FROM '/Users/clairesarraille/git-repos/ph2finproj/dsc-phase-2-project-main/data/kc_house_data.csv' WITH DELIMITER ',' CSV HEADER;


-- Created new column 'water_bin' to set value to 0 where it was NULL:
UPDATE original SET water_bin = '0.0' where waterfront IS NULL;

-- Update all instances of sqft_basement having character '?' to '0.0'
UPDATE original SET sqft_basement = replace(sqft_basement, '?', '0.0');


'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'condition', 'grade', 'yr_built'

