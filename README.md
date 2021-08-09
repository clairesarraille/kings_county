# Phase Two Project for Flatiron School

- Location of Code:
  -  dsc-phase-2-project-main/***student_deliver_08-07.ipynb***

The data used is a curated collection of home sale data from Kings County, Washington.
All column descriptions can be found on the real Kings County website, here:
[Residential Glossary of Terms](https://info.kingcounty.gov/assessor/esales/Glossary.aspx?type=r#s)


The purpose of this project was to model King County's House Sale Price Data and make recommendations for the most lucrative renovations. The biggest hurdle was finding a way to subset the data so that it would conform to the necessary assumptions of linear regression: linearity and homoscedasticty.

- In early data exploration I noticed that when dividing the data into waterfront and non-waterfront properties, they appeared as quite distinct segments. This led me to segmenting the housing dataset by waterfront. My conclusion is that homeowners in Kings County who own waterfront properties should increase sqaure footage as well as using the King's County Grading System to guide their renovations:
        - Finish work is better and more design quality is seen in the floor plans. Generally have a larger square footage.
        - Custom design and higher quality finish work with added amenities of solid woods, bathroom fixtures and more luxurious options.
        - Large amount of highest quality cabinet work, wood trim, marble, entry ways etc.

![kde](https://user-images.githubusercontent.com/71570329/128661075-b4559156-e107-42ae-8a79-d646155f530d.png)
