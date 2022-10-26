# Module 1 Python 5 of 5 (Capstone project).
# Hands-on Lab 2: Data Wrangling
# Guillermo Dominguez, PhD.

# Import libraries
# Requests allows us to make HTTP requests which we will use to get data from an API
import pandas as pd
import numpy as np

# Show all columns in the PyCharm preview
pd.set_option('display.width', 800)  # avoid truncated view
pd.set_option('display.max_columns', 50)  # columns shown
pd.set_option('display.max_rows', 999)  # rows shown

# ===== Data Analysis ====== #
# Load Space X dataset, from last section.
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")
print("Imported DataFrame contains:", len(df), "rows and ", len(df.columns), "columns.")
print(df.head(10))
# Percentage of the missing values in each attribute
print(" === Percentage of the missing values in each column")
print(df.isnull().sum()/df.count()*100)
# Numerical/categorial columns
# print(df.dtypes)


# TASK 1: Calculate the number of launches on each site
# Space X launch facilities
# - Cape Canaveral Space Launch Complex 40 VAFB SLC 4E
# - Vandenberg Air Force Base Space Launch Complex 4E (SLC-4E)
# - Kennedy Space Center Launch Complex 39A KSC LC 39A

# Number of launches for each site
print(" === Number of launches for each site")
print(df['LaunchSite'].value_counts())

# TASK 2: number and occurrence of each orbit
print(" === Number and occurrence of each orbit")
print(df['Orbit'].value_counts())

# TASK 3: number and occurrence of mission outcome per orbit type
landing_outcomes = df.groupby(['Orbit'])['Outcome'].value_counts()
print(" === Number and occurrence of mission outcome per orbit type")
print(landing_outcomes)

# True Ocean    means the mission outcome was successfully landed to a specific region of the ocean
# False Ocean   means the mission outcome was unsuccessfully landed to a specific region of the ocean
# True RTLS     means the mission outcome was successfully landed to a ground pad
# False RTLS    means the mission outcome was unsuccessfully landed to a ground pad
# True ASDS     means the mission outcome was successfully landed to a drone ship
# False ASDS    means the mission outcome was unsuccessfully landed to a drone ship
# None ASDS     represents a failure to land.
# None None     represents a failure to land.

# loop to show outcome, I personally prefer to use groupby function
for i, outcome in enumerate(landing_outcomes.keys()):
    print(i, outcome)

# set of outcomes where the second stage did not land successfully
# bad_outcomes = set(landing_outcomes.keys()[[1, 3, 5, 6, 7]]) # <-- clumsy method where one has to manually look for rows where None is present.
bad_outcomes = df['Outcome'].str.contains('None', regex=False).sum()
print(" === Outcomes where the second stage did not land successfully (None is present)")
print(bad_outcomes)
print("=== Outcomes overview table")
print(df['Outcome'].value_counts())

# TASK 4: Create a landing outcome label from Outcome column
# Using the Outcome, create a list where the element is zero if the corresponding row in Outcome is in
# the set bad_outcome; otherwise, it's one.
landing_class = []
for outcome in df['Outcome']:
    if outcome in bad_outcomes:
        landing_class.append(0)
    else:
        landing_class.append(1)

# Add this class to the dataframe
# df['Class'] = landing_class
# df['Class'] = []
# df['Class'] = ''
#
# # Using a fast numpy method:
# Failure_condition = ['None', 'False'] # contains these words
# for i, row in df.interrows():
#     eval = row['Outcome']
#     print("Loop iteration", i)
#     print("evaluating:", row['Outcome', 'Class'])
#     if eval.isin(Failure_condition):
#         row['Class'] == 0
#     else:
#         row['Class'] == 1

# df['Class'] = np.where(df['Outcome'].str.contains('None'), 0, 1) # But this does not account for the False cases, hence gives only 21 failures
df['Class'] = np.where((df['Outcome'].str.contains('None')) | (df['Outcome'].str.contains('False')), 0, 1)
df['Class'] = np.where(df['LandingPad']== np.NaN, 0, 1) # rif Landing pas is no kwnon, does not count
# df['Class'] = np.where(np.logical_and((df['Outcome'].str.contains('None'), 0, 1), (df['Outcome'].str.contains('False'), 0, 1))) # does not work

print("=== Number of failures to land")
print((df['Class'].values == 0).sum())

print(" === Complete table with Class outcome added")
print(df.head())

# Success rate
print("=== Success rate")
print(df['Class'].mean()*100, "%")

# Export to .csv
df.to_csv('dataset_part_2.csv', index=False)
print("Falcon 9 outcomes data exported to .csv")
print(">>>>--- End of Hand-on lab Data Wrangling ---<<<<")

