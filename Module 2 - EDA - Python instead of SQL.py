# Module 2 Python 5 of 5 - Capstone project.
# Hands-on Lab: Complete the EDA with SQL
# I used Python instead of SQL. See discussion thread https://courses.edx.org/courses/course-v1:IBM+DS0720EN+2T2021/discussion/forum/course/threads/61cc5f6eb38ee505bf0365ef
# I do not agree that in this course the SQL should be added to perform the EDA. There is an independent IBM Python course for that.
# Guillermo Dominguez

import pandas as pd

# Remove warning (not important since I am rewriting files)
pd.options.mode.chained_assignment = None  # default='warn'

# Show all columns in the PyCharm preview
pd.set_option('display.width', 800)  # avoid truncated view
pd.set_option('display.max_columns', 50)  # columns shown
pd.set_option('display.max_rows', 999)  # rows shown

file_path = 'C:\\Users\\guillermo.dominguez\\Downloads\\Spacex.csv'
df = pd.read_csv(file_path, sep=',', skiprows=0, encoding=None)
# print(df)
# Task 1: Display the names of the unique launch sites in the space mission
launch_site = df['Launch_Site'].unique()
print("Names of the unique launch sites in the space mission")
print(launch_site)
# Task 2: Display 5 records where launch sites begin with the string 'KSC'
KSC = df[df['Launch_Site'].str.startswith('KSC')]
print("5 records where launch sites begin with the string 'KSC'")
print(KSC.head(5))
# Task 3: Display the total payload mass carried by boosters launched by NASA (CRS)
total_mass = df[df['Customer'] == 'NASA (CRS)']['PAYLOAD_MASS__KG_'].sum()
print("Total mass of NASA (CRS) (kg): ", total_mass)
# Task 4: Display average payload mass carried by booster version F9 v1.1
avg_mass = df[df['Booster_Version']=='F9 v1.1']['PAYLOAD_MASS__KG_'].mean()
print("Average payload mass of the Booster F9 V1.1 (kg): ", avg_mass)
# Task 5: List the date where the first succesful landing outcome in drone ship was acheived.
# First_success = df[df['Landing _Outcome'] == 'Success (drone ship)']['Date'].first_valid_index()
First_success = df[df['Landing _Outcome'] == 'Success (drone ship)']['Date'].iat[0]
print('First Successful landing in drone ship date: ', First_success)
# Task 6: List the names of the boosters which have success in ground pad and have payload mass greater than 4000 but less than 6000
big_success = df[(df['Landing _Outcome']=='Success (ground pad)') & (df['PAYLOAD_MASS__KG_'].between(4000, 6000,inclusive='left'))]['Booster_Version']
print("List of boosters which had success in ground pad and have payload mass greater than 4000 but less than 6000")
print(big_success)
# Task 7: List the total number of successful and failure mission outcomes
success = df[df['Mission_Outcome'].str.contains('Success')]
failure = df[df['Mission_Outcome'].str.contains('Failure')]
print("Successful missions:", len(success))
print(success)
print("Failure missions:", len(failure))
print(failure)
# Task 8: List the names of the booster_versions which have carried the maximum payload mass. Use a subquery
max_boosters = df.groupby(['Booster_Version'])['PAYLOAD_MASS__KG_'].max()
print("Booster version names carrying the maximum payload mass. Table length: ", len(max_boosters))
print(max_boosters)
# Task 9: List the records which will display the month names, succesful landing_outcomes in ground pad ,booster versions, launch_site for the months in year 2017
df['Date_clean'] = pd.to_datetime(df['Date']) #convert to date tuple
df['Month'] = df['Date_clean'].dt.strftime('%b') # extract month
df['Year'] = df['Date_clean'].dt.strftime('%Y') # extract month
df_selection = df[(df['Landing _Outcome'] == 'Success (ground pad)') & (df['Year'] == '2017')]
print('month names, succesful landing_outcomes in ground pad ,booster versions, launch_site for the months in year 2017')
print(df_selection)
# Task 10: Rank the count of successful landing_outcomes between the date 2010-06-04 and 2017-03-20 in descending order.
success_landings = df[(df['Landing _Outcome'].str.contains('Success')) & (df['Date_clean'].between('2010-06-04', '2017-03-20'))].sort_values(by='Date_clean', ascending=False)#.reset_index(drop=True)
print("Rank the count of successful landing_outcomes between the date 2010-06-04 and 2017-03-20 in descending order.")
print(success_landings)

# ===== GRADED QUESTIONS

# Question 1: retrieve the most recent date from the Spacex table
recent = df['Date_clean'].max()
print("Most recent date on the table")
print(recent)

# Question 2: display the minimum payload mass
min_mass = df['PAYLOAD_MASS__KG_'].min()
print("Minimum payload mass")
print(min_mass)
