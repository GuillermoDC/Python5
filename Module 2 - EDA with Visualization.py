# Module 2 Python 5 of 5 - Capstone project.
# Hands-on Lab: Complete the EDA with Visualization
# Guillermo Dominguez

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Show all columns in the PyCharm preview
pd.set_option('display.width', 800)  # avoid truncated view
pd.set_option('display.max_columns', 50)  # columns shown
pd.set_option('display.max_rows', 999)  # rows shown

# ==== Exploratory Data Analysis

# Read the SpaceX dataset into a Pandas dataframe
file_path = 'dataset_Module1_DataWrangling.csv'
# df = pd.read_csv(file_path)
df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

df.rename(columns={'Class': 'Success'}, inplace=True)
# print(df.head())

# See how the FlightNumber (indicating the continuous launch attempts.) and Payload variables would affect the launch outcome.
# plot out the FlightNumber vs. PayloadMassand overlay the outcome of the launch.
# We see that as the flight number increases, the first stage is more likely to land successfully.
# The payload mass is also important; it seems the more massive the payload, the less likely the first stage will return.

sns.catplot(y="PayloadMass", x="FlightNumber", hue="Success", data=df, aspect=4, legend=False)
plt.title("SpaceX Falcon 9 - Flight Number vs. Payload Mass", loc='center')
plt.xlabel("Flight Number", fontsize=15)
plt.ylabel("Pay load Mass (kg)", fontsize=15)
plt.legend(loc="upper left", title='Success: 1 | Failure: 0', shadow=True)
plt.grid(axis='y', color='green', linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()

# TASK 1: Visualize the relationship between Flight Number and Launch Site
# plot FlightNumber vs LaunchSite, set the parameter x parameter to FlightNumber,set the y to Launch Site and
# set the parameter hue to 'class'
sns.catplot(y="FlightNumber", x="LaunchSite", hue="Success", data=df, aspect=1, legend=False)
plt.title("SpaceX Falcon 9 - Flight Number vs. LaunchSite", loc='center')
plt.ylabel("Flight Number", fontsize=15)
plt.xlabel("LaunchSite", fontsize=15)
plt.legend(loc="upper center", title='Success: 1 | Failure: 0', shadow=True)
plt.grid(axis='x', color='green', linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()

# TASK 2: Visualize the relationship between Flight Number and Launch Site
# Now if you observe Payload Vs. Launch Site scatter point chart you will find for the VAFB-SLC launchsite
# there are no rockets launched for heavy payload mass (greater than 10000).
sns.catplot(y="PayloadMass", x="LaunchSite", hue="Success", data=df, aspect=1, legend=False)
plt.title("SpaceX Falcon 9 - Payload Mass vs. LaunchSite", loc='center')
plt.ylabel("Pay load mass (kg)", fontsize=15)
plt.xlabel("LaunchSite", fontsize=15)
plt.legend(loc="upper center", title='Success: 1 | Failure: 0', shadow=True)
plt.grid(axis='x', color='green', linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()

# TASK 3: Visualize the relationship between success rate of each orbit type
# visually check if there are any relationship between success rate and orbit type.
# bar chart for the success rate of each orbit

Orbit_success = df.groupby(['Orbit'])['Success'].mean().sort_values(ascending=False).reset_index()
Orbit_success.rename(columns={Orbit_success.columns[1]: "Success rate"}, inplace=True)
# print(Orbit_success)

Orbits_types = Orbit_success['Orbit'].tolist()
# print(Orbits_types)
title = 'Success rate per Orbit'
fig = plt.figure()
ind = np.arange(len(Orbit_success['Orbit'])) # the n locations for the Orbits marks
plt.bar(Orbit_success['Orbit'], Orbit_success['Success rate'], color="blue")
plt.title(title, loc='center')
plt.ylabel('Success Rate (x100%)')
plt.xlabel('Orbit type')
plt.grid(axis='y', color='green', linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()

# TASK 4: Visualize the relationship between FlightNumber and Orbit type
# For each orbit, we want to see if there is any relationship between FlightNumber and Orbit type.
# scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value
sns.catplot(y="Orbit", x="FlightNumber", hue="Success", data=df, aspect=1, legend=False)
plt.title("SpaceX Falcon 9 - Orbit type vs. Flight number", loc='center')
plt.ylabel("Orbit type", fontsize=15)
plt.xlabel("Flight Number #", fontsize=15)
plt.legend(loc="lower left", title='Success: 1 | Failure: 0', shadow=True)
plt.grid(axis='y', color='green', linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()

# TASK 5: Visualize the relationship between Payload and Orbit type
# plot Payload vs. Orbit scatter point charts to reveal the relationship
sns.catplot(y="PayloadMass", x="Orbit", hue="Success", data=df, aspect=1, legend=False)
plt.title("SpaceX Falcon 9 - Pay load mass vs. Orbit type", loc='center')
plt.ylabel("Pay load mass (kg)", fontsize=15)
plt.xlabel("Orbit type", fontsize=15)
plt.legend(loc="upper left", title='Success: 1 | Failure: 0', shadow=True)
plt.grid(axis='x', color='green', linestyle='--', linewidth=0.5)
plt.grid(axis='y', color='green', linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()
# With heavy payloads the successful landing or positive landing rate are more for Polar,LEO and ISS.
# However, for GTO we cannot distinguish this well as both positive landing rate and negative landing(unsuccessful mission) are both there here.

#TASK 6: Visualize the launch success yearly trend
# plot a line chart with x axis to be Year and y axis to be average success rate, to get the average launch success trend.
# function to get the year from the date:
year=[]
def Extract_year(date_input):
    for i in date_input:
        year.append(i.split("-")[0])
    return year
year = Extract_year(df['Date'])
# Insert the year list into dataframe
df.insert(1, 'Year', year)
# print(df.head())
# Group per year and make mean of success rate
df_lineplot = df.groupby(['Year'])['Success'].mean().reset_index()

# line chart with x axis to be the extracted year and y axis to be the success rate
plt.cla() # clear instance to free memory
sns.lineplot(y="Success", x='Year', data=df_lineplot, legend=False)#, markers=True) #estimator=None
plt.title("SpaceX Falcon 9 - Success rate vs. Year", loc='center')
plt.ylabel("Success rate (%)", fontsize=15)
plt.xlabel("Year", fontsize=15)
plt.grid(axis='y', color='green', linestyle='--', linewidth=0.5)
plt.grid(axis='x', color='green', linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()

# Select features to be used in prediction models later
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Success', 'Serial']]
# print(features.head())

# TASK 7: Create dummy variables to categorical columns
# Use the function get_dummies and features dataframe to apply OneHotEncoder to the column Orbits, LaunchSite,
# LandingPad, and Serial. Assign the value to the variable features_one_hot, display the results using the method head.
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])

# TASK 8: Cast all numeric columns to float64
# Now that our features_one_hot dataframe only contains numbers cast the entire dataframe to variable type float64
features_one_hot = features_one_hot.astype('float64')
print(features_one_hot.head())

# Export to .csv
features_one_hot.to_csv('dataset_part_3_EDA_Visualization.csv', index=False)
print("End of Hand-on lab EDA with Visualization")

