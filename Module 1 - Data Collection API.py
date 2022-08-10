# Module 1 Python 5 of 5 (Capstone project).
# Hands-on Lab Complete Data Collection API Lab
# Guillermo Dominguez, PhD

# Import libraries
# Requests allows us to make HTTP requests which we will use to get data from an API
import requests
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import datetime

# Show all columns in the PyCharm preview
pd.set_option('display.width', 800)  # avoid truncated view
pd.set_option('display.max_columns', 50)  # columns shown
pd.set_option('display.max_rows', 999)  # rows shown

# ===== 1 - API ====== #
# Calls to SpaceX API to get data
# An application programming interface (API) is a way for two or more computer programs to communicate with each other.

# Takes the dataset and uses the rocket column to call the API and append the data to the list
def getBoosterVersion(data):
    for x in data['rocket']:
        response = requests.get("https://api.spacexdata.com/v4/rockets/"+str(x)).json()
        BoosterVersion.append(response['name'])

# Takes the dataset and uses the launchpad column to call the API and append the data to the list
def getLaunchSite(data):
    for x in data['launchpad']:
        response = requests.get("https://api.spacexdata.com/v4/launchpads/"+str(x)).json()
        Longitude.append(response['longitude'])
        Latitude.append(response['latitude'])
        LaunchSite.append(response['name'])

# Takes the dataset and uses the payloads column to call the API and append the data to the lists
def getPayloadData(data):
    for load in data['payloads']:
        response = requests.get("https://api.spacexdata.com/v4/payloads/"+load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])

# Takes the dataset and uses the cores column to call the API and append the data to the lists
def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get("https://api.spacexdata.com/v4/cores/"+core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])

spacex_url="https://api.spacexdata.com/v4/launches/past"

response = requests.get(spacex_url)
#print(response.content) # It is a mess, needs to be normalized

# Task 1: Request and parse the SpaceX launch data using the GET request

# To make the requested JSON results more consistent, we will use the following static response object for this project:
static_json_url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json'
# We should see that the request was successfull with the 200 status response code
# print(response.status_code)

# create a Dataframe from the json
df = response.json()
# normalize the dataframe
data = pd.json_normalize(df)
# print(data)

# We will now use the API again to get information about the launches using the IDs given for each launch.
# Specifically we will be using columns rocket, payloads, launchpad, and cores.

# Lets take a subset of our dataframe keeping only the features we want and the flight number, and date_utc.
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# We will remove rows with multiple cores because those are falcon rockets with 2 extra rocket boosters and rows that
# have multiple payloads in a single rocket.
data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]

# Since payloads and cores are lists of size 1 we will also extract the single value in the list and replace the feature.
data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

# We also want to convert the date_utc to a datetime datatype and then extracting the date leaving the time
data['date'] = pd.to_datetime(data['date_utc']).dt.date

# Using the date we will restrict the dates of the launches
data = data[data['date'] <= datetime.date(2020, 11, 13)]

# From the rocket we would like to learn the booster name
#
# From the payload we would like to learn the mass of the payload and the orbit that it is going to
#
# From the launchpad we would like to know the name of the launch site being used, the longitude, and the latitude.
#
# From cores we would like to learn the outcome of the landing, the type of the landing, number of flights with
# that core, whether gridfins were used, whether the core is reused, whether legs were used, the landing pad used,
# the block of the core which is a number used to seperate version of cores, the number of times this specific core
# has been reused, and the serial of the core.
#
# The data from these requests will be stored in lists and will be used to create a new dataframe.

#Global variables
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []

# Call getBoosterVersion
getBoosterVersion(data)
# Call getLaunchSite
getLaunchSite(data)
# Call getPayloadData
getPayloadData(data)
# Call getCoreData
getCoreData(data)

# Finally lets construct our dataset using the data we have obtained. We we combine the columns into a dictionary.
launch_dict = {'FlightNumber': list(data['flight_number']),
    'Date': list(data['date']),
    'BoosterVersion':BoosterVersion,
    'PayloadMass':PayloadMass,
    'Orbit':Orbit,
    'LaunchSite':LaunchSite,
    'Outcome':Outcome,
    'Flights':Flights,
    'GridFins':GridFins,
    'Reused':Reused,
    'Legs':Legs,
    'LandingPad':LandingPad,
    'Block':Block,
    'ReusedCount':ReusedCount,
    'Serial':Serial,
    'Longitude':Longitude,
    'Latitude':Latitude}

# Create a data from launch_dict
df_launch = pd.DataFrame.from_dict(launch_dict, orient='index').transpose()
# df_launch = df_launch.transpose()
# print(df_launch)

# Task 2: Filter the dataframe to only include Falcon 9 launches
data_falcon9 = df_launch[df_launch['BoosterVersion']!='Falcon 1']
# reset the FlightNumber column
data_falcon9.loc[:,'FlightNumber'] = list(range(1, data_falcon9.shape[0]+1))
# print(data_falcon9)

# Data Wrangling - Task 3: Dealing with Missing values
# missing values count
# print(data_falcon9.isnull().sum())
# Mean of payload
Payload_mean = data_falcon9['PayloadMass'].mean()
print("Mean of Payloadmass:", round(Payload_mean,2))
# Replace the NaN values with the mean
data_falcon9['PayloadMass'].replace(round(Payload_mean,2), np.nan, inplace=True)
# print(data_falcon9)
# Export to .csv
data_falcon9.to_csv('dataset_Module1_DataCollection.csv', index=False)
print("Falcon 9 data exported to .csv")
print("End of Hand-on lab Data Collection API")