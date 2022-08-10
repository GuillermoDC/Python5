# Module 1 Python 5 of 5 (Capstone project).
# Hands-on Lab Complete Data Collection with Web Scraping
# Guillermo Dominguez, PhD.

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

# ===== 2 - Web Scraping ====== #
# import required packages for this lab --> added with PyCharm to Python Interpreter 3.9
import sys
import requests
from bs4 import BeautifulSoup as bs
import re
import unicodedata
import pandas as pd

# helper functions to process web scraped HTML table
def date_time(table_cells):
    """
    This function returns the data and time from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]
def booster_version(table_cells):
    """
    This function returns the booster version from the HTML  table cell
    Input: the  element of a table data cell extracts extra row
    """
    out = ''.join([booster_version for i, booster_version in enumerate(table_cells.strings) if i % 2 == 0][0:-1])
    return out
def landing_status(table_cells):
    """
    This function returns the landing status from the HTML table cell
    Input: the  element of a table data cell extracts extra row
    """
    out = [i for i in table_cells.strings][0]
    return out
def get_mass(table_cells):
    mass = unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass = mass[0:mass.find("kg") + 2]
    else:
        new_mass = 0
    return new_mass
def extract_column_from_header(row):
    """
    This function returns the landing status from the HTML table cell
    Input: the  element of a table data cell extracts extra row
    """
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()

    colunm_name = ' '.join(row.contents)

    # Filter the digit and empty names
    if not (colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name

# for data consistency: data from a snapshot of the List of Falcon 9 and Falcon Heavy launches Wikipage updated on 9th June 2021
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

# TASK 1: Request the Falcon9 Launch Wiki page from its URL
# perform an HTTP GET method to request the Falcon9 Launch HTML page, as an HTTP response.
response = requests.get(static_url)
# print(response.content) # it is a mess, needs to be normalized
# print(response.status_code) # 200 means is correct

# assign the response to an object
# df = response.json()
# data = pd.json_normalize(df)
# print("Static URL request")
# print(df)

# Create a BeautifulSoup object from the HTML response
# soup = bs(response)
# soup = bs(response.text, "html.parser")
soup = bs(response.content)
# print(soup)
# print(soup.prettify())

# Check if BeautifulSoup worked
# Use soup.title attribute
print(soup.title)

# TASK 2: Extract all relevant column/variable names from the HTML table header
# Let's try to find all tables on the wiki page first.
# Use the find_all function in the BeautifulSoup object, with element type `table`
# Assign the result to a list called `html_tables`
html_tables = soup.find_all('table')
# print(html_tables)

# 3rd table contains Launch records
first_launch_table = html_tables[2]
# print(first_launch_table)

# Apply find_all() function with `th` element on first_launch_table
first_launch_table.find_all('th')

#  Iterate through the <th> elements and apply the provided extract_column_from_header() to extract column name one by one
# Append the Non-empty column name (`if name is not None and len(name) > 0`) into a list called column_names
column_names = []
for x in first_launch_table.find_all('th'):
    name = extract_column_from_header(x)
    if name is not None and len(name) > 0:
        column_names.append(name)
# print(column_names)

# TASK 3: Create a data frame by parsing the launch HTML tables
# Create an empty dictionary with keys from the extracted column names in the previous task.
# Dictionary will be converted into a Pandas dataframe
launch_dict = dict.fromkeys(column_names)

# Remove an irrelevant column
del launch_dict['Date and time ( )']

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster'] = []
launch_dict['Booster landing'] = []
launch_dict['Date'] = []
launch_dict['Time'] = []

# fill up the launch_dict with launch records extracted from table rows

# To simplify the parsing process, we have provided an incomplete code snippet below to help you to fill up the launch_dict.
# Please complete the following code snippet with TODOs or you can choose to write your own logic to parse all launch tables:
extracted_row = 0
# Extract each table
for table_number, table in enumerate(soup.find_all('table', "wikitable plainrowheaders collapsible")):
    # get table row
    for rows in table.find_all("tr"):
        # check to see if first table heading is as number corresponding to launch a number
        if rows.th:
            if rows.th.string:
                flight_number = rows.th.string.strip()
                flag = flight_number.isdigit()
        else:
            flag = False
        # get table element
        row = rows.find_all('td')
        # if it is number save cells in a dictonary
        if flag:
            extracted_row += 1
            # Flight Number value
            # TODO: Append the flight_number into launch_dict with key `Flight No.`
            launch_dict["Flight No."].append(flight_number)
            # print(flight_number)
            datatimelist = date_time(row[0])

            # Date value
            # TODO: Append the date into launch_dict with key `Date`
            date = datatimelist[0].strip(',')
            launch_dict["Date"].append(date)
            # print(date)

            # Time value
            # TODO: Append the time into launch_dict with key `Time`
            time = datatimelist[1]
            launch_dict["Time"].append(time)
            # print(time)

            # Booster version
            # TODO: Append the bv into launch_dict with key `Version Booster`
            bv = booster_version(row[1])
            if not (bv):
                bv = row[1].a.string
            launch_dict["Version Booster"].append(bv)
            # print(bv)

            # Launch Site
            # TODO: Append the bv into launch_dict with key `Launch site`
            launch_site = row[2].a.string
            launch_dict["Launch site"].append(launch_site)
            # print(launch_site)

            # Payload
            # TODO: Append the payload into launch_dict with key `Payload`
            payload = row[3].a.string
            launch_dict["Payload"].append(payload)
            # print(payload)

            # Payload Mass
            # TODO: Append the payload_mass into launch_dict with key `Payload mass`
            payload_mass = get_mass(row[4])
            launch_dict["Payload mass"].append(payload_mass)
            # print(payload)

            # Orbit
            # TODO: Append the orbit into launch_dict with key `Orbit`
            orbit = row[5].a.string
            launch_dict["Orbit"].append(orbit)
            # print(orbit)

            # Customer
            # TODO: Append the customer into launch_dict with key `Customer`
            # ===> Found in (https://www.kaggle.com/code/bryanraybantilan/notebookac2cfbf3e6/notebook)
            # Try Except to handle badly formed data from table in Customer column.
            try:
                customer = row[6].a.string
            except(TypeError, AttributeError) as e:
                customer = "Various"
            launch_dict["Customer"].append(customer)
            # print(customer)

            # Launch outcome
            # TODO: Append the launch_outcome into launch_dict with key `Launch outcome`
            launch_outcome = list(row[7].strings)[0]
            launch_dict["Launch outcome"].append(launch_outcome)
            # print(launch_outcome)

            # Booster landing
            # TODO: Append the launch_outcome into launch_dict with key `Booster landing`
            booster_landing = landing_status(row[8])
            launch_dict["Booster landing"].append(booster_landing)
            # print(booster_landing)

# After filling in the parsed launch record values into launch_dict, create a dataframe from it:
df = pd.DataFrame(launch_dict)
# print("[=== Web Scrapped values from Table 3 ===]")
print(df.head(20))
# Export to .csv
df.to_csv('spacex_web_scraped.csv', index=False)
print("Falcon 9 data exported to .csv")
print("End of Hand-on lab Data Collection Web Scraping")

