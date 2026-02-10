import requests
import pandas as pd
import matplotlib.pyplot as plt
import os as os
from datetime import datetime, timedelta
from typing import cast, TypeAlias
from dataclasses import dataclass

# ################################################################ #
#                         API PREPARATION                          #
# ################################################################ #

"""
sample response:    
{
    'latitude': 48.84, 
    'longitude': 2.3599997, 
    'generationtime_ms': 0.05257129669189453, 
    'utc_offset_seconds': 0, 
    'timezone': 'GMT', 
    'timezone_abbreviation': 'GMT', 
    'elevation': 46.0, 
    'daily_units': {
        'time': 'iso8601', 
        'temperature_2m_max': '°C', 
        'temperature_2m_min': '°C'
    }, 
    'daily': {
        'time': ['2026-01-27', '2026-01-28', '2026-01-29', '2026-01-30', '2026-01-31', '2026-02-01', '2026-02-02', '2026-02-03'], 
        'temperature_2m_max': [10.0, 8.3, 5.0, 11.7, 9.5, 9.7, 10.1, 10.4], 
        'temperature_2m_min': [5.1, 2.9, 0.3, 4.6, 5.3, 6.0, 3.1, 7.3] 
    }
}
"""

@dataclass
class dataParam:
    latitude: str
    longitude: str
    startDate: str
    endDate: str

typeData: TypeAlias = dict[str, float | int | str | dict[str, str] | dict[str, list[str | float]]]

def fetch_weather_data(param: dataParam) -> typeData:
    url: str = f"https://api.open-meteo.com/v1/forecast?latitude={param.latitude}&longitude={param.longitude}&start_date={param.startDate}&end_date={param.endDate}&daily=temperature_2m_max,temperature_2m_min"
    response: requests.Response = requests.get(url)
    return response.json()

# ################################################################ #
#                     SEND REQUEST & GET DATA                      #
# ################################################################ #

# today's date & 7 days delta
today : datetime = datetime.now()
delta: timedelta = timedelta(days=7)

# Format dates for API (YYYY-MM-DD)
# Start Date - 1 week Before
# End Date - Today
start_date : str = (today - delta).strftime("%Y-%m-%d")
end_date : str = today.strftime("%Y-%m-%d")

data: typeData = fetch_weather_data(
    param=dataParam(
        latitude="48.8566",      # Paris Latitude
        longitude="2.3522",      # Paris Longitude
        startDate=start_date,
        endDate=end_date
    )
)


# ################################################################ #
#                   FORMAT DATA INTO TABLE FORM                    #
# ################################################################ #

table: pd.DataFrame = pd.DataFrame({
    'date': cast(dict[str, list[str | float]], data['daily'])['time'],
    'max temperature': cast(dict[str, list[str | float]], data['daily'])['temperature_2m_max'],
    'min temperature': cast(dict[str, list[str | float]], data['daily'])['temperature_2m_min']
})

# the date column is currently string type
# convert date column to `datetime64[us]` type
# datetime64 - time as nanoseconds in a 64-bit signed integer
table['date'] = pd.to_datetime(table['date'])

# Min Boundary: September 21, 1677
# Max Boundary: April 11, 2262

# this will print out the data in table format
# rather than in json format
print(table)

# ################################################################ #
#                     CONVERT INTO GRAPH/CHART                     #
# ################################################################ #

# to silent the warning errors
# install this:
# 
# ````sh
# pip install matplotlib-stubs
# ````
# 

# create new blank canvas (called figure)
# 10 in width, 6 in height, in inches
# default size is (6.4, 4.8 inches)
plt.figure(figsize=(10, 6))

# plot the coordinates
plt.plot(table['date'], table['max temperature'], marker='o', label='Max Temp')
plt.plot(table['date'], table['min temperature'], marker='o', label='Min Temp')

# add labels to x-axis and y-axis
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')

# add title (display above the chart)
plt.title('Paris Weather - Past 7 Days')

# add legend (the instruction that explains the marker lines) to differentiate the lines
plt.legend()

# Rotate x-axis labels for readability
# by default values are display horizontally
# we tilt them 45 degrees so that values don't overlap if too close
plt.xticks(rotation=45)

# adjust margin/padding so that the graphs and labels fit well
plt.tight_layout()

# save the graph as an image file
plt.savefig('weather_chart.png')

# display the chart
# open GUI window to show the chart
plt.show()

# NOTE
# this .show() function will pause the code execution
# until you close the chart window

# ################################################################ #
#                         CONVERT INTO CSV                         #
# ################################################################ #

# check if 'data' folder exists, if not, create it
if not os.path.exists('data'):
    os.makedirs('data')

# Save to CSV

# when index=True
# ,City,Temperature,Humidity
# 0,Paris,15,65
# 1,London,12,70
# 2,Tokyo,18,55

# when index=False
# City,Temperature,Humidity
# Paris,15,65
# London,12,70
# Tokyo,18,55

table.to_csv('data/paris_weather.csv', index=False)

print("\nData saved to data/paris_weather.csv")