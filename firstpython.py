url = https://api.spacexdata.com/v4/launches/past
response = erquests.get(url)
response.json()
data = pd.json_normalize(response.json())
