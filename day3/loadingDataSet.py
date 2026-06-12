import requests

response = requests.get("https://wttr.in/Hyderabad?format=j1")

data = response.json()

print(data)