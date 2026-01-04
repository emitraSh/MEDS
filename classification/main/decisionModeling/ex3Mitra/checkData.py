import csv

arcs = set()
cities = set()
distances = dict()

with open('travelTimes50Cities.csv', mode='r' , encoding='utf-8-sig') as file:
    reader = csv.DictReader(file, delimiter=';')

    for row in reader:
#        origin = row['origin']
#        destination = row['destination']
        distance_miles = (row['distance_miles'])
