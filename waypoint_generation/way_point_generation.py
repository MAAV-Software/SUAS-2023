import random

starting_coord = (0, 0, 0) # degree starting position plus height
total_way_points = 10
total_distance = 3/69 # 3 miles in degrees (1 degree = 69 miles)
minHeight = 100 # agl or would be 217 feet msl
maxHeight = 125
way_points = []
way_points.append(starting_coord)
#generate ten random points where the total distance between them and the starting point is less than total_distance
for i in range(1, total_way_points): #run through are total points
    x = random.uniform(-total_distance/2, total_distance/2) #generate a random x and y coordinate that is within the total distance remaining
    y = random.uniform(-total_distance/2, total_distance/2)
    height = random.uniform(minHeight, maxHeight)
    way_points.append((x, y, height))
    total_distance -= (((abs(x)-abs(way_points[i-1][0]))**2+(abs(y)-abs(way_points[i-1][1]))**2)**0.5) #subtract the distance between the two points from the total distance so that we stay within 10
    if total_distance < 0:
        total_distance = 0
        break
    # print(total_distance*69)
# print(total_distance*69)
f = open("wayPoints.txt", "w")
for i in way_points:
    f.write(str(i))
    f.write("\n")
# f.write(str(way_points))
# print(way_points)
