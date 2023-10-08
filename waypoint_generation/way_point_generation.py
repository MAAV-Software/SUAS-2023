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

#we're going to create an sdf file with the points
# <spherical_coordinates>
#   <surface_model>EARTH_WGS84</surface_model>
#   <world_frame_orientation>ENU</world_frame_orientation>
#   <latitude_deg>-22.9</latitude_deg>
#   <longitude_deg>-43.2</longitude_deg>
#   <elevation>0</elevation>
#   <heading_deg>0</heading_deg>
# </spherical_coordinates>



# <light name='user_way_Point_0' type='point'>
#       <pose>2.17619 -2.70115 1 0 -0 0</pose>
#       <diffuse>0.5 0.5 0.5 1</diffuse>
#       <specular>0.1 0.1 0.1 1</specular>
#       <attenuation>
#         <range>20</range>
#         <constant>0.5</constant>
#         <linear>0.01</linear>
#         <quadratic>0.001</quadratic>
#       </attenuation>
#       <cast_shadows>0</cast_shadows>
#       <direction>0 0 -1</direction>
#       <spot>
#         <inner_angle>0</inner_angle>
#         <outer_angle>0</outer_angle>
#         <falloff>0</falloff>
#       </spot>
#     </light>


f = open("wayPoints.sdf", "w")
f.write("<sdf version='1.7'>\n")
f.write("<world name='default'>\n")
for i in range(0, len(way_points)):
    f.write("<light name='user_way_Point_" + str(i) + "' type='point'>\n")
    f.write("\t<pose>" + str(way_points[i][0]*69*5280) + " " + str(way_points[i][1]*69*5280) + " " + str(way_points[i][2]) + " 0 -0 0</pose>\n")
    f.write("\t<diffuse>0.5 0.5 0.5 1</diffuse>\n")
    f.write("\t<specular>0.1 0.1 0.1 1</specular>\n")
    f.write("\t<attenuation>\n")
    f.write("\t\t<range>20</range>\n")
    f.write("\t\t<constant>0.5</constant>\n")
    f.write("\t\t<linear>0.01</linear>\n")
    f.write("\t\t<quadratic>0.001</quadratic>\n")
    f.write("\t</attenuation>\n")
    f.write("\t<cast_shadows>0</cast_shadows>\n")
    f.write("\t<direction>0 0 -1</direction>\n")
    f.write("\t<spot>\n")
    f.write("\t\t<inner_angle>0</inner_angle>\n")   
    f.write("\t\t<outer_angle>0</outer_angle>\n")
    f.write("\t\t<falloff>0</falloff>\n")
    f.write("\t</spot>\n")
    f.write("</light>\n")
    
# f.write(str(way_points))
# print(way_points)
f.write("</world>\n")
f.write("</sdf>\n")