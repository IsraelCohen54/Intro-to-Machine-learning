import scipy.io.wavfile
import sys
import numpy as np
#np.set_printoptions(precision=10)
np.set_printoptions(suppress=True)

sample, centroids = sys.argv[1],sys.argv[2]
fs, y = scipy.io.wavfile.read(sample) #reading
x=np.array(y.copy())
centroids=np.loadtxt(centroids)

#my code:
iter_counter = 30

file = open(r"C:\Users\Israel\PycharmProjects\pythonProject1\output.txt","w")

#Boolian to check if convergance:
Not_converganced = False
update = False
pointsToCentroids=np.empty(32000, int)

#lastCentroid = np.empty([20], float) #checing cenr convergance for debbug purpose
lastCentroid = np.ones_like(centroids)*999999999999 #checing cenr convergance for debbug purpose

new_values=[] #compressed points as centroids only
minimal_dis=10000000000000.
counterForAvg = 0
point=np.array([0.0,0.0])
#while loop - checking if convergance than stop
while iter_counter!=0 and Not_converganced == False:
    iter_counter-=1
    #assign each point in X to the cluster with the nearest representative
    #going through all points
    for index_points in range(len(x)):
        # going through all centroids for each point
        for index_Centroids in range(len(centroids)):
            #checking dis point to centroids, oclides dis
            current_dis = ((x[index_points][0] - centroids[index_Centroids][0])**2+(x[index_points][1] - centroids[index_Centroids][1])**2)
            if current_dis < minimal_dis:
                minimal_dis = current_dis
                min_dis_to_centroid_centindex=index_Centroids
                #update = True #so we shouldn't convergance
        #add centroid index to list equal to the points array
        pointsToCentroids[index_points]=min_dis_to_centroid_centindex
        minimal_dis = 1000000000000000.
    #changing centroids locations, for all centroids:
    for index_Centroids2 in range(len(centroids)):
        #change its location to avg of his points
        for pointsToCentroids_index in range(len(pointsToCentroids)):
            if(pointsToCentroids[pointsToCentroids_index]==index_Centroids2):
                point[0]+= x[pointsToCentroids_index][0]
                point[1] += x[pointsToCentroids_index][1]
                counterForAvg+=1
        #div by counter for X and Y axid, than apply to current centroid
        if counterForAvg!=0:
            point[0] = point[0]/counterForAvg
            point[1] = point[1]/counterForAvg
        #centroids.insert(index_Centroids2, point)
        centroids[index_Centroids2][0] = round(point[0])
        centroids[index_Centroids2][1] = round(point[1])
        point = [0.0, 0.0]
        counterForAvg = 0

    print("Iteration ended ", 29-iter_counter)
    print(centroids)

    str1 = "[iter " + str(29-iter_counter) + "]:"

    file.write(str1)
    for i in range (len(centroids)):
        file.write(str(centroids[i]))
        if i!=len(centroids)-1:
            file.write(",")
    file.write("\n")

    diff = centroids-lastCentroid
    diff_zeroes = np.zeros_like(centroids)
    if diff.any() != diff_zeroes.any():
        print ("continue")
    else:
        break
    lastCentroid = centroids.copy()

file.close()

#change points value to centroids values to commpress
for index_points_to_compress in range(len(pointsToCentroids)):
    new_values.append(centroids[min_dis_to_centroid_centindex])

scipy.io.wavfile.write("compressed.wav", fs ,np.array(new_values, dtype=np.int16))#saving
