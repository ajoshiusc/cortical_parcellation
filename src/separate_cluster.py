import scipy as sp

from centroid import  modified_find_centroid


def separate(labels, r,temp):
    count1 = count2 = count3 =count_4=count_5 = 0
    c1 = []
    c2 = []
    c3 = []
    c4=[]
    c4=[]
    for i in range(r.labels.shape[0]):
        if r.labels[i] == 1:
            count1 = count1 + 1
            if count1 == 1:
                c1 = sp.array(r.vertices[i])
            else:
                c1 = sp.vstack([c1, r.vertices[i]])
        elif r.labels[i] == 2:
            count2 = count2 + 1
            if count2 == 1:
                c2 = sp.array(r.vertices[i])
            else:
                c2 = sp.vstack([c2, r.vertices[i]])
        elif r.labels[i] == 3:
            count3 = count3 + 1
            if count3 == 1:
                c3 = sp.array(r.vertices[i])
            else:
                c3 = sp.vstack([c3, r.vertices[i]])
        elif r.labels[i] == 4:
            count_4 = count_4 + 1
            if count_4 == 1:
                c4 = sp.array(r.vertices[i])
            else:
                c4 = sp.vstack([c4, r.vertices[i]])
        elif r.labels[i] == 5:
            count_5 = count_5 + 1
            if count_5 == 1:
                c5 = sp.array(r.vertices[i])
            else:
                c5 = sp.vstack([c5, r.vertices[i]])
    centroid_1=sp.array(modified_find_centroid(c1))
    centroid_2=sp.array(modified_find_centroid(c2))
    #centroid_3=sp.array(modified_find_centroid(c3))
    #return (centroid_1,centroid_2,centroid_3)
    return(centroid_1,centroid_2)
    #return (centroid_1, centroid_2, centroid_3,centroid_4,centroid_5)