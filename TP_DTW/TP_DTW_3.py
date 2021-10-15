import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from tslearn.clustering import TimeSeriesKMeans
listofclass = []
listoflist = []
# lecture du fichier Plane et création de la liste des séries temporelles et de
# la liste des classes
with open('/home/bertrand/Cours Bioinfo/DCD/UCR_TS_Archive_2015/Plane/Plane_TRAIN') as f:
    for l in f:
        classe = l.split(",")[0] # récupère la classe de la série temporelle
        liste = l.split(",")[1:] # récupère la série temporelle
        liste = [i.strip("\n") for i in liste] # enlève le \n de la fin de la série temporelle
        listofclass.append(classe) # créer une liste des classes
        listoflist.append(liste) # créer une liste des séries temporelles

test_list = [int(i) for i in listofclass] # converti les valeurs en numériques
my_array = np.array(listoflist,dtype=float) # converti la liste en matrice

model = TimeSeriesKMeans(n_clusters=7, metric="dtw")
prediction = model.fit_predict(my_array)
print(prediction)
print(f"Score de Rand pour le clustering K-means avec distance DTW: {adjusted_rand_score(prediction,test_list)}") # calcul du score de Rand pour estimer

model1 = TimeSeriesKMeans(n_clusters=7)
prediction1 = model1.fit_predict(my_array)
print(prediction1)
print(f"Score de Rand pour le clustering K-means avec distance euclidienne: {adjusted_rand_score(prediction1,test_list)}") # calcul du score de Rand pour estimer
