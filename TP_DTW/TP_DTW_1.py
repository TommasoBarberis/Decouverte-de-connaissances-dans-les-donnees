# script permettant de faire de la classification hiérarchique avec les distances 
# euclidiennes et les distances Dynamic Time Warping 
# ce script compare les résultats obtenus par les deux distances par le calcul 
# du score de Rand

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import complete, fcluster
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

listofclass = []
listoflist = []
i=0
# lecture du fichier Plane et création de la liste des séries temporelles et de
# la liste des classes
with open('/home/bertrand/Cours Bioinfo/DCD/UCR_TS_Archive_2015/Plane/Plane_TRAIN') as f:
    for l in f:
        i+=1
        classe = l.split(",")[0] # récupère la classe de la série temporelle
        liste = l.split(",")[1:] # récupère la série temporelle
        liste = [i.strip("\n") for i in liste] # enlève le \n de la fin de la série temporelle
        listofclass.append(classe) # créer une liste des classes
        listoflist.append(liste) # créer une liste des séries temporelles

test_list = [int(i) for i in listofclass] # converti les valeurs en numériques
my_array = np.array(listoflist) # converti la liste en matrice

### Classification hiérarchique avec distance euclidienne

distance = pdist(my_array) # calcul de la distance euclidienne entre les séries temporelles
Z = complete(distance) # classification hiérarchique par la méthode complète
clust = fcluster(Z, 7, criterion='maxclust') # assignation des séries temporelles 
# aux 7 clusters correspondant aux 7 classes
clust = list(clust)
print(f"Score de Rand pour les distances euclidiennes: {adjusted_rand_score(clust,test_list)}") # calcul du score de Rand pour estimer
# la correspondance entre les clusters créés et les classes initiales

### Classification hiérarchique avec distance du Dynamic Time Warping 

n_series = len(listoflist)
distance_matrix = np.zeros(shape=(n_series, n_series))
for i in range(n_series):
    for j in range(n_series):
        x = (listoflist[i])
        y = (listoflist[j])
        if i != j:
            distance, path = fastdtw(x, y) # calcul de la distance DTW entre 
# les séries temporelles
            distance_matrix[i, j] = distance
Y = complete(distance_matrix) # classification hiérarchique par la méthode complète
clust1 = fcluster(Y, 7, criterion='maxclust')  # assignation des séries temporelles 
# aux 7 clusters correspondant aux 7 classes
clust1 = list(clust1)
print(f"Score de Rand pour les distances du DTW: {adjusted_rand_score(clust1,test_list)}") # calcul du score de Rand pour estimer
# la correspondance entre les clusters créés et les classes initiales