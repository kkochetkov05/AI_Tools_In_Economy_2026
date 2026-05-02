from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  cross_val_score

def find_best_k(X, y, kf):
    scores = []
    for k in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=k)
        # Считаем качество на кросс-валидации
        score = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
        scores.append(score.mean())
    
    best_score = max(scores)
    best_k = scores.index(best_score) + 1 
    return best_k, best_score
