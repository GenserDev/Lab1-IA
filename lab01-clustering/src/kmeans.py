"""
Implementación de K-Means desde cero
Sin usar librerías especializadas de clustering
"""

import numpy as np


class KMeans:
    """
    Implementación de K-Means clustering desde cero.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Número de clusters a formar
    max_iter : int, default=300
        Número máximo de iteraciones
    random_state : int, default=None
        Semilla para reproducibilidad
    tol : float, default=1e-4
        Tolerancia para convergencia
    """
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=None, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def _initialize_centroids(self, X):
        """
        Inicializar centroides aleatoriamente
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[random_indices]
        return centroids
    
    def _compute_distances(self, X, centroids):
        """
        Calcular distancia euclidiana entre cada punto y cada centroide
        
        Returns
        -------
        distances : array, shape (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for k in range(n_clusters):
            # Distancia euclidiana: sqrt(sum((x - centroid)^2))
            diff = X - centroids[k]
            distances[:, k] = np.sqrt(np.sum(diff ** 2, axis=1))
        
        return distances
    
    def _assign_clusters(self, X, centroids):
        """
        Asignar cada punto al cluster más cercano
        
        Returns
        -------
        labels : array, shape (n_samples,)
        """
        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Actualizar centroides como el promedio de los puntos en cada cluster
        
        Returns
        -------
        centroids : array, shape (n_clusters, n_features)
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Puntos asignados al cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Promedio de los puntos
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Si un cluster está vacío, reinicializar aleatoriamente
                centroids[k] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """
        Calcular inercia (suma de distancias al cuadrado)
        """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                diff = cluster_points - centroids[k]
                inertia += np.sum(diff ** 2)
        return inertia
    
    def fit(self, X):
        """
        Ajustar el modelo K-Means
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Datos de entrenamiento
            
        Returns
        -------
        self
        """
        # Convertir a numpy array si es necesario
        X = np.array(X)
        
        # Inicializar centroides
        self.centroids = self._initialize_centroids(X)
        
        # Iteraciones del algoritmo
        for iteration in range(self.max_iter):
            # Guardar centroides anteriores para verificar convergencia
            old_centroids = self.centroids.copy()
            
            # Paso 1: Asignar clusters
            self.labels = self._assign_clusters(X, self.centroids)
            
            # Paso 2: Actualizar centroides
            self.centroids = self._update_centroids(X, self.labels)
            
            # Verificar convergencia
            centroid_shift = np.sum((self.centroids - old_centroids) ** 2)
            if centroid_shift < self.tol:
                print(f"Convergió en la iteración {iteration + 1}")
                break
        
        # Calcular inercia final
        self.inertia_ = self._compute_inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """
        Predecir el cluster más cercano para cada muestra
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            
        Returns
        -------
        labels : array, shape (n_samples,)
        """
        X = np.array(X)
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """
        Ajustar el modelo y devolver las etiquetas
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            
        Returns
        -------
        labels : array, shape (n_samples,)
        """
        self.fit(X)
        return self.labels


def elbow_method(X, max_clusters=10, random_state=42):
    """
    Método del codo para determinar el número óptimo de clusters
    
    Parameters
    ----------
    X : array-like
        Datos
    max_clusters : int
        Número máximo de clusters a probar
    random_state : int
        Semilla para reproducibilidad
        
    Returns
    -------
    inertias : list
        Lista de inercias para cada k
    """
    inertias = []
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return inertias
