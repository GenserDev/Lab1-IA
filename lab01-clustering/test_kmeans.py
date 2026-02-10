#!/usr/bin/env python3
"""
Script de prueba rápida para verificar la implementación de K-Means
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sys
sys.path.append('src')
from kmeans import KMeans

def test_kmeans():
    """Prueba rápida de K-Means"""
    print("🧪 Probando implementación de K-Means...")
    
    # Generar datos de prueba
    X, y_true = make_blobs(n_samples=300, centers=4, 
                           cluster_std=0.60, random_state=42)
    
    print(f"\n📊 Datos generados: {X.shape}")
    
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    
    print(f"✅ K-Means completado")
    print(f"   Inercia: {kmeans.inertia_:.2f}")
    print(f"   Centroides: {kmeans.centroids.shape}")
    
    # Visualizar
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.6)
    plt.title('Ground Truth')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
                c='red', marker='X', s=200, edgecolors='black', label='Centroides')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/test_kmeans.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Gráfica guardada en: results/test_kmeans.png")
    plt.show()
    
    print("\n✅ ¡Test completado exitosamente!")

if __name__ == '__main__':
    test_kmeans()
