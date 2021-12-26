import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA:
    def fit_transform(self, X, n_components=2):
        # get number of samples and components
        self.n_samples = X.shape[0]
        self.n_components = n_components
        
        # mean center the original data-matrix
        self.A = self.mean_center(X)
        
        # calculate covariance matrix
        covariance_matrix = self.get_covariance_matrix()
        
        # retrieve selected eigenvectors
        eigenvectors = self.get_eigenvectors(covariance_matrix)
        
        # project into lower dimension
        projected_matrix = self.project_matrix(eigenvectors)
        return projected_matrix

    def mean_center(self, X):
        # calculate the mean columnwise and substract from data
        return X - np.mean(X, axis=0)
    
    def get_covariance_matrix(self):
        # calculate covariance matrix with mean-centered matrix A
        C = np.dot(self.A.T, self.A) / self.n_samples
        return C

    def get_eigenvectors(self, C):
        # calculate eigenvalues & eigenvectors of covariance matrix 'C'
        eigenvalues, eigenvectors = np.linalg.eig(C)

        # sort eigenvalues descending and select columns based on n_components
        n_cols = np.argsort(eigenvalues)[::-1][:self.n_components]
        selected_vectors = eigenvectors[:, n_cols]
        return selected_vectors

    def project_matrix(self, eigenvectors):
        P = np.dot(self.A, eigenvectors)
        return P

# Testing
if __name__ == "__main__":
    # load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # instantiate and fit_transform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X, n_components=2)

    # plot results
    fig, ax = plt.subplots(1, 1, figsize=(10,6))

    sns.scatterplot(
        x = X_pca[:,0],
        y = X_pca[:,1],
        hue=y
    )

    ax.set_title('Iris Dataset')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    sns.despine()
    plt.show()