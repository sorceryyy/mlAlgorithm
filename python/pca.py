import numpy as np
import sklearn

def pca(X, d = 3):
    """use principal components analysis to reduce demention
    arguments:
        X: (m * prev_d)data
        d: new demension

    return:
        data with lower demention
    """
    X = np.array(X)
    m, prev_d = X.shape[0], X.shape[1]

    mean_X = X - np.mean(X, axis=0)
    cov_mat = np.matmul(mean_X.T, mean_X)/(m-1)
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    idx = np.argsort(eig_val)[::-1] #!降序！
    eig_val, eig_vec = eig_val[idx], eig_vec[:,idx]
    new_X = np.matmul(mean_X, eig_vec[:,:d])
    return new_X


def diff_PCA(X , num_components = 3):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced



def main():
    np.random.seed(0)
    X = np.random.randint(10,50,100).reshape(20,5)
    x_d1 = pca(X)
    x_d2 = diff_PCA(X)
    diff = np.sum(x_d1-x_d2)
    print(f'difference:{diff:.3f}')


if __name__ == "__main__":
    main()
