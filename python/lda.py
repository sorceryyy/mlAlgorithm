import numpy as np
import sklearn

def LDA(X:np.ndarray, y, d=1):
    m, prev_d = X.shape[0], X.shape[1]
    mean_X = np.mean(X, axis=0)
    cls_label = np.unique(y)

    sw = np.zeros((prev_d,prev_d))
    sb = np.zeros((prev_d,prev_d))

    for c in cls_label:
        c_X = X[y==c]  # (c_m, prev_d)
        c_mean_X = np.mean(c_X, axis=0)  
        dc_X = c_X - c_mean_X
        sw += np.matmul(dc_X.T, dc_X)

        c_n = c_X.shape[0]
        diff_c = np.reshape((c_mean_X - mean_X), (prev_d, 1))
        sb +=  c_n * np.matmul(diff_c, diff_c.T)
    A = np.matmul(np.linalg.inv(sw), sb)
    eig_val, eig_vec = np.linalg.eig(A)
    idx = np.argsort(abs(eig_val))[::-1]
    eig_val, eig_vec = eig_val[idx], eig_vec[:,idx]
    
    new_X = np.matmul(X, eig_vec[:,:d])
    return new_X
    
def diff_LDA(X, y, d = 1):
    n_features = X.shape[1]
    class_labels = np.unique(y)

    # Within class scatter matrix:
    # SW = sum((X_c - mean_X_c)^2 )

    # Between class scatter:
    # SB = sum( n_c * (mean_X_c - mean_overall)^2 )

    mean_overall = np.mean(X, axis=0)
    SW = np.zeros((n_features, n_features))
    SB = np.zeros((n_features, n_features))
    for c in class_labels:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        # (4, n_c) * (n_c, 4) = (4,4) -> transpose
        SW += (X_c - mean_c).T.dot((X_c - mean_c))

        # (4, 1) * (1, 4) = (4,4) -> reshape
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
        SB += n_c * (mean_diff).dot(mean_diff.T)

    # Determine SW^-1 * SB
    A = np.linalg.inv(SW).dot(SB)
    # Get eigenvalues and eigenvectors of SW^-1 * SB
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # -> eigenvector v = [:,i] column vector, transpose for easier calculations
    # sort eigenvalues high to low
    eigenvectors = eigenvectors.T
    idxs = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    # store first n eigenvectors
    linear_discriminants = eigenvectors[0:d]

    return np.dot(X, linear_discriminants.T)

        




def main():
    np.random.seed(0)
    X1 = np.random.randint(5,size=(20,5))
    y1 = np.zeros((20))

    X2 = np.random.randint(3,8,size=(20,5))
    y2 = np.ones((20))

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    x_d1 = LDA(X, y)
    x_d2 = diff_LDA(X, y)
    diff = np.sum(x_d1-x_d2)
    print(f'difference:{diff:.3f}')

if __name__ == "__main__":
    main()