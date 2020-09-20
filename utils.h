#include<vector>
#include<cassert>

// Multiplication of two matrices
// This considers locality of reference
template<class T>
std::vector<std::vector<T> > matmul(std::vector<std::vector<T> > A, std::vector<std::vector<T> > B) {
    int r1 = A.size(), c1 = A[0].size();
    int r2 = B.size(), c2 = B[0].size();
    std::vector<std::vector<T> > result(r1, std::vector<T>(c2, 0.0));
    for(int i=0; i<r1; i++) {
        for(int j=0; j<c2; j++) {
            for(int k=0; k<r2; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Transpose of matrix
template<class T>
std::vector<std::vector<T> > transpose(std::vector<std::vector<T>> A) {
    std::vector<std::vector<T> > result(A[0].size(), std::vector<T>(A.size(), 0.0));
    for(int i=0; i<A.size(); i++) {
        for(int j=0; j<A[i].size(); j++) {
            result[j][i] = A[i][j];
        }
    }   
    return result;
}


// Subtraction of two matrices
template<class T>
std::vector<std::vector<T> > subtract(std::vector<std::vector<T> > A, std::vector<std::vector<T> > B) {
    int r1 = A.size(), c1 = A[0].size(), r2 = B.size(), c2 = B[0].size();

    std::vector<std::vector<T> > result(r1, std::vector<T>(c1, 0.0));
    for(int i=0; i<r1; i++) {
        for(int j=0; j<c1; j++) {
            if(i >= 0 && i < r2 && j >= 0 && j < c2) {
                result[i][j] = A[i][j] - B[i][j];
            }
        }
    }

    return result;
}

// Matrix multiplication with Scalar
template<class T>
std::vector<std::vector<T> > multiply_matrix_with_scalar(std::vector<std::vector<T> > A, double B) {
    int r1 = A.size(), c1 = A[0].size();
    std::vector <std::vector<T>> result(r1, std::vector<T>(c1, 0.0));
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
            result[i][j] = A[i][j] * B;
        }
    }
    return result;
}