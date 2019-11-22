#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

// Definition of the kNN result struct
typedef struct knnresult
{
    int * nidx;     //!< Indices (0-based) of nearest neighbors
    double * ndist; //!< Distance of nearest neighbors
    int m;          //!< Number of query points
    int k;          //!< Number of nearest neighbors
} knnresult;


// Function Prototypes
double* calculateD(double * X, double * Y, int n, int m, int d, int k);
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);
void swap_double(double *a, double *b);
void swap_int(int *a, int *b);
int partition (double arr[], int *ids, int low, int high);
double kselect(double arr[], int *ids, int length, int k);
double quickselect(double arr[], int *ids, int length, int idx);


//! Compute k nearest neighbors of each point in X [n-by-d]
/*
 * \paramX	Corpus data points	[n-by-d]
 * \paramY	Query data points	[m-by-d]
 * \paramn	Number of corpus points	[scalar]
 * \paramm	Number of query points	[scalar]
 * \paramd	Number of dimensions	[scalar]
 * \paramk	Number of neighbors	[scalar]
 * \return	The kNN result
 */


// Application Entry Point
knnresult kNN(double * X, double * Y, int n, int m, int d, int k)
{

	// Calculate distances matrix D - D is row-major and nxm
	double* D = calculateD(X, Y, n, m, d, k);

	printf("Flag\n");

	// Transpose D to mxn
	cblas_dimatcopy(CblasRowMajor, CblasTrans, n, m, 1.0, D, m, n);

	// Create results struct
	knnresult results;
	results.k = k;
	results.m = m;
	results.ndist = calloc(m*k, sizeof(double));
	results.nidx = calloc(m*k, sizeof(int));


	// Create ids array for the X array
	int* ids = calloc(n, sizeof(int));


	// K-Select using quickselect to find k smallest distances for each point of Y
	for(int j = 0; j < m; j++){

		// Re-set ids vector before executing quickselect each time
		for(int i = 0; i < n; i++)
				ids[i] = i;

		// Call quickselect on the row of D to sort it up to its k-th element
		kselect(D + j * n, ids, n, k);

		// Write results (point id and distance)
		for(int l = 0; l < k; l++){

			results.ndist[j * k + l] = D[j * n + l];

			//TODO: Code up to here checks out fine - something is wrong with ids though and always returns 0
			results.nidx[j * k + l] = ids[l];
		}
	}

	// Free memory and return results
	free(ids);
	return results;

}

// Function used to calculate D = sqrt(sum(X.^2,2) -2* X*Y.' + sum(Y.^2,2).');
double* calculateD(double * X, double * Y, int n, int m, int d, int k){

		// Temporary variable
		double temp = 0;

		// Distance matrix for results
		double* D = calloc(n*m, sizeof(double));

		// Euclidean norm vectors
		double* normX = calloc(n, sizeof(double));
		double* normY = calloc(m, sizeof(double));

		// n and m sized one vectors
		double* onesN = calloc(n, sizeof(double));
		double* onesM = calloc(m, sizeof(double));
		for(int i = 0; i < n; i++)
			onesN[i] = 1;
		for(int i = 0; i < m; i++)
			onesM[i] = 1;

		// Matrice to store -2*X*Y'
		double* XY = calloc(n*m, sizeof(double));

		// Calculate -2*XY (https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d, 0, XY, m);

		// Calculate sum(X.^2,2), sum(Y.^2,2)
		for(int i = 0; i < n; i++){
			temp = cblas_dnrm2(d, X+i*d, 1);
			normX[i] = temp * temp;
		}
		for(int i = 0; i < m; i++){
			temp = cblas_dnrm2(d, Y+i*d, 1);
			normY[i] = temp * temp;
		}

		// XY = sum(X.^2,2) -2* X*Y.'
		cblas_dger(CblasRowMajor, n, m, 1, normX, 1, onesN, 1, XY, m);

		// XY = sum(X.^2,2) -2* X*Y.' + sum(Y.^2,2).'
		cblas_dger(CblasRowMajor, n, m, 1, onesM, 1, normY, 1, XY, m);

		// D = sqrt(sum(X.^2,2) -2* X*Y.' + sum(Y.^2,2).');
		for(int i = 0; i < n*m; i++)
			D[i] = sqrt(XY[i]);

		// Free memory
		free(normX);
		free(normY);
		free(onesN);
		free(onesM);
		free(XY);

		return D;
}

// A utility function to swap two elements
void swap_double(double *a, double *b){
    double t = *a;
    *a = *b;
    *b = t;
}

// A utility function to swap two elements
void swap_int(int *a, int *b){
    int t = *a;
    *a = *b;
    *b = t;
}

// QuickSort Partition function. low and high are the range of indexes in arr where partition should work
int partition (double arr[], int *ids, int low, int high){

    // Select a pivot and initialize flag to position of smallest element before pivot
    double pivot = arr[high];
    int i = (low - 1);

    // Go through the array examining each element
    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot, increment i and swap it out with the one currently pointed by i
        if (arr[j] < pivot)
        {
            i++;
            // Swap distances and corresponding point ids
            swap_double(&arr[i], &arr[j]);
            swap_int(&ids[i], &ids[j]);
        }
    }

    // Finally place pivot in its correct position in the array and return the position as the middle point
    swap_double(&arr[i + 1], &arr[high]);
    swap_int(&ids[i + 1], &ids[high]);
    return (i + 1);
}

// Returns the median using the QuickSelect algorithm
double kselect(double arr[], int *ids, int length, int k){
	return quickselect(arr, ids, length, k);
}

// Returns the idx-th element of arr when arr is sorted. idx is the index (starting from 1) of the point we want to find when the array is sorted.
double quickselect(double arr[], int *ids, int length, int idx){

    // Check to end recursion
    if (length == 1){
        return arr[0];
    }

    // Select last array element as pivot
    double pivot = arr[length - 1];
    // Get index of pivot after we partition the array
    int pivotIndex = partition(arr, ids, 0, length - 1);

    // Create the higher and lower arrays that occur after partitioning in QuickSort fashion
    int lowerLength = pivotIndex;
    pivotIndex++;
    int higherLength = (length - (lowerLength + 1));
    // At this point pivotIndex, lowerLength and higherLength all start from 1 not 0

    // Variable to store result of following recursive calls
    double result = 0;

    // This means that the point we're looking (median in our case) is in the lower partition
    if (idx <= lowerLength){
        result = quickselect(arr, ids, lowerLength, idx);
    }
    // This means that idx-th element is our pivot point
    else if(idx == pivotIndex){
        result = pivot;
    }
    // This means that the median is in the higher partition
    else{
        result = quickselect(arr + pivotIndex, ids + pivotIndex, higherLength, idx - pivotIndex);
    }

    // Return result
    return result;
}


// Main function: Used for tests and to call the knn application entry point
int main()
{

    double A[6] = {1.0, 2.0, 11.0, 13.0, 24.0, 22.0};
    double B[6] = {2.0, 4.0, 11.0, 15.0, 23.0, 26.0};

	knnresult results = kNN(A, B, 3, 3, 2, 1);

	for(int i = 0; i < 3; i++){
		printf("Point %d was paired with %d with a distance of %f \n", i, results.nidx[i], results.ndist[i]);
	}

	//printf("Done");
	return 0;

}