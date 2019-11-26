#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <string.h>
#include "../inc/knnring.h"
#include "mpi.h"


// Application Entry Point
/*
 * X is this node's points from which distances will be calculated
 * n is the number of points in X and each incoming-outcoming block
 * d is the number of dimensions of each point
 * k is the number of minimum points to calculate
 */
knnresult distrAllkNN(double * X, int n, int d, int k){

	// Processes and process id
	int p, id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID
	MPI_Comm_size(MPI_COMM_WORLD, &p); // # tasks

	// IDs of previous and next in line for ring
	int prev, nxt;
	if(id == 0)
		prev = p - 1;
	else
		prev = id - 1;
	if(id == p-1)
		nxt = 0;
	else
		nxt = id + 1;


	// Generate ids matrix
	int* ids = calloc(n, sizeof(int));
	for(int i = 0; i < n; i++){
		// The first node gets points [0, n], second [n+1, 2n] etc. Node 0 gets the last n points
		if(id != 0)
			ids[i] = (id - 1) * n + i;
		else
			ids[i] = (p-1) * n + i;
	}

	printf("Node %d has ids starting from %d and ending at %d\n", id, ids[0], ids[n-1]);

	// First calculate using the original dataset, then move on to sending/receiving blocks from others
	knnresult results = kNN(X, X, n, n, d, k);


	// IDs in the knnresult structure are relative - we need to map them to the actual ids generated above
	for(int i = 0; i < n * k; i++)
		results.nidx[i] = ids[results.nidx[i]];


	// Temporary results struct, used for calculations in incoming block
	knnresult newResults;


	// Y and temporary Y and ids arrays, used for trading. Temporary arrays are only used in nodes with even ids
	double* Y = calloc(n*d, sizeof(double));
	memcpy(Y, X, n * d * sizeof(double));
	double* tempY = calloc(n*d, sizeof(double));
	int* tempids = calloc(n, sizeof(int));


	// Pointers and temporary arrays used for merging results.
	int ptrResults = 0;
	int ptrNewResults = 0;
	double* tempResDis = calloc(n * k, sizeof(double));
	int* tempResIDs = calloc(n * k, sizeof(int));


	printf("p is %d", p);

	// Trade blocks p-1 times in the ring
	for(int i = 0; i < p-1; i++){

		// If process id is even, send data first and then receive. Do the opposite for even
		if(id % 2 == 0){

			// Send data and ids matrix - use tags 2 & 3 respectively
			MPI_Send(Y, n*d, MPI_DOUBLE, nxt, 2, MPI_COMM_WORLD);
			MPI_Send(ids, n, MPI_INT, nxt, 3, MPI_COMM_WORLD);

			// Receive data and ids matrix - use tags 2 & 3 respectively
			MPI_Recv(Y, n*d, MPI_DOUBLE, prev, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(ids, n, MPI_INT, prev, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		}else{

			// Receive data and ids matrix - use tags 2 & 3 respectively
			MPI_Recv(tempY, n*d, MPI_DOUBLE, prev, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(tempids, n, MPI_INT, prev, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Send data and ids matrix - use tags 2 & 3 respectively
			MPI_Send(Y, n*d, MPI_DOUBLE, nxt, 2, MPI_COMM_WORLD);
			MPI_Send(ids, n, MPI_INT, nxt, 3, MPI_COMM_WORLD);

			// Write temporary arrays into the regular ones
			memcpy(Y, tempY, n * d * sizeof(double));
			memcpy(ids, tempids, n * sizeof(int));

		}

		printf("Node %d completed trading\n", id);

		// Run calculations
		newResults = kNN(Y, X, n, n, d, k);
		printf("Node %d completed calculations\n", id);


		// Again map ids
		for(int l = 0; l < n * k; l++)
			newResults.nidx[l] = ids[newResults.nidx[l]];
		printf("Node %d finished mapping ids after calculations\n", id);


		// Copy points and ids array of current results to temporary arrays to safely alter the results struct bellow
		memcpy(tempResDis, results.ndist, n * k * sizeof(double));
		memcpy(tempResIDs, results.nidx, n * k * sizeof(int));
		printf("Node %d finished copying original results to temporary arrays and started merging\n", id);

		// Merge results and newResults arrays
		for(int r = 0; r < n; r++){
			ptrResults = 0;
			ptrNewResults = 0;


			for(int j = 0; j < k; j++){
				// If the point
				if(tempResDis[r*k + ptrResults] < newResults.ndist[r*k + ptrNewResults]){
					results.ndist[r*k + j] = tempResDis[r*k + ptrResults];
					results.nidx[r*k + j] = tempResIDs[r*k + ptrResults];
					ptrResults++;
				}else{
					results.ndist[r*k + j] = newResults.ndist[r*k + ptrNewResults];
					results.nidx[r*k + j] = newResults.nidx[r*k + ptrNewResults];
					ptrNewResults++;
				}
			}

		}

	}

	return results;
}

// Calculate results
knnresult kNN(double * X, double * Y, int n, int m, int d, int k)
{

	// Calculate distances matrix D - D is row-major and nxm
	double* D = calculateD(X, Y, n, m, d, k);


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

		// Quicksort the results
		quickSort(D + j * n, ids, 0, n-1);

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
		cblas_dger(CblasRowMajor, n, m, 1, normX, 1, onesM, 1, XY, m);

		// XY = sum(X.^2,2) -2* X*Y.' + sum(Y.^2,2).'
		cblas_dger(CblasRowMajor, n, m, 1, onesN, 1, normY, 1, XY, m);

		// D = sqrt(sum(X.^2,2) -2* X*Y.' + sum(Y.^2,2).');
		for(int i = 0; i < n*m; i++)
			D[i] = sqrt(XY[i]);

		// Free memory
		//free(normX);
		//free(normY);
		//free(onesN);
		//free(onesM);
		//free(XY);

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
	return quickSelect(arr, ids, length, k);
}

// Returns the idx-th element of arr when arr is sorted. idx is the index (starting from 1) of the point we want to find when the array is sorted.
double quickSelect(double arr[], int *ids, int length, int idx){

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
        result = quickSelect(arr, ids, lowerLength, idx);
    }
    // This means that idx-th element is our pivot point
    else if(idx == pivotIndex){
        result = pivot;
    }
    // This means that the median is in the higher partition
    else{
        result = quickSelect(arr + pivotIndex, ids + pivotIndex, higherLength, idx - pivotIndex);
    }

    // Return result
    return result;
}

// Simple quicksort implementation that also sorts the ids array according to the way the main array was sorted
void quickSort(double arr[], int *ids, int low, int high)
{
    if (low < high)
    {
        // idx is the partition point
        int idx = partition(arr, ids, low, high);

        // Divide and conquer
        quickSort(arr, ids, low, idx - 1);
        quickSort(arr, ids, idx + 1, high);
    }
}
