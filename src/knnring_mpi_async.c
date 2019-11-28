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

	// Get processes number and process id
	int p, id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id); // Task ID
	MPI_Comm_size(MPI_COMM_WORLD, &p); // # tasks

	// Find IDs of previous and next node in ring
	int prev, nxt;
	if(id == 0)
		prev = p - 1;
	else
		prev = id - 1;
	if(id == p-1)
		nxt = 0;
	else
		nxt = id + 1;

	// MPI variables used for async communication
    MPI_Request reqSend, reqReceive;
    //MPI_Status status;

    // Y matrix for receiving points
	double* Y = calloc(n*d, sizeof(double));

	// Start pre-fetching and sending data asynchronously
    MPI_Isend(X, n*d, MPI_DOUBLE, nxt, 2, MPI_COMM_WORLD, reqSend);
    MPI_Irecv(Y, n*d, MPI_DOUBLE, prev, 2, MPI_COMM_WORLD, reqReceive);

	// This is the id that corresponds to the node that initially had the current block of points Y
	// It's used to reconstruct the ids array we receive so we don't have to also send ids
	int blockID = id;

	// Generate ids matrix
	int* ids = calloc(n, sizeof(int));
	for(int i = 0; i < n; i++){
		// The first node gets points [0, n], second [n+1, 2n] etc. Node 0 gets the last n points
		if(id != 0)
			ids[i] = (id - 1) * n + i;
		else
			ids[i] = (p-1) * n + i;
	}

	// First calculate using the original dataset, then move on to sending/receiving blocks from others
	knnresult results = kNN(X, X, n, n, d, k);


	// IDs in the knnresult structure are relative. They start from 0 and go up to n-1. We need to map them to the actual ids generated above
	for(int i = 0; i < n * k; i++)
		results.nidx[i] = ids[results.nidx[i]];


	// Temporary results struct, used for calculations in incoming block
	knnresult newResults;


	// Temporary Y matrix, used for trading in nodes with even ids so that we don't overwrite Y before sending it while receiving
	double* tempY = calloc(n*d, sizeof(double));

	// Pointers and temporary arrays used for merging results.
	int ptrResults = 0;
	int ptrNewResults = 0;
	double* tempResDis = calloc(n * k, sizeof(double));
	int* tempResIDs = calloc(n * k, sizeof(int));


	// Trade blocks p-1 times in the ring
	for(int i = 0; i < p-1; i++){

		// Wait for sending/receiving to complete before proceeding
		MPI_Wait(reqSend, MPI_STATUS_IGNORE);
		MPI_Wait(reqReceive, MPI_STATUS_IGNORE);

		// Write the received points into Y so we can start pre-fetching the next ones in tempY
		memcpy(Y, tempY, n * d * sizeof(double));

		// Last time we go in data doesn't need to be prefetched
		if(i < p-2){
			// Start pre-fetching
			MPI_Isend(Y, n*d, MPI_DOUBLE, nxt, 2, MPI_COMM_WORLD, reqSend);
			MPI_Irecv(tempY, n*d, MPI_DOUBLE, prev, 2, MPI_COMM_WORLD, reqReceive);
		}

		// Reconstruct ids array of received block of points based on the node and the number of blocks already traded
		blockID--;
		if(blockID < 0)
			blockID = p-1;
		for(int j = 0; j < n; j++){
			if(blockID != 0)
				ids[j] = (blockID - 1) * n + j;
			else
				ids[j] = (p-1) * n + j;
		}

		// Run calculations on received points
		newResults = kNN(Y, X, n, n, d, k);


		// Again map ids as done initially
		for(int l = 0; l < n * k; l++)
			newResults.nidx[l] = ids[newResults.nidx[l]];


		// Copy points and ids array of current results to temporary arrays to safely alter the results struct bellow
		memcpy(tempResDis, results.ndist, n * k * sizeof(double));
		memcpy(tempResIDs, results.nidx, n * k * sizeof(int));

		// Merge results and newResults arrays in a merge-sort fashion
		// Iterate for each query point
		for(int r = 0; r < n; r++){
			ptrResults = 0;
			ptrNewResults = 0;

			// Iterate for each neighbor
			for(int j = 0; j < k; j++){

				// If the point in older results is closer than the new one, add it, else add the other one and increment the appropriate array pointer
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


// Application Entry Point
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
	        for (int i=0; i<n; i++)
	    		for(int j=0; j<m; j++)
				XY[i*m+j] += normX[i] + normY[j];

		// D = sqrt(sum(X.^2,2) -2* X*Y.' + sum(Y.^2,2).');
		for(int i = 0; i < n*m; i++)
			D[i] = sqrt(fabs(XY[i]));

		// Free memory
		free(normX);
		free(normY);
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
