#ifndef KNNRING_H_
#define KNNRING_H_

// Definition of the kNN result struct
typedef struct knnresult
{
    int * nidx;     //!< Indices (0-based) of nearest neighbors
    double * ndist; //!< Distance of nearest neighbors
    int m;          //!< Number of query points
    int k;          //!< Number of nearest neighbors
} knnresult;

// Function Prototypes

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
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);

//! Compute distributed all-kNN of points in X
/*!
    \param X    Data points                 [n-by-d]
    \param n    Number of data points       [scalar]
    \param d    Number of dimensions        [scalar]
    \param k    Number of neighbors         [scalar]

    \return The kNN result
*/
knnresult distrAllkNN(double * X, int n, int d, int k);

double* calculateD(double * X, double * Y, int n, int m, int d, int k);

void swap_double(double *a, double *b);

void swap_int(int *a, int *b);

int partition (double arr[], int *ids, int low, int high);

double kselect(double arr[], int *ids, int length, int k);

double quickSelect(double arr[], int *ids, int length, int idx);

void quickSort(double arr[],  int *ids, int low, int high);

#endif
