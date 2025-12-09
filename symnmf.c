#define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BETA 0.5
#define EPSILON 1e-4
#define MAX_ITER 300

/* Function Decleration */
double calc_distance(double vector_1[], double vector_2[], int d);
double **create_matrix(int rows, int cols);
double **compute_similarity_matrix(double **points, int N, int d);
double **compute_ddg_matrix(double **points, int N, int d);
double **symnmf(double **W, double **H_init, int N, int k);
void calculate_new_h(int N, int k, double **WH, double **HHTH, double **H_new, double **H_curr);
double **symnmf_release_failed_alloc_of_new_h(double **WH, int N, double **HT, int k, double **HTH, double **HHTH);
void symnmf_release_all_memory(double **WH, int N, double **HT, int k, double **HTH, double **HHTH, double **H_curr);
double **multiply_matrix(double **A, double **B, int m, int n, int l);
double **compute_norm_matrix(double **points, int n, int d);
void release_matrix(double **matrix, int rows);
double **transpose_matrix(double **H, int n, int k);
double frobenius_norm_diff(double **A, double **B, int n, int k);
int calc_dimension(const char *line);
double *parse_line(char *line, int d);
double **read_input(const char *file_name, int *out_N, int *out_d);
double **create_deep_copy_matrix(double **src, int rows, int cols);
double **compute_inverse_sqrt_degree(double **D, int N);
void print_matrix(double **mat, int N, int d);
void *print_error();
int print_main_error();


/* Compute A */
double **compute_similarity_matrix(double **points, int N, int d)
{
    int i, j;
    double euclidean_distance, temp;
    double **A;
    A = create_matrix(N, N); /* Creating mat A */
    if (A == NULL){
        return NULL;
    }
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            if (i != j)
            {
                euclidean_distance = calc_distance(points[i], points[j], d); /* Calculate the squared Euclidean distance */
                temp = pow(euclidean_distance, 2);
                A[i][j] = exp(-0.5 * temp); /* Calculate aij */
            }
            else{
                A[i][j] = 0.0;
            }
        }
    }
    return A;
}

/* Compute D */
double **compute_ddg_matrix(double **points, int N, int d)
{
    int i, j;
    double degree;
    double **A, **D;
    A = compute_similarity_matrix(points, N, d); /* Calculate A */
    if (A == NULL){
        return NULL;
    }
    D = create_matrix(N, N); /* Creating matrix D */
    if (D == NULL){
        release_matrix(A, N);
        return NULL;
    }

    for (i = 0; i < N; i++){
        degree = 0.0;
        /* Compute di */
        for (j = 0; j < N; j++){
            degree += A[i][j];
        }
        D[i][i] = degree;
    }
    release_matrix(A, N);
    return D;
}

/* Compute W */
double **compute_norm_matrix(double **points, int N, int d)
{
    double **A, **D, **D_inv_sqrt, **temp, **W;
    A = compute_similarity_matrix(points, N, d);
    if (A == NULL){
        return NULL;
    }

    D = compute_ddg_matrix(points, N, d);
    if (D == NULL){
        release_matrix(A, N);
        return NULL;
    }

    D_inv_sqrt = compute_inverse_sqrt_degree(D, N);
    if (D_inv_sqrt == NULL){
        release_matrix(A, N);
        release_matrix(D, N);
        return NULL;
    }

    temp = multiply_matrix(D_inv_sqrt, A, N, N, N);
    W = (temp != NULL) ? multiply_matrix(temp, D_inv_sqrt, N, N, N) : NULL;

    release_matrix(A, N);
    release_matrix(D, N);
    release_matrix(D_inv_sqrt, N);
    release_matrix(temp, N);
    return W;
}

/* Update H - Symnmf update loop */
double **symnmf(double **W, double **H_init, int N, int k)
{
    int iter = 0;
    double change;
    double **H_curr, **H_new, **WH, **HT, **HTH, **HHTH;
    change = EPSILON + 1;                           /* Ensure at least one loop */
    H_curr = create_deep_copy_matrix(H_init, N, k); /* Create a deep copy of H_init */
    if (H_curr == NULL) return print_error();
    while (iter < MAX_ITER && pow(change, 2) >= EPSILON) {
        WH = multiply_matrix(W, H_curr, N, N, k); /* Compute W*H */
        if (WH == NULL) return print_error();
        HT = transpose_matrix(H_curr, N, k); /* Compute Ht = H Transpose */
        if (HT == NULL) {
            release_matrix(WH, N);
            return print_error();
        }
        HTH = multiply_matrix(HT, H_curr, k, N, k); /* Compute HTH = H^T * H_curr */
        if (HTH == NULL) {
            release_matrix(WH, N);
            release_matrix(HT, k);
            return print_error();
        }
        HHTH = multiply_matrix(H_curr, HTH, N, k, k); /* Compute HHTH = H_curr * (H^T * H_curr) */
        if (HHTH == NULL) {
            release_matrix(WH, N);
            release_matrix(HT, k);
            release_matrix(HTH, k);
            return print_error();
        }
        H_new = create_matrix(N, k);
        if (H_new == NULL) return symnmf_release_failed_alloc_of_new_h(WH, N, HT, k, HTH, HHTH);
        calculate_new_h(N, k, WH, HHTH, H_new, H_curr);
        change = frobenius_norm_diff(H_curr, H_new, N, k); /* Check convergence */
        symnmf_release_all_memory(WH, N, HT, k, HTH, HHTH, H_curr);
        H_curr = H_new;
        iter++;
    }
    return H_curr;
}


/* Helper functions */

void calculate_new_h(int N, int k, double **WH, double **HHTH, double **H_new, double **H_curr)
{
    int i, j;
    double numerator, denominator, multiplier;
    for (i = 0; i < N; i++)
        for (j = 0; j < k; j++){
            numerator = WH[i][j];
            denominator = HHTH[i][j];
            multiplier = (1 - BETA) + BETA * (numerator / denominator);
            H_new[i][j] = H_curr[i][j] * multiplier;
        }
}

double **symnmf_release_failed_alloc_of_new_h(double **WH, int N, double **HT, int k, double **HTH, double **HHTH)
{
    release_matrix(WH, N);
    release_matrix(HT, k);
    release_matrix(HTH, k);
    release_matrix(HHTH, N);
    return print_error();
}

void symnmf_release_all_memory(double **WH, int N, double **HT, int k, double **HTH, double **HHTH, double **H_curr)
{
    release_matrix(WH, N);
    release_matrix(HT, k);
    release_matrix(HTH, k);
    release_matrix(HHTH, N);
    release_matrix(H_curr, N);
}

void *print_error()
{
    printf("An Error Has Occurred\n");
    return NULL;
}

/* Euclidian distance of vectors */
double calc_distance(double vector_1[], double vector_2[], int d)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < d; i++){
        sum += pow((vector_1[i] - vector_2[i]), 2);
    }
    return sqrt(sum);
}

void release_matrix(double **matrix, int rows)
{
    int i;
    for (i = 0; i < rows; i++){
        free(matrix[i]);
    }
    free(matrix);
}

/* Matrix memory allocation */
double **create_matrix(int rows, int cols)
{
    /* Matrix rows */
    double **matrix = calloc(rows, sizeof(double *));
    int i, n;
    if (matrix == NULL){
        return print_error();
    }
    /* Matrix columns */
    for (n = 0; n < rows; n++){
        matrix[n] = calloc(cols + 1, sizeof(double));
        if (matrix[n] == NULL){
            /* Free memory of matrix */
            for (i = 0; i < n; i++){
                free(matrix[i]);
            }
            free(matrix);
            return print_error();
        }
    }
    return matrix;
}

/* Multiply two matrices */
double **multiply_matrix(double **A, double **B, int m, int n, int l)
{
    double **C;
    int i, j, k;
    C = create_matrix(m, l);
    if (C == NULL){
        return print_error();
    }
    for (i = 0; i < m; i++){
        for (j = 0; j < l; j++){
            C[i][j] = 0.0;
            for (k = 0; k < n; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

/* Transpose H[n][k] into HT[k][n] */
double **transpose_matrix(double **H, int n, int k)
{
    int i, j;
    double **HT;
    HT = create_matrix(k, n);
    if (HT == NULL){
        return print_error();
    }

    for (i = 0; i < k; i++){
        for (j = 0; j < n; j++){
            HT[i][j] = H[j][i];
        }
    }
    return HT;
}

/* Compute the Frobenius norm difference between two n x k matrices */
double frobenius_norm_diff(double **A, double **B, int n, int k)
{
    int i, j;
    double sum = 0.0;
    double diff;
    for (i = 0; i < n; i++){
        for (j = 0; j < k; j++){
            diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

/* Calculate the dimension of a vector */
int calc_dimension(const char *line)
{
    int dim = 1, i;
    for (i = 0; line[i] != '\0'; i++){
        if (line[i] == ','){
            dim++;
        }
    }
    return dim;
}

/* Parses lines from the text file */
double *parse_line(char *line, int d)
{
    int i;
    char *token , *endptr;
    double *vec = malloc(d * sizeof(double));
    if (vec == NULL){
        return NULL;
    }
    token = strtok(line, ",\n");
    for (i = 0; i < d && token != NULL; i++){
        vec[i] = strtod(token, &endptr); /* Parses the string to double */
        if (*endptr != '\0'){
            free(vec);
            return NULL;
        }
        token = strtok(NULL, ",\n");  /* Moves one cell up in token to next coordiante */
    }
    return vec;
}

/* Reading input line by line from file */
double **read_input(const char *file_name, int *out_N, int *out_d)
{
    int cap = 10, row = 0, d = -1, i;
    size_t len = 0;
    char *line = NULL;
    double **mat, **new_mat;
    FILE *f = fopen(file_name, "r");
    if (f == NULL) return NULL;
    if (!(mat = malloc(cap * sizeof(double *)))) {
        fclose(f);
        return NULL;
    }
    while (getline(&line, &len, f) != -1){
        if (d == -1){
            d = calc_dimension(line); /* First valid line: calculate dimension */
        }
        if (row == cap){
            cap *= 2;
            if (!(new_mat = realloc(mat, cap * sizeof(double *)))){
                for (i = 0; i < row; i++){
                    free(mat[i]);
                }
                free(mat); free(line);fclose(f);
                return NULL;
            }
            mat = new_mat;
        }
        mat[row] = parse_line(line, d);
        if (mat[row] == NULL){
            for (i = 0; i < row; i++){
                free(mat[i]);
            }
            free(mat); free(line); fclose(f);
            return NULL;
        }
        row++;
    }
    free(line);
    fclose(f);
    *out_N = row;
    *out_d = d;
    return mat;
}

/* Print matrix according to required format */
void print_matrix(double **mat, int N, int d)
{
    int i, j;
    if (mat == NULL){
        print_error();
        return;
    }
    for (i = 0; i < N; i++){
        for (j = 0; j < d; j++){
            if (j == d - 1){
                printf("%.4f\n", mat[i][j]);
            }
            else{
                printf("%.4f,", mat[i][j]);
            }
        }
    }
}

double **create_deep_copy_matrix(double **src, int rows, int cols)
{
    int i, j;
    double **copy = create_matrix(rows, cols);
    if (copy == NULL){
        return print_error(); /* Memory allocation failed */
    }
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            copy[i][j] = src[i][j]; /* Copy each element */
        }
    }
    return copy;
}

double **compute_inverse_sqrt_degree(double **D, int N)
{
    int i;
    double **D_inv_sqrt = create_matrix(N, N);
    if (D_inv_sqrt == NULL){
        return print_error();
    }
    for (i = 0; i < N; i++){
        D_inv_sqrt[i][i] = (D[i][i] > 0) ? (1.0 / sqrt(D[i][i])) : 0.0;
    }
    return D_inv_sqrt;
}

int print_main_error()
{
    printf("An Error Has Occurred\n");
    return 1;
}

/* Main */
int main(int argc, char *argv[])
{
    int N, d;
    const char *goal, *file_name;
    double **points, **result;
    if (argc != 3){
        return print_main_error();
    }
    goal = argv[1];
    file_name = argv[2];
    if ((goal == NULL) || (file_name == NULL)){
        return print_main_error();
    }
    points = read_input(file_name, &N, &d);
    if (points == NULL || N <= 0 || d <= 0){
        return print_main_error();
    }
    if (strcmp(goal, "sym") == 0){
        result = compute_similarity_matrix(points, N, d);
    }
    else if (strcmp(goal, "ddg") == 0){
        result = compute_ddg_matrix(points, N, d);
    }
    else if (strcmp(goal, "norm") == 0){
        result = compute_norm_matrix(points, N, d);
    }
    else{
        release_matrix(points, N);
        return print_main_error();
    }
    if (result == NULL){
        release_matrix(points, N);
        return print_main_error();
    }
    print_matrix(result, N, N);
    release_matrix(result, N);
    release_matrix(points, N);
    return 0;
}