#ifndef SYMNMF_H
#define SYMNMF_H

/* EPSILON AND MAX ITER */
#define EPSILON 1e-4
#define MAX_ITER 300

/* Memory Handling */
double **create_matrix(int rows, int cols);
void release_matrix(double **matrix, int rows);
void symnmf_release_all_memory(double **WH, int N, double **HT, int k, double **HTH, double **HHTH, double **H_curr);


/* Core Functions */
double **compute_similarity_matrix(double **points, int N, int d);
double **compute_ddg_matrix(double **points, int N, int d);
double **compute_norm_matrix(double **points, int n, int d);
double **symnmf(double **W, double **H_init, int N, int k);

/* Helper Functions */
double calc_distance(double vector_1[], double vector_2[], int d);
double **multiply_matrix(double **A, double **B, int m, int n, int l);
double **transpose_matrix(double **H, int n, int k);
double frobenius_norm_diff(double **A, double **B, int n, int k);
void calculate_new_h(int N, int k, double **WH, double **HHTH, double **H_new, double **H_curr);
double **symnmf_release_failed_alloc_of_new_h(double **WH, int N, double **HT, int k, double **HTH, double **HHTH);
double **create_deep_copy_matrix(double **src, int rows, int cols);
double **compute_inverse_sqrt_degree(double **D, int N);

/* User Input Helper Functions */
double* parse_line(char* line, int d);
double** read_input(const char* file_name, int* out_N, int* out_d);
int calc_dimension(const char* line);

/* User Output Helper Functions */
void print_matrix(double **mat, int N, int d);
void *print_error();
int print_main_error();

#endif