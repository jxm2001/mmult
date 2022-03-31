#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]

void AddDot( int K, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
	for(int k=0; k<K; k++){
		C(0,0) += A(0,k) * B(k,0);
	}
}

void MY_MMult( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	for(int j=0; j<N; j++)
		for(int i=0; i<M; i++)
			AddDot(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
}


