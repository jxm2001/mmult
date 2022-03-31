const int stride=4;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]

void AddDot( int K, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
	float reg_a[stride],reg_b[stride],reg_c[stride][stride];
	for(int i=0; i<stride; i++)
		for(int j=0; j<stride; j++)
		   reg_c[i][j]=0;	
	for(int k=0; k<K; k++){
		for(int i=0; i<stride; i++){
			reg_a[i] = A(i,k);
			reg_b[i] = B(k,i);
		}
		for(int i=0; i<stride; i++){
			for(int j=0; j<stride; j++){
				reg_c[i][j] += reg_a[i] * reg_b[j];
			}
		}
	}
	for(int i=0; i<stride; i++)
		for(int j=0; j<stride; j++)
		   C(i,j)=reg_c[i][j];	
}

void MY_MMult( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	for(int j=0;j<N;j+=stride)
		for(int i=0;i<M;i+=stride)
			AddDot(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
}


