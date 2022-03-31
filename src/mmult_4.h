#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
typedef union
{
  __m128 v;
  float d[4];
} v2df_t;

const int stride=4;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]

void AddDot( int K, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
	v2df_t reg_a[stride],reg_b,reg_c[stride];
	for(int i=0; i<stride; i++)
		reg_c[i].v = _mm_setzero_ps();	
	for(int k=0; k<K; k++){
		for(int i=0; i<stride; i++){
			reg_a[i].v = _mm_set_ps1(A(i,k));
		}
		reg_b.v = _mm_load_ps(&B(k,0));
		for(int i=0; i<stride; i++){
			reg_c[i].v += reg_a[i].v * reg_b.v;
		}
	}
	for(int i=0; i<stride; i++)
		for(int j=0; j<stride; j++)
		   C(i,j)=reg_c[i].d[j];	
}

void MY_MMult( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	for(int j=0;j<N;j+=stride)
		for(int i=0;i<M;i+=stride)
			AddDot(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
}


