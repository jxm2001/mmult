#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
typedef union
{
  __m128 v;
  float d[4];
} v2df_t;

const int block = 128;
const int stride = 4;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]
#define min(i, j) ((i) < (j) ? (i): (j))

void AddDot( int K, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
	v2df_t reg_a[stride],reg_b,reg_c[stride];
	for(int i=0; i<stride; i++)
		reg_c[i].v = _mm_setzero_ps();	
	for(int k=0; k<K; k++){
		for(int i=0; i<stride; i++){
			reg_a[i].v = _mm_set_ps1(a[i]);
		}
		reg_b.v = _mm_load_ps(b);
		for(int i=0; i<stride; i++){
			reg_c[i].v += reg_a[i].v * reg_b.v;
		}
		a+=stride;
		b+=stride;
	}
	for(int i=0; i<stride; i++)
		for(int j=0; j<stride; j++)
		   C(i,j)+=reg_c[i].d[j];	
}

void PackMatrixA( int K, float *a, int lda, float * a_to)
{
	for(int k=0;k<K;k++){
		for(int i=0;i<stride;i++)
			a_to[i] = A(i,k);
		a_to+=stride;
	}
}

void PackMatrixB( int K, float *b, int ldb, float *b_to)
{
	for(int k=0;k<K;k++){
		for(int j=0;j<stride;j++)
			b_to[j] = B(k,j);
		b_to+=stride;
	}
}

void InnerKernel( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	static float buf_a[block*block],buf_b[block*block];	
	for(int i=0;i<M;i+=stride)
		PackMatrixA(K, &A(i,0), lda, buf_a + i*K);
	for(int j=0;j<N;j+=stride)
		PackMatrixB(K, &B(0,j), ldb, buf_b + j*K);

	for(int j=0;j<N;j+=stride)
		for(int i=0;i<M;i+=stride)
			AddDot(K, buf_a + i*K, lda, buf_b + j*K, ldb, &C(i,j), ldc);
}

void MY_MMult( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	for(int j=0;j<N;j+=block)
		for(int i=0;i<M;i+=block)
			for(int k=0;k<K;k+=block)
				InnerKernel(min(block, M-i), min(block,N-j), min(block,K-k), &A(i,k), lda, &B(k,j), ldb, &C(i,j), ldc);
}


