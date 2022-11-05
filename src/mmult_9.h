#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h> //avx2
typedef union
{
  __m256 v;
  float d[8];
} v2df_t;

const int block = 256;
const int stride = 8;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]
#define min(i, j) ((i) < (j) ? (i): (j))

void AddDot( int K, float *a, float *b, float *c, int ldc )
{
	v2df_t reg_a[stride],reg_b,reg_c[stride];
	for(int i=0; i<stride; i++)
		for(int j=0; j<stride; j++)
			reg_c[i].d[j]=C(i,j);
	for(int k=0; k<K; k++){
		reg_b.v = _mm256_load_ps(b);
		for(int i=0; i<stride; i++){
			reg_a[i].v = _mm256_set1_ps(a[i]);
			reg_c[i].v += reg_a[i].v * reg_b.v;
		}
		a+=stride;
		b+=stride;
	}
	for(int i=0; i<stride; i++)
		for(int j=0; j<stride; j++)
			C(i,j)=reg_c[i].d[j];
}

void PackMatrixA( int K, float *a, int lda, float * a_to)
{
	float buf_a[stride*stride];
	for(int k=0;k<K;k+=stride){
		for(int i=0;i<stride;i++)
			for(int k2=0;k2<stride;k2++)
				buf_a[i*stride+k2] = A(i,k+k2);
		for(int k2=0;k2<stride;k2++)
			for(int i=0;i<stride;i++)
				a_to[k2*stride+i] = buf_a[i*stride+k2];
		a_to+=stride*stride;
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

void InnerKernel( int M, int N, int K, float *a, float *b, float *c, int ldc )
{
	for(int j=0;j<N;j+=stride){
		int prefetch_idx=j+stride*2;
		if(prefetch_idx<N){
			for(int i=0; i<stride; i++)
				_mm_prefetch(&C(i,prefetch_idx), _MM_HINT_T0);
		}
		AddDot(K, a, b+j*K, &C(0,j), ldc);
	}
}
void gepb( int M, int N, int K, float *a, float *b, int ldb, float *c, int ldc, float *pack_b){
	for(int j=0;j<N;j+=stride)
		PackMatrixB(K, &B(0,j), ldb, pack_b+j*K);
	for(int i=0;i<M;i+=stride)
		InnerKernel(stride, N, K, a+i*K, pack_b, &C(i,0), ldc);
}

void gepp(int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc, float *pack_a, float *pack_b){
	for(int i=0;i<M;i+=stride)
		PackMatrixA(K, &A(i,0), lda, pack_a+i*K);
	for(int j=0;j<N;j+=block)
		gepb(M, min(block,N-j), K, pack_a, &B(0,j), ldb, &C(0,j), ldc, pack_b);
}

void MY_MMult(int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
	float* pack_a = (float*)aligned_alloc(256/8, M*block*sizeof(float));
	float* pack_b = (float*)aligned_alloc(256/8, block*block*sizeof(float));
	for(int k=0;k<K;k+=block)
		gepp(M, N, min(block,K-k), &A(0,k), lda, &B(k,0), ldb, &C(0,0), ldc, pack_a, pack_b);
	free(pack_a);
	free(pack_b);
}


