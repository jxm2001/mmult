#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h> //avx2

const int nc = 384;
const int kc = 384;
const int mr = 4;
const int nr = 32;
const int regLen = 512/8/sizeof(float);

typedef union
{
  __m512 v;
  float d[regLen];
} v2df_t;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]
#define min(i, j) ((i) < (j) ? (i): (j))

void AddDot( int K, float *a, float *b, float *c, int ldc )
{
	v2df_t reg_a[mr],reg_b[nr/regLen],reg_c[mr][nr/regLen];
	for(int i=0; i<mr; i++)
		for(int j=0; j<nr/regLen; j++)
			for(int j2=0; j2<regLen; j2++)
				reg_c[i][j].d[j2]=C(i,j*regLen+j2);
	for(int k=0; k<K; k++){
		for(int j=0; j<nr/regLen; j++)
			reg_b[j].v = _mm512_load_ps(b+j*regLen);
		for(int i=0; i<mr; i++){
			reg_a[i].v = _mm512_set1_ps(a[i]);
			for(int j=0; j<nr/regLen; j++)
				reg_c[i][j].v += reg_a[i].v * reg_b[j].v;
		}
		a+=mr;
		b+=nr;
	}
	for(int i=0; i<mr; i++)
		for(int j=0; j<nr/regLen; j++)
			for(int j2=0; j2<regLen; j2++)
				C(i,j*regLen+j2)=reg_c[i][j].d[j2];
}

void PackMatrixA( int K, float *a, int lda, float * a_to)
{
	float buf_a[mr*mr];
	for(int k=0;k<K;k+=mr){
		for(int i=0;i<mr;i++)
			for(int k2=0;k2<mr;k2++)
				buf_a[i*mr+k2] = A(i,k+k2);
		for(int k2=0;k2<mr;k2++)
			for(int i=0;i<mr;i++)
				a_to[k2*mr+i] = buf_a[i*mr+k2];
		a_to+=mr*mr;
	}
}

void PackMatrixB( int K, float *b, int ldb, float *b_to)
{
	for(int k=0;k<K;k++){
		for(int j=0;j<nr;j++)
			b_to[j] = B(k,j);
		b_to+=nr;
	}
}

void InnerKernel( int M, int N, int K, float *a, float *b, float *c, int ldc )
{
	for(int j=0;j<N;j+=nr){
		int prefetch_idx=j+nr*2;
		if(prefetch_idx<N){
			for(int i=0; i<nr; i++)
				_mm_prefetch(&C(i,prefetch_idx), _MM_HINT_T0);
		}
		AddDot(K, a, b+j*K, &C(0,j), ldc);
	}
}
void gepb( int M, int N, int K, float *a, float *b, int ldb, float *c, int ldc, float *pack_b){
	for(int j=0;j<N;j+=nr)
		PackMatrixB(K, &B(0,j), ldb, pack_b+j*K);
	for(int i=0;i<M;i+=mr)
		InnerKernel(mr, N, K, a+i*K, pack_b, &C(i,0), ldc);
}

void gepp(int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc, float *pack_a, float *pack_b){
	for(int i=0;i<M;i+=mr)
		PackMatrixA(K, &A(i,0), lda, pack_a+i*K);
	for(int j=0;j<N;j+=nc)
		gepb(M, min(nc,N-j), K, pack_a, &B(0,j), ldb, &C(0,j), ldc, pack_b);
}

void MY_MMult(int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc)
{
	float* pack_a = (float*)aligned_alloc(regLen*sizeof(float), M*kc*sizeof(float));
	float* pack_b = (float*)aligned_alloc(regLen*sizeof(float), kc*nc*sizeof(float));
	for(int k=0;k<K;k+=kc)
		gepp(M, N, min(kc,K-k), &A(0,k), lda, &B(k,0), ldb, &C(0,0), ldc, pack_a, pack_b);
	free(pack_a);
	free(pack_b);
}


