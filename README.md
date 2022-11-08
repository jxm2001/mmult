# 矩阵乘法优化总结

本机（11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz 8 Cores）浮点峰值如下

```
Thread(s): 1
avx512_vnni int8 perf: 564.9395 gops.
avx512f fp32 perf: 141.5989 gflops.
avx512f fp64 perf: 70.5464 gflops.
fma fp32 perf: 141.8257 gflops.
fma fp64 perf: 71.0795 gflops.
avx fp32 perf: 69.2917 gflops.
avx fp64 perf: 34.6215 gflops.
sse fp32 perf: 35.3735 gflops.
sse fp64 perf: 17.3450 gflops.
```

实验结果如下，其中 mmult_1-mmult_8 为参考 [how-to-optimize-gemm](https://github.com/BBuf/how-to-optimize-gemm)  重新编写的代码，mmult_9 和 mmult_10 为参考 goto paper 编写的代码

<img src="https://gitlab.com/jxm2001/picture/-/raw/main/pictures/2022/11/8_23_33_54_202211082333275.png" alt="MY_MMult_res" style="zoom: 33%;" />

### mmult_1

矩阵乘法基线

```c++
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
```

### mmult_2

不难发现基线中矩阵 B 的访存模式对缓存非常不友好，考虑每次同时计算 $4\times 4$ 的矩阵快，提高缓存中矩阵 B 的数据重用率。

```c++
const int stride=4;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]

void AddDot( int K, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
	for(int k=0; k<K; k++){
		for(int j=0; j<stride; j++){
			for(int i=0; i<stride; i++){
				C(i,j) += A(i,k) * B(k,j);
			}
		}
	}
}

void MY_MMult( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	for(int j=0;j<N;j+=stride)
		for(int i=0;i<M;i+=stride)
			AddDot(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
}
```

### mmult_3

考虑将缓存中经常访问的数据放入寄存器，进一步提高访存效率。

```c++
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
```

### mmult_4

采用 SSE 指令集加速计算。

```c++
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
```

### mmult_5

使用矩阵分块，确保每轮计算时的矩阵块能够装入缓存。

```c++
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
			reg_a[i].v = _mm_set_ps1(A(i,k));
		}
		reg_b.v = _mm_load_ps(&B(k,0));
		for(int i=0; i<stride; i++){
			reg_c[i].v += reg_a[i].v * reg_b.v;
		}
	}
	for(int i=0; i<stride; i++)
		for(int j=0; j<stride; j++)
		   C(i,j)+=reg_c[i].d[j];	
}

void InnerKernel( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	for(int j=0;j<N;j+=stride)
		for(int i=0;i<M;i+=stride)
			AddDot(K, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
}

void MY_MMult( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	for(int j=0;j<N;j+=block)
		for(int i=0;i<M;i+=block)
			for(int k=0;k<K;k+=block)
				InnerKernel(min(block, M-i), min(block,N-j), min(block,K-k), &A(i,k), lda, &B(k,j), ldb, &C(i,j), ldc);
}
```

### mmult_6

对矩阵 A，B进行了打包，使得访存连续，不过实际测试时没有明显提升效果，甚至产生了负优化，可能因为这部分目前不是性能瓶颈。

```c++
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
```

### mmult_7

将 SSE 指令集替换为 AVX 指令集。

```c++
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

const int block = 128;
const int stride = 8;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]
#define min(i, j) ((i) < (j) ? (i): (j))

void AddDot( int K, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
	v2df_t reg_a[stride],reg_b,reg_c[stride];
	for(int i=0; i<stride; i++)
		reg_c[i].v = _mm256_setzero_ps();	
	for(int k=0; k<K; k++){
		reg_b.v = _mm256_set_ps(b[7], b[6], b[5], b[4], b[3], b[2], b[1], b[0]);
		for(int i=0; i<stride; i++){
			reg_a[i].v = _mm256_set1_ps(a[i]);
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
```

### mmult_8

访存优化，主要分为以下几点

1. 申请地址空间时保证 64 字节对齐，提高了 AVX 指令的效率
2. 扩大了矩阵分块的大小，使得数据尽量占满 L2 cache
3. 优化了矩阵 A 打包时的访存方式
4. 尝试了数据预取，不过产生了负优化，于是未采用

```c++
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

const int block = 640;
const int stride = 8;

#define A( i, j ) a[ (i)*lda + (j) ]
#define B( i, j ) b[ (i)*ldb + (j) ]
#define C( i, j ) c[ (i)*ldc + (j) ]
#define min(i, j) ((i) < (j) ? (i): (j))

void AddDot( int K, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
	v2df_t reg_a[stride],reg_b,reg_c[stride];
	for(int i=0; i<stride; i++)
		reg_c[i].v = _mm256_setzero_ps();	
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
		   C(i,j)+=reg_c[i].d[j];	
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

void InnerKernel( int M, int N, int K, float *a, int lda, float *b, int ldb, float *c, int ldc )
{
	static float __attribute__((aligned(64))) buf_a[block*block];
   	static float __attribute__((aligned(64))) buf_b[block*block];	
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
```

### mmult_9

参考 goto paper 实现

```c++
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h> //avx2

const int nc = 384;
const int kc = 384;
const int mr = 4;
const int nr = 16;
const int regLen = 256/8/sizeof(float);

typedef union
{
  __m256 v;
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
			reg_b[j].v = _mm256_load_ps(b+j*regLen);
		for(int i=0; i<mr; i++){
			reg_a[i].v = _mm256_set1_ps(a[i]);
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
```

### mmult_10

参考 goto paper 实现

```c++
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
```
