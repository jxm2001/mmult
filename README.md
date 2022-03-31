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

实验结果如下，其中 mmult_1-mmult_8 为参考 [how-to-optimize-gemm](https://github.com/BBuf/how-to-optimize-gemm)  重新编写的代码，MMult_4x4_13 和 MMult_4x4_14 为 [how-to-optimize-gemm](https://github.com/BBuf/how-to-optimize-gemm) 仓库 x86 版本的最后两个文件。

![](https://gitlab.com/jxm2001/mmult/-/raw/master/src/res/MY_MMult_res.png?inline=false)

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

## 代码阅读情况

mmult_1-mmult_7 参考了 [how-to-optimize-gemm](https://github.com/BBuf/how-to-optimize-gemm) x86 版本的所有代码，mmult_8 参考了 [how-to-optimize-gemm](https://github.com/BBuf/how-to-optimize-gemm) armv7 版本的 MMult_4x4_14.h，armv7 版本 MMult_4x4_18.h-MMult_4x4_20.h 仅仅粗略阅读，其中 MMult_4x4_18.h 是 MMult_4x4_14.h 的汇编版本，MMult_4x4_19.h 和 MMult_4x4_20.h 是基于汇编的调优，armv7 版本的几个 conv1x1s1 文件貌似是在优化神经网络的卷积运算，不太清楚和矩阵乘法优化之间的关系，没有仔细阅读。至于 armv8 版本的代码，基本和 armv7 相同。

