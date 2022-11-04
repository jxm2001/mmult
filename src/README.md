## 测试方法

`sh ./test.sh` 进行全体测试，同时生成测试文件 `test_matrix_multiply.cpp`，通过 `python3 plot_gflops.py` 可以对测试结果进行绘制

修改 `test_matrix_multiply.cpp` 的使得 `MY_MMult()` 函数指向 `mmult_*.h`，然后 `./unit_test` 得到单个测试结果
