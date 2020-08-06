自定义OP编译:
2. sh make.sh编译成correlation_lib.so动态库
3. 添加动态库路径到LD_LIBRARY_PATH：
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3.7 -c 'import paddle; print(paddle.sysconfig.get_lib())'`
```
4. 添加correlation op的python路径:
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```
5. python test_correlation.py运行单测，验证是否加载成功。

PS: 如果paddle whl包是从官网上下载的，需要使用gcc 4.8，即把make.sh中的g++ 改为 g++-4.8
