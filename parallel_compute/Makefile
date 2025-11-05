# 使用 NVIDIA 的编译器
NVCC = nvcc

# 目标程序名
TARGET = matrix_add

# 源文件
SRC = matrix_add.cu

# 目标文件
OBJ = $(SRC:.cu=.o)

# 编译选项
# -O2 优化
# -arch 指定 GPU 架构，例如 sm_80 (A100) / sm_89 (H100)
# -std=c++17 使用 C++17
NVCCFLAGS = -O2 -std=c++17

# 默认目标
all: $(TARGET)

# 链接生成可执行文件
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# 编译 .cu 文件为 .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c  -o $@ $<

# 清理中间文件
clean:
	rm -f $(OBJ) $(TARGET)
