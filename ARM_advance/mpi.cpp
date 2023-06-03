#include <iostream>
#include <mpi.h>
#include<omp.h>
#include<arm_neon.h>
#include<sys/time.h>
using namespace std;

float matrix[10000][10000];
float matrix1[10000][10000];
float matrix2[10000][10000];
float matrix3[10000][10000];
float matrix4[10000][10000];
int Num_thread = 4;//每个进程4线程编程
void Initialize(int N)
{
    for (int i = 0; i < N; i++)
    {
        //首先将全部元素置为0，对角线元素置为1
        for (int j = 0; j < N; j++)
        {
            matrix[i][j] = 0;
            matrix1[i][j] = 0;
            matrix2[i][j] = 0;
            matrix3[i][j] = 0;
            matrix4[i][j] = 0;
        }
        matrix[i][i] = matrix1[i][i] = matrix2[i][i] = matrix3[i][i]=matrix4[i][i] = 1.0;
        //将上三角的位置初始化为随机数
        for (int j = i + 1; j < N; j++)
        {
            matrix[i][j] = matrix1[i][j] = matrix2[i][j] = matrix3[i][j]= matrix4[i][j] = rand();
        }
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix[i][j] += matrix[k][j];
                matrix1[i][j] += matrix1[k][j];
                matrix2[i][j] += matrix2[k][j];
                matrix3[i][j] += matrix3[k][j];
                matrix4[i][j] += matrix4[k][j];

            }
        }
    }
}
//平凡算法
void normal(float matrix[][10000], int N) {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++)
            matrix[k][j] /= matrix[k][k];
        matrix[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            matrix[i][k] = 0;
        }
    }
}
//一维循环划分
void MPI_cycle(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;

    // 做消元运算
    for (int k = 0; k < N; k++)
    {
        // 查看是否由本进程负责除法运算
        if (k % size == rank)
        {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] /= matrix[k][k];
            matrix[k][k] = 1;
            for (int p = 0; p < size; p++) {
                if (p != rank)
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);//广播给其他进程
            }
        }
        // 其余进程接收
        else {
            MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作,在某行的范围内才负责
        for (int i = k + k % size; i < N; i += size)
        {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //消去完成，进行回代操作,结果回代给0号进程

    if (rank != 0)
        for (int i = rank; i < N; i += size)
            MPI_Send(&matrix[i][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    else//接收数据
        for (int i = 1; i < size; i++)
            for (int j = i; j < N; j += size)
                MPI_Recv(&matrix[j][0], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_cycle time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

//流水线划分
void MPI_pipeLine(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    int part = N / size;//每个进程分配的行数
    int start = rank * part;
    int end = start + part - 1;
    if (rank == size - 1) {
        end = N - 1;
        part = N - start;//最后一个进程分配的行数
    }
    // 做消元运算
    for (int k = 0; k < N; k++)
    {
        // 查看是否由本进程负责除法运算
        if (k >= start && k <= end)
        {
            for (int j = k + 1; j < N; j++)
                matrix[k][j] /= matrix[k][k];
            if (rank < size - 1)//如果不是最后一个进程，就广播给下一个进程
                MPI_Send(&matrix[k][0], N, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        }
        // 之后的进程接收来自前一个进程的消息,并将除法结果发送给下一个进程
        else {
            if (k < end)
            {
                MPI_Recv(&matrix[k][0], N, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (rank < size - 1)
                    MPI_Send(&matrix[k][0], N, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            }
        }
        // 进行消元操作,在某行的范围内才负责
        for (int i = max(k + 1, start); i <= end; i++)
        {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //消去完成，进行回代操作,结果回代给0号进程
    if (rank != 0)
        MPI_Send(&matrix[start][0], part * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    else//从其他进程接收接收数据
        for (int i = 1; i < size; i++)
            MPI_Recv(&matrix[end + 1 + (i - 1) * part][0], part * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_pipeline time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

//循环划分和openmp结合
void MPI_omp(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    int part = N / size;//每个进程分配的行数
    int start = rank * part;
    int end = start + part - 1;
    if (rank == size - 1) {
        end = N - 1;
        part = N - start;//最后一个进程分配的行数
    }
    // 做消元运算
#pragma omp parallel num_threads(Num_thread) shared(matrix,rank,size,N)
    for (int k = 0; k < N; k++)
    {
        // 查看是否由本进程负责除法运算
#pragma omp single
        {
            if (k >= start && k <= end)
            {
                for (int j = k + 1; j < N; j++)
                    matrix[k][j] /= matrix[k][k];
                matrix[k][k] = 1;
                for (int p = 0; p < size; p++) {
                    if (p != rank)
                        MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);//广播给其他进程
                }
            }
            // 其余进程接收
            else {
                MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // 进行消元操作,在某行的范围内才负责
#pragma omp for schedule(guided)
        for (int i = max(k + 1, start); i <= end; i++)
        {
            for (int j = k + 1; j < N; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j];
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //消去完成，进行回代操作,结果回代给0号进程
    if (rank != 0)
        MPI_Send(&matrix[start][0], part * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    else//从其他进程接收接收数据
        for (int i = 1; i < size; i++)
            MPI_Recv(&matrix[end + 1 + (i - 1) * part][0], part * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_omp time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

void MPI_omp_SIMD(float matrix[][10000], int N) {
    int rank;
    int size;
    double begin_time, end_time;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0)
        begin_time = MPI_Wtime();
    else
        begin_time = 0;
    int part = N / size;//每个进程分配的行数
    int start = rank * part;
    int end = start + part - 1;
    if (rank == size - 1) {
        end = N - 1;
        part = N - start;//最后一个进程分配的行数
    }
    // 做消元运算
#pragma omp parallel num_threads(Num_thread) shared(matrix,rank,size,N)
    
    for (int k = 0; k < N; k++)
    {
        // 查看是否由本进程负责除法运算
#pragma omp single
        {
            if (k >= start && k <= end)
            {
                float32x4_t r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
                r0 = vmovq_n_f32(matrix[k][k]);//初始化4个包含gdatakk的向量
                int j;
                for (j = k + 1; j + 4 <= N; j += 4)
                {
                    r1 = vld1q_f32(matrix[k] + j);
                    r1 = vdivq_f32(r1, r0);//将两个向量对位相除
                    vst1q_f32(matrix[k], r1);//相除结果重新放回内存
                }
                for (j; j < N; j++)
                {
                    matrix[k][j] = matrix[k][j] / matrix[k][k];
                }
                matrix[k][k] = 1;
                for (int p = 0; p < size; p++) {
                    if (p != rank)
                        MPI_Send(&matrix[k][0], N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);//广播给其他进程
                }
            }
            // 其余进程接收
            else {
                MPI_Recv(&matrix[k][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // 进行消元操作,在某行的范围内才负责
#pragma omp for schedule(guided)
        for ( int i = max(k + 1, start); i <= end; i++)
        {
            float32x4_t r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
            r0 = vmovq_n_f32(matrix[i][k]);
            int j;
            for ( j = k + 1; j+4 <= N; j+=4) {
                r1 = vld1q_f32(matrix[k] + j);
                r2 = vld1q_f32(matrix[i] + j);
                r3 = vmulq_f32(r0, r1);
                r2 = vsubq_f32(r2, r3);
                vst1q_f32(matrix[i] + j, r2);
            }
            for (j; j < N; j++)
            {
                matrix[i][j] = matrix[i][j] - (matrix[i][k] * matrix[k][j]);
            }
            matrix[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //消去完成，进行回代操作,结果回代给0号进程
    if (rank != 0)
        MPI_Send(&matrix[start][0], part * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    else//从其他进程接收接收数据
        for (int i = 1; i < size; i++)
            MPI_Recv(&matrix[end + 1 + (i - 1) * part][0], part * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    end_time = MPI_Wtime();
    if (rank == 0)
        cout << "MPI_omp_SIMD time: " << (end_time - begin_time) * 1000 << " ms" << endl;

}

int main() {

    int N = 500;
    int rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        cout << "size=4，" << "matrix= " << N <<" thread="<<Num_thread << endl;
    struct timeval begin, end;
    long long res;
    gettimeofday(&begin, NULL);
    Initialize(N);
    gettimeofday(&end, NULL);
    res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "initalize time:" << res << " us" << endl;


    gettimeofday(&begin, NULL);
    normal(matrix, N);
    gettimeofday(&end, NULL);
    res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        cout << "normal time:" << res << " us" << endl;



    MPI_cycle(matrix1, N);
    MPI_pipeLine(matrix2, N);
    MPI_omp(matrix3, N);
    MPI_omp_SIMD(matrix4, N);
    MPI_Finalize();

}