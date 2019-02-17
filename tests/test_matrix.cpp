/*
MIT License

Copyright (c) 2017 SAE Institute Switzerland AG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <vector>
#include <numeric>
#include <iostream>
#include <functional>
#include "benchmark/benchmark.h"
#include <dot.h>

#include <chrono>

class Matrix
{
public:
	Matrix(int width, int height): width(width), height(height)
	{
		values.resize(static_cast<unsigned long>(height) * static_cast<unsigned long>(width));
	}

	void Prefill()
	{
		for (auto& v : values)
		{
			v = static_cast<float>(rand()%1000)/100.0f;
		}
	}

	Matrix Transpose() const
	{
		Matrix m(height, width);
		for(int i = 0; i < height; i++)
		{
			for(int j = 0; j < width; j++)
			{
				m.values[j*width+i] = values[i*width+j];
			}
		}
		return m;
	}

	Matrix Mult(const Matrix& m2) const
	{
		if(width == m2.height)
		{
			Matrix newMatrix(m2.width, height);
			for (int y = 0; y < newMatrix.height; y++)
			{
				for (int x = 0; x < newMatrix.width; x++)
				{
					float newValue = 0.0f;

					for(int i = 0; i < width;i++)
					{
						newValue += values[y*width+i] * m2.values[i*m2.width+x];
					}
							//

					newMatrix.values[x*newMatrix.width+y] = newValue;
				}
			}
			return newMatrix;
		}
		else
		{
			return Matrix(0, 0);
		}
	}


  Matrix MultOptimize(const Matrix& m2T) const
  {

    const int localWidth = width;
    if (localWidth == m2T.width)
      {
        Matrix newMatrix(m2T.width, height);
        const int newWidth = newMatrix.width;
        const int newHeight = newMatrix.height;
        for (int y = 0; y < newHeight; y++)
          {
            for (int x = 0; x < newWidth; x++)
              {
                const int xOrigin = x*m2T.width;
                const int yOrigin = y*width;
                float newValue = MultOpti (&values[yOrigin], &m2T.values[xOrigin], localWidth);
                newMatrix.values[x*newWidth+y] = newValue;
              }
          }
        return newMatrix;
      }
    else
      {
        return Matrix(0, 0);
      }
  }
  Matrix MultGpu(const Matrix& m2T) const
	{
	  const int localWidth = width;
	  if (localWidth == m2T.width)
	  {
		  Matrix newMatrix(m2T.width, height);
		  const int newWidth = newMatrix.width;
		  const int newHeight = newMatrix.height;
		  matrixMultGpu(values.data(), m2T.values.data(), newMatrix.values.data(), newWidth, newHeight, localWidth);
		  return newMatrix;
	  }
	  else
	  {
		  return Matrix(0, 0);
	  }
	}
	Matrix operator+(const Matrix& m2) const
	{
		if(width == m2.width && height == m2.height)
		{
			Matrix newMatrix(width, height);
			for(int i = 0; i < width*height;i++)
			{
				newMatrix.values[i] = values[i]+m2.values[i];
			}
			return newMatrix;
		}
		return Matrix(0,0);
	}
	Matrix operator-(const Matrix& m2) const
	{
		if(width == m2.width && height == m2.height)
		{
			Matrix newMatrix(width, height);
			for(int i = 0; i < width*height;i++)
			{
				newMatrix.values[i] = values[i]-m2.values[i];
			}
			return newMatrix;
		}
		return Matrix(0,0);
	}

	float ValuesSum()
	{
		float sum = 0.0f;
		for(int i = 0; i < width*height;i++)
		{
			sum+=values[i];
		}
		return sum;
	}


private:
	int width = 0;
	int height = 0;
	std::vector<float> values;
};


static void BM_MatrixMult(benchmark::State& state)
{
	for (auto _ : state)
	{
		state.PauseTiming();
		Matrix m1(state.range(0), state.range(1));
		m1.Prefill();
		Matrix m2(state.range(1), state.range(0));
		m2.Prefill();
		state.ResumeTiming();

		benchmark::DoNotOptimize(m1.Mult(m2));
	}
}
BENCHMARK(BM_MatrixMult)->Args ({16,16})->Args ({64,64})->Args({128,128})->Args({256,256})->Args({512, 512})->Args({1024, 1024});


static void BM_MatrixMultOptimize(benchmark::State& state) {
	float error = 0.0f;
	int iteration = 0;
	for (auto _ : state)
	{
		state.PauseTiming();
		Matrix m1(state.range(0)+1, state.range(1)+1);
		m1.Prefill();
		Matrix m2(state.range(1)+1, state.range(0)+1);
		m2.Prefill();
		state.ResumeTiming();

		const Matrix m2T = m2.Transpose();
		const Matrix matrixResult = m1.MultOptimize(m2T);

		state.PauseTiming();
		error += (matrixResult - (m1.Mult(m2))).ValuesSum();
		iteration++;
		state.ResumeTiming();
	}

	std::cout<<"Cpu Transpose Mult average error: "<<(error/iteration)<<"\n";
}
BENCHMARK(BM_MatrixMultOptimize)->Args ({16, 16})->Args ({64,64})->Args({128,128})->Args({256,256})->Args({512, 512})->Args({1024, 1024});

static void BM_MatrixMultGpu(benchmark::State& state) {

	float error = 0.0f;
	int iteration = 0;
	for (auto _ : state)
	{
		state.PauseTiming();
		Matrix m1(state.range(0) + 1, state.range(1) + 1);
		m1.Prefill();
		Matrix m2(state.range(1) + 1, state.range(0) + 1);
		m2.Prefill();
		state.ResumeTiming();
		auto start = std::chrono::high_resolution_clock::now();
		const Matrix m2T = m2.Transpose();
		const Matrix matrixResult = m1.MultGpu(m2T);
		benchmark::DoNotOptimize(matrixResult);
		auto end = std::chrono::high_resolution_clock::now();

		auto elapsed_seconds =
				std::chrono::duration_cast<std::chrono::duration<double>>(
						end - start);

		state.SetIterationTime(elapsed_seconds.count());

		state.PauseTiming();
		error += (matrixResult - (m1.Mult(m2))).ValuesSum();
		iteration ++;
		state.ResumeTiming();
	}
	std::cout<<"Gpu Mult error average: "<<(error/iteration) <<"\n";
}
BENCHMARK(BM_MatrixMultGpu)->Args ({16,16})->Args({ 64,64 })->Args({ 128,128 })->Args({ 256,256 })->Args({ 512, 512 })->Args({ 1024, 1024 });


BENCHMARK_MAIN();

