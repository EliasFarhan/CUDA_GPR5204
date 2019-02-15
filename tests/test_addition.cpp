
#include <vector>
#include <algorithm>
#include "benchmark/benchmark.h"
#include <addition.h>

void Prefill(float* values, const size_t size)
{
	for(int i = 0; i<size;i++)
	{
		values[i] = (float)rand();
	}
}

void CpuAdd(const float* v1, const float* v2, float* result, const size_t length)
{
	for(size_t i = 0; i<length;i++)
	{
		result[i] = v1[i] + v2[i];
	}
}

const int fromRange = 16;
const int toRange = 1 << 23;

static void BM_CpuDot(benchmark::State& state)
{
	std::vector<float> v1(state.range(0));
	std::vector<float> v2(state.range(0));
	std::vector<float> result(state.range(0));

	Prefill(v1.data(), v1.size());
	Prefill(v2.data(), v2.size());
	for (auto _ : state)
	{
		CpuAdd(v1.data(), v2.data(), result.data(), result.size());
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_CpuDot)->Range(fromRange, toRange);

static void BM_GpuDot(benchmark::State& state)
{
	std::vector<float> v1(state.range(0));
	std::vector<float> v2(state.range(0));
	std::vector<float> result(state.range(0));

	Prefill(v1.data(), v1.size());
	Prefill(v2.data(), v2.size());
	for (auto _ : state)
	{
		GpuAdd(v1.data(), v2.data(), result.data(), result.size());
		benchmark::DoNotOptimize(result);
	}
}
BENCHMARK(BM_GpuDot)->Range(fromRange, toRange);

BENCHMARK_MAIN();

