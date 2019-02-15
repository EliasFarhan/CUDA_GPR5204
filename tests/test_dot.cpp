
#include "benchmark/benchmark.h"
#include <dot.h>
#include <misc.h>
#include <iostream>
#include <chrono>


const int fromRange = 16;
const int toRange = 1 << 15;

static void BM_CpuDot(benchmark::State& state)
{
	std::vector<float> v1(state.range(0));
	std::vector<float> v2(state.range(0));
	Prefill(v1.data(), v1.size());
	Prefill(v2.data(), v2.size());
	
	float error = 0.0f;
	float answer = MultLocal(v1.data(), v2.data(), state.range(0));
	for (auto _ : state)
	{

		const float result = MultOpti(v1.data(), v2.data(), state.range(0));
		error = result - answer;
	}
	std::cout << "Cpu Opti Dot "<< state.range(0) <<" Error: " << error << "\n";
}
BENCHMARK(BM_CpuDot)->Range(fromRange, toRange);

static void BM_GpuDot(benchmark::State& state)
{
	std::vector<float> v1(state.range(0));
	std::vector<float> v2(state.range(0));


	Prefill(v1.data(), v1.size());
	Prefill(v2.data(), v2.size());

	float error = 0.0f;
	float answer = MultLocal(v1.data(), v2.data(), state.range(0));

	for (auto _ : state)
	{

		auto start = std::chrono::high_resolution_clock::now();
		
		const float result = dotGpu(v1.data(), v2.data(), state.range(0));
		error = answer - result;

		auto end = std::chrono::high_resolution_clock::now();
		
		auto elapsed_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(
				end - start);

		state.SetIterationTime(elapsed_seconds.count());
	}
	std::cout << "Gpu Dot "<< state.range(0) <<" Error: "<<error<<"\n";
}
BENCHMARK(BM_GpuDot)->Range(fromRange, toRange);

BENCHMARK_MAIN();