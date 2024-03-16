#include <benchmark/benchmark.h>

#include "blas.hpp"
#include "isolate.hpp"

auto main(int argc, char** argv) -> int {
    char arg0_default[] = "benchmark";
    char* args_default = arg0_default;
    if (!argv) { argc = 1; argv = &args_default; }
    if (!rtml::isolate::init_rtml_runtime()) return 1;
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    rtml::isolate::shutdown_rtml_runtime();
    return 0;
}
