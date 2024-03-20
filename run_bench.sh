cmake -S . -B bin
cmake --build bin --config Release -j12
bench=bin/benchmark/rtml_benchmark # Difference build than from the IDE
plot=benchmark/plot.py
rm benchmark.csv
$bench --benchmark_format=csv > benchmark.csv
python3 $plot -f benchmark.csv