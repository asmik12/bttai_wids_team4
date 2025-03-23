import json

# Function to load benchmark from a JSON file
def load_benchmark(benchmark_file="benchmark.json"):
    """
    Loads the benchmark accuracy from a JSON file.

    Args:
        benchmark_file (str): Path to the JSON file.

    Returns:
        float: Benchmark accuracy value.
    """
    try:
        with open(benchmark_file, "r") as file:
            data = json.load(file)
            return data.get("benchmark", 0.85)  # Default to 0.85 if key is missing
    except (FileNotFoundError, json.JSONDecodeError):
        print("Benchmark file not found or invalid. Using default benchmark = 0.85.")
        return 0.85

# Function to save new benchmark
def save_benchmark(new_benchmark, benchmark_file="benchmark.json"):
    """
    Saves the updated benchmark accuracy to a JSON file.

    Args:
        new_benchmark (float): The new benchmark accuracy.
        benchmark_file (str): Path to the JSON file.
    """
    with open(benchmark_file, "w") as file:
        json.dump({"benchmark": new_benchmark}, file)
    print(f"Updated benchmark saved: {new_benchmark}")