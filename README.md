# TestDataGenerator

Generators for various kinds of data.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `test_data_generator` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:test_data_generator, "~> 0.1.0"}
  ]
end
```

## Usage

### Binary matrices

```elixir
BinaryMatrixGenerator.create_random_binary_matrix(70, 70)

BinaryMatrixGenerator.create_clustered_features(70, 70, 4)

BinaryMatrixGenerator.create_multiscale_features(70, 70, [3, 9, 15])
```
