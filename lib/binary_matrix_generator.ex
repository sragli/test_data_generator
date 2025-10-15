defmodule BinaryMatrixGenerator do
  @moduledoc """
  Generates binary matrices with different-sized features emerging from random patterns using Nx.
  """

  @doc """
  Creates a binary matrix from a random pattern with specified dimensions.
  """
  def create_random_binary_matrix(height, width, threshold \\ 0.5) do
    # Generate random tensor with values between 0 and 1
    key = Nx.Random.key(System.system_time())
    {random_tensor, _new_key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {height, width})

    # Convert to binary by applying threshold
    Nx.greater(random_tensor, threshold)
    |> Nx.as_type({:u, 8})
  end

  @doc """
  Creates a binary matrix with clustered features using convolution-like operations.
  """
  def create_clustered_features(height, width, cluster_size \\ 3, threshold \\ 0.6) do
    # Start with random noise
    key = Nx.Random.key(System.system_time())
    {noise, _new_key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {height, width})

    # Create a simple averaging kernel
    kernel_size = min(cluster_size, min(height, width))
    kernel = Nx.broadcast(1.0 / (kernel_size * kernel_size), {kernel_size, kernel_size})

    # Apply smoothing by averaging neighborhoods
    smoothed = apply_kernel(noise, kernel)

    # Convert to binary
    Nx.greater(smoothed, threshold)
    |> Nx.as_type({:u, 8})
  end

  @doc """
  Creates a binary matrix with multiple feature sizes by combining different scales.
  """
  def create_multiscale_features(height, width, scales \\ [3, 7, 15], weights \\ [0.5, 0.3, 0.2]) do
    # Ensure weights sum to 1
    weight_sum = Enum.sum(weights)
    normalized_weights = Enum.map(weights, &(&1 / weight_sum))

    # Generate features at different scales
    features =
      Enum.zip(scales, normalized_weights)
      |> Enum.map(fn {scale, weight} ->
        feature_matrix = create_scale_features(height, width, scale)
        Nx.multiply(feature_matrix, weight)
      end)

    # Combine all scales and convert to binary
    Enum.reduce(features, fn matrix, acc -> Nx.add(acc, matrix) end)
    |> Nx.greater(0.4)
    |> Nx.as_type({:u, 8})
  end

  @doc """
  Creates a binary matrix with cellular automata-like evolution from random seed.
  """
  def create_evolved_features(height, width, iterations \\ 3, survival_threshold \\ 4) do
    # Start with random binary pattern
    initial = create_random_binary_matrix(height, width, 0.4)

    # Evolve through iterations
    Enum.reduce(1..iterations, initial, fn _i, current ->
      evolve_step(current, survival_threshold)
    end)
  end

  @doc """
  Creates a binary matrix with island-like features of varying sizes.

  Starting with an empty matrix, placing random seeds and growing islands from these seeds.
  """
  def create_island_features(height, width, num_seeds \\ 10, growth_probability \\ 0.7) do
    Nx.broadcast(0, {height, width})
    |> Nx.as_type({:u, 8})
    |> place_random_seeds(num_seeds)
    |> grow_islands(growth_probability, 5)
  end

  defp apply_kernel(tensor, kernel) do
    {h, w} = Nx.shape(tensor)
    {kh, kw} = Nx.shape(kernel)

    # Simple convolution implementation
    pad_h = div(kh, 2)
    pad_w = div(kw, 2)

    result = Nx.broadcast(0.0, {h, w})

    # Apply kernel (simplified - would be more efficient with proper convolution)
    indices =
      for i <- pad_h..(h - pad_h - 1),
          j <- pad_w..(w - pad_w - 1),
          do: {i, j}

    Enum.reduce(indices, result, fn {i, j}, acc ->
      # Extract neighborhood
      neighborhood = Nx.slice(tensor, [i - pad_h, j - pad_w], [kh, kw])

      # Compute weighted sum
      weighted_sum = Nx.multiply(neighborhood, kernel) |> Nx.sum()

      # Update result
      Nx.put_slice(acc, [i, j], Nx.reshape(weighted_sum, {1, 1}))
    end)
  end

  defp create_scale_features(height, width, scale) do
    # Generate noise and smooth it
    key = Nx.Random.key(System.system_time())
    {noise, _new_key} = Nx.Random.uniform(key, 0.0, 1.0, shape: {height, width})

    kernel_size = min(scale, min(height, width))

    if kernel_size >= 3 do
      kernel = Nx.broadcast(1.0 / (kernel_size * kernel_size), {kernel_size, kernel_size})
      apply_kernel(noise, kernel)
    else
      noise
    end
  end

  defp evolve_step(binary_matrix, survival_threshold) do
    {h, w} = Nx.shape(binary_matrix)

    # Count neighbors for each cell (simplified implementation)
    neighbor_counts = Nx.broadcast(0, {h, w}) |> Nx.as_type({:u, 8})

    # For each cell, count living neighbors
    indices = for i <- 1..(h - 2), j <- 1..(w - 2), do: {i, j}

    neighbor_counts =
      Enum.reduce(indices, neighbor_counts, fn {i, j}, acc ->
        # Get 3x3 neighborhood
        neighborhood = Nx.slice(binary_matrix, [i - 1, j - 1], [3, 3])

        # Count neighbors (exclude center cell)
        center_value = binary_matrix[i][j] |> Nx.to_number()
        neighbor_sum = Nx.sum(neighborhood) |> Nx.to_number()
        neighbor_count = neighbor_sum - center_value

        # Update count
        Nx.put_slice(acc, [i, j], Nx.reshape(Nx.tensor(neighbor_count), {1, 1}))
      end)

    # Apply evolution rules
    current_cells = Nx.greater(binary_matrix, 0)
    enough_neighbors = Nx.greater_equal(neighbor_counts, survival_threshold)

    # Cell survives if it has enough neighbors
    Nx.logical_and(current_cells, enough_neighbors)
    |> Nx.as_type({:u, 8})
  end

  # Place seeds at random positions
  defp place_random_seeds(matrix, num_seeds) do
    {h, w} = Nx.shape(matrix)

    1..num_seeds
    |> Enum.map(fn _ ->
      i = :rand.uniform(h) - 1
      j = :rand.uniform(w) - 1
      {i, j}
    end)
    |> Enum.reduce(matrix, fn {i, j}, acc ->
        Nx.put_slice(acc, [i, j], Nx.reshape(Nx.tensor(1), {1, 1}))
      end)
  end

  defp grow_islands(seed_matrix, growth_prob, max_iterations) do
    Enum.reduce(1..max_iterations, seed_matrix, fn _iter, current ->
      grow_one_step(current, growth_prob)
    end)
  end

  defp grow_one_step(current_matrix, growth_prob) do
    {h, w} = Nx.shape(current_matrix)
    new_matrix = current_matrix

    # For each empty cell adjacent to a filled cell, maybe grow
    indices = for i <- 1..(h - 2), j <- 1..(w - 2), do: {i, j}

    Enum.reduce(indices, new_matrix, fn {i, j}, acc ->
      current_value = current_matrix[i][j] |> Nx.to_number()

      if current_value == 0 do
        # Check if adjacent to a filled cell
        neighbors = [
          current_matrix[i - 1][j] |> Nx.to_number(),
          current_matrix[i + 1][j] |> Nx.to_number(),
          current_matrix[i][j - 1] |> Nx.to_number(),
          current_matrix[i][j + 1] |> Nx.to_number()
        ]

        has_neighbor = Enum.any?(neighbors, fn n -> n > 0 end)

        if has_neighbor && :rand.uniform() < growth_prob do
          Nx.put_slice(acc, [i, j], Nx.reshape(Nx.tensor(1), {1, 1}))
        else
          acc
        end
      else
        acc
      end
    end)
  end
end
