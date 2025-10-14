defmodule TestDataGenerator.MixProject do
  use Mix.Project

  def project do
    [
      app: :test_data_generator,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.7"}
    ]
  end
end
