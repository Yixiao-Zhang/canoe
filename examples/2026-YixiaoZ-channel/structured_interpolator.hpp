#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>
#include <pnetcdf.h>

template<typename T, std::size_t N>
class CoordinateSystem {
public:
  struct Location {
  public:
    Location(const std::array<T, N>& values_in,
         const CoordinateSystem& coord_in)
      : values_(values_in), coord_(coord_in) {}

    const T& operator[](std::size_t i) const {
      return values_[i];
    }

    const T& operator[](std::string_view axis) const {
      return values_[coord_.axis_index(axis)];
    }

    const std::array<T, N>& values() const {
      return values_;
    }

    const CoordinateSystem& coordinate_system() const {
      return coord_;
    }

  private:
    std::array<T, N> values_;
    const CoordinateSystem& coord_;
  };

  class Field {
  public:
    using Coordinate = CoordinateSystem<T, N>;
    using LocationType = typename Coordinate::Location;

    Field(const Coordinate& coord,
        std::array<std::vector<T>, N> axes,
        std::vector<T> values)
      : coord_(coord),
        axes_(std::move(axes)),
        values_(std::move(values))
    {
      for (std::size_t d = 0; d < N; ++d) {
        if (axes_[d].empty()) {
          throw std::invalid_argument(
            "Each axis must have at least one grid point."
          );
        }

        if (!std::is_sorted(axes_[d].begin(), axes_[d].end())) {
          throw std::invalid_argument(
            "Each axis must be sorted in ascending order."
          );
        }

        shape_[d] = axes_[d].size();
      }

      strides_[N - 1] = 1;
      for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(N) - 2; d >= 0; --d) {
        strides_[d] = strides_[d + 1] * shape_[d + 1];
      }

      std::size_t expected_size = 1;
      for (std::size_t d = 0; d < N; ++d) {
        expected_size *= shape_[d];
      }

      if (values_.size() != expected_size) {
        throw std::invalid_argument(
          "Values size does not match grid shape."
        );
      }
    }

    const Coordinate& coordinate_system() const {
      return coord_;
    }

    const std::array<std::vector<T>, N>& axes() const {
      return axes_;
    }

    const std::vector<T>& values() const {
      return values_;
    }

    const std::array<std::size_t, N>& shape() const {
      return shape_;
    }

    T interpolate(const LocationType& loc) const {
      if (&loc.coordinate_system() != &coord_) {
        throw std::invalid_argument(
          "Location uses a different CoordinateSystem."
        );
      }

      std::array<std::array<std::size_t, 2>, N> idx{};
      std::array<T, N> frac{};

      for (std::size_t d = 0; d < N; ++d) {
        const auto& x = axes_[d];
        const T p = loc[d];

        if (x.size() == 1) {
          idx[d][0] = 0;
          idx[d][1] = 0;
          frac[d] = static_cast<T>(0);
          continue;
        }

        if (p <= x.front()) {
          idx[d][0] = 0;
          idx[d][1] = 1;
          frac[d] = static_cast<T>(0);
          continue;
        }

        if (p >= x.back()) {
          idx[d][0] = x.size() - 2;
          idx[d][1] = x.size() - 1;
          frac[d] = static_cast<T>(1);
          continue;
        }

        auto it = std::upper_bound(x.begin(), x.end(), p);
        const std::size_t i =
          static_cast<std::size_t>(std::distance(x.begin(), it)) - 1;

        idx[d][0] = i;
        idx[d][1] = i + 1;
        frac[d] = (p - x[i]) / (x[i + 1] - x[i]);
      }

      T result = static_cast<T>(0);
      const std::size_t num_corners = std::size_t{1} << N;

      for (std::size_t mask = 0; mask < num_corners; ++mask) {
        T weight = static_cast<T>(1);
        std::size_t flat_index = 0;

        for (std::size_t d = 0; d < N; ++d) {
          const bool upper = ((mask >> d) & 1U) != 0U;
          const std::size_t axis_index = idx[d][upper ? 1 : 0];

          flat_index += axis_index * strides_[d];

          weight *= upper ? frac[d] : (static_cast<T>(1) - frac[d]);
        }

        result += weight * values_[flat_index];
      }

      return result;
    }

  private:
    const Coordinate& coord_;
    std::array<std::vector<T>, N> axes_;
    std::vector<T> values_;
    std::array<std::size_t, N> shape_{};
    std::array<std::size_t, N> strides_{};
  };

  template<typename... Names>
  explicit CoordinateSystem(Names... names)
    : axis_names_{std::string(names)...}
  {
    static_assert(sizeof...(Names) == N,
            "Number of axis names must match dimension.");

    for (std::size_t i = 0; i < N; ++i) {
      if (axis_names_[i].empty()) {
        throw std::invalid_argument("Axis name cannot be empty.");
      }

      for (std::size_t j = 0; j < i; ++j) {
        if (axis_names_[i] == axis_names_[j]) {
          throw std::invalid_argument(
            "Duplicate axis name: " + axis_names_[i]
          );
        }
      }
    }
  }

  const std::array<std::string, N>& axis_names() const {
    return axis_names_;
  }

  std::size_t axis_index(std::string_view axis) const {
    for (std::size_t i = 0; i < N; ++i) {
      if (axis_names_[i] == axis) {
        return i;
      }
    }
    throw std::out_of_range("Unknown axis: " + std::string(axis));
  }

  template<typename... Args>
  Location location(Args... args) const {
    static_assert(sizeof...(Args) == N,
            "Number of coordinates must match dimension.");

    return Location(
      std::array<T, N>{static_cast<T>(args)...},
      *this
    );
  }

  Field field(std::array<std::vector<T>, N> axes,
        std::vector<T> values) const
  {
    return Field(*this, std::move(axes), std::move(values));
  }

  Field load(const std::string& filename,
         const std::string& variable_name) const
  {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
            "This version currently supports only float and double.");

    int ncid = -1;
    pnetcdf_check(
      ncmpi_open(MPI_COMM_WORLD, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid),
      "ncmpi_open failed for file " + filename
    );

    try {
      int varid = -1;
      pnetcdf_check(
        ncmpi_inq_varid(ncid, variable_name.c_str(), &varid),
        "ncmpi_inq_varid failed for variable " + variable_name
      );

      int ndims = 0;
      pnetcdf_check(
        ncmpi_inq_varndims(ncid, varid, &ndims),
        "ncmpi_inq_varndims failed for variable " + variable_name
      );

      if (ndims != static_cast<int>(N)) {
        throw std::runtime_error(
          "Variable dimension mismatch: expected " +
          std::to_string(N) + ", got " + std::to_string(ndims)
        );
      }

      std::array<int, N> source_dimids{};
      pnetcdf_check(
        ncmpi_inq_vardimid(ncid, varid, source_dimids.data()),
        "ncmpi_inq_vardimid failed for variable " + variable_name
      );

      std::array<std::string, N> source_axis_names{};
      std::array<std::size_t, N> source_shape{};

      for (std::size_t s = 0; s < N; ++s) {
        char dim_name[NC_MAX_NAME + 1];
        MPI_Offset dim_len = 0;

        pnetcdf_check(
          ncmpi_inq_dim(ncid, source_dimids[s], dim_name, &dim_len),
          "ncmpi_inq_dim failed"
        );

        source_axis_names[s] = dim_name;
        source_shape[s] = static_cast<std::size_t>(dim_len);
      }

      std::array<std::size_t, N> target_to_source{};
      for (std::size_t d = 0; d < N; ++d) {
        bool found = false;
        for (std::size_t s = 0; s < N; ++s) {
          if (axis_names_[d] == source_axis_names[s]) {
            target_to_source[d] = s;
            found = true;
            break;
          }
        }

        if (!found) {
          throw std::runtime_error(
            "Variable is missing required axis: " + axis_names_[d]
          );
        }
      }

      std::array<std::vector<T>, N> target_axes{};
      std::array<std::size_t, N> target_shape{};

      for (std::size_t d = 0; d < N; ++d) {
        int axis_varid = -1;
        pnetcdf_check(
          ncmpi_inq_varid(ncid, axis_names_[d].c_str(), &axis_varid),
          "ncmpi_inq_varid failed for coordinate variable " + axis_names_[d]
        );

        int axis_ndims = 0;
        pnetcdf_check(
          ncmpi_inq_varndims(ncid, axis_varid, &axis_ndims),
          "ncmpi_inq_varndims failed for coordinate variable " + axis_names_[d]
        );

        if (axis_ndims != 1) {
          throw std::runtime_error(
            "Coordinate variable must be 1D: " + axis_names_[d]
          );
        }

        std::array<int, 1> axis_dimids{};
        pnetcdf_check(
          ncmpi_inq_vardimid(ncid, axis_varid, axis_dimids.data()),
          "ncmpi_inq_vardimid failed for coordinate variable " + axis_names_[d]
        );

        char axis_dim_name[NC_MAX_NAME + 1];
        MPI_Offset axis_len = 0;
        pnetcdf_check(
          ncmpi_inq_dim(ncid, axis_dimids[0], axis_dim_name, &axis_len),
          "ncmpi_inq_dim failed for coordinate variable " + axis_names_[d]
        );

        target_axes[d].resize(static_cast<std::size_t>(axis_len));
        read_full_var_all(ncid, axis_varid, target_axes[d].data());

        target_shape[d] = static_cast<std::size_t>(axis_len);

        if (target_shape[d] != source_shape[target_to_source[d]]) {
          throw std::runtime_error(
            "Coordinate size mismatch for axis: " + axis_names_[d]
          );
        }
      }

      std::size_t total_size = 1;
      for (std::size_t d = 0; d < N; ++d) {
        total_size *= target_shape[d];
      }

      std::vector<T> source_values(total_size);
      read_full_var_all(ncid, varid, source_values.data());

      std::vector<T> target_values(total_size);

      const auto source_strides = compute_strides(source_shape);
      const auto target_strides = compute_strides(target_shape);

      std::array<std::size_t, N> target_multi{};
      std::array<std::size_t, N> source_multi{};

      for (std::size_t target_linear = 0; target_linear < total_size; ++target_linear) {
        unravel_index(target_linear, target_shape, target_strides, target_multi);

        for (std::size_t d = 0; d < N; ++d) {
          source_multi[target_to_source[d]] = target_multi[d];
        }

        const std::size_t source_linear =
          ravel_index(source_multi, source_strides);

        target_values[target_linear] = source_values[source_linear];
      }

      pnetcdf_check(
        ncmpi_close(ncid),
        "ncmpi_close failed for file " + filename
      );
      ncid = -1;

      return Field(*this, std::move(target_axes), std::move(target_values));
    }
    catch (...) {
      if (ncid >= 0) {
        ncmpi_close(ncid);
      }
      throw;
    }
  }

private:
  static void pnetcdf_check(int err, const std::string& message) {
    if (err != NC_NOERR) {
      throw std::runtime_error(
        message + ": " + std::string(ncmpi_strerror(err))
      );
    }
  }

  static void read_full_var_all(int ncid, int varid, float* data) {
    pnetcdf_check(
      ncmpi_get_var_float_all(ncid, varid, data),
      "ncmpi_get_var_float_all failed"
    );
  }

  static void read_full_var_all(int ncid, int varid, double* data) {
    pnetcdf_check(
      ncmpi_get_var_double_all(ncid, varid, data),
      "ncmpi_get_var_double_all failed"
    );
  }

  static std::array<std::size_t, N>
  compute_strides(const std::array<std::size_t, N>& shape)
  {
    std::array<std::size_t, N> strides{};
    strides[N - 1] = 1;

    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(N) - 2; d >= 0; --d) {
      strides[d] = strides[d + 1] * shape[d + 1];
    }

    return strides;
  }

  static void unravel_index(std::size_t linear,
                const std::array<std::size_t, N>& shape,
                const std::array<std::size_t, N>& strides,
                std::array<std::size_t, N>& multi)
  {
    for (std::size_t d = 0; d < N; ++d) {
      multi[d] = (linear / strides[d]) % shape[d];
    }
  }

  static std::size_t ravel_index(const std::array<std::size_t, N>& multi,
                   const std::array<std::size_t, N>& strides)
  {
    std::size_t linear = 0;
    for (std::size_t d = 0; d < N; ++d) {
      linear += multi[d] * strides[d];
    }
    return linear;
  }

  std::array<std::string, N> axis_names_;
};
