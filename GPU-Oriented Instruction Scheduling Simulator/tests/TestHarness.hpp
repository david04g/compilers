#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

inline void requireTrue(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename T, typename U>
void requireEqual(const T &actual, const U &expected, const std::string &message) {
  if (!(actual == expected)) {
    std::ostringstream out;
    out << message << " (actual=" << actual << ", expected=" << expected << ")";
    throw std::runtime_error(out.str());
  }
}

inline void requireNear(double actual, double expected, double tolerance,
                        const std::string &message) {
  if (std::fabs(actual - expected) > tolerance) {
    std::ostringstream out;
    out << message << " (actual=" << actual << ", expected=" << expected << ")";
    throw std::runtime_error(out.str());
  }
}
