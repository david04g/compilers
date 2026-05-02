#include <exception>
#include <cstdlib>
#include <iostream>
#include <string>

int runParserTests();
int runDependencyTests();
int runSchedulerTests();
int runSimulatorTests();
int runRegressionTests();

namespace {

int runSuite(const std::string &name, int (*suite)()) {
  try {
    int count = suite();
    std::cout << "[PASS] " << name << " (" << count << " checks)\n";
    return count;
  } catch (const std::exception &ex) {
    std::cerr << "[FAIL] " << name << ": " << ex.what() << "\n";
    std::exit(1);
  }
}

} // namespace

int main() {
  int total = 0;
  total += runSuite("parser", runParserTests);
  total += runSuite("dependency", runDependencyTests);
  total += runSuite("scheduler", runSchedulerTests);
  total += runSuite("simulator", runSimulatorTests);
  total += runSuite("regression", runRegressionTests);
  std::cout << "[PASS] total checks: " << total << "\n";
  return 0;
}
