CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -funroll-loops -ffast-math \
             -Wall -Wextra -Wno-unused-parameter \
             -I include
LDFLAGS  := -pthread

# Detect macOS vs Linux
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    # Clang on macOS — OpenMP via homebrew libomp
    OMP_FLAGS := $(shell pkg-config --cflags --libs libomp 2>/dev/null || echo "")
    CXXFLAGS  += $(OMP_FLAGS)
    LDFLAGS   += $(OMP_FLAGS)
else
    # GCC on Linux
    CXXFLAGS  += -fopenmp
    LDFLAGS   += -fopenmp -lm
endif

TARGET   := montecarlo
SRCS     := src/main.cpp
HEADERS  := include/rng.hpp include/models.hpp include/statistics.hpp include/visualizer.hpp

.PHONY: all clean run perf

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "Built: $(TARGET)"

run: all
	./$(TARGET)

# Open report in browser after run
report: run
	open report.html 2>/dev/null || xdg-open report.html 2>/dev/null || echo "Open report.html manually"

# Performance build with PGO (two-pass)
pgo-build:
	$(CXX) $(CXXFLAGS) -fprofile-generate $(SRCS) -o $(TARGET)_pgo_gen $(LDFLAGS)
	./$(TARGET)_pgo_gen 2>/dev/null || true
	$(CXX) $(CXXFLAGS) -fprofile-use -fprofile-correction $(SRCS) -o $(TARGET)_pgo $(LDFLAGS)
	@echo "PGO build: $(TARGET)_pgo"

clean:
	rm -f $(TARGET) $(TARGET)_pgo $(TARGET)_pgo_gen *.gcda *.gcno report.html

# Show compiler version and flags
info:
	@echo "CXX:      $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@$(CXX) --version | head -1
