CXX = g++-15

CXXFLAGS = -I../ -DPYTHON -DNDEBUG -O3 -ffast-math -Wall -shared -std=c++17 -fPIC \
	`python3 -m pybind11 --includes` mobility.cpp -o mobility_cpp`python3-config --extension-suffix` \
	-I/opt/homebrew/include/eigen3 -fopenmp

ifeq ($(shell uname -s),Darwin)
	CXXFLAGS += -mcpu=apple-m3
	CXXFLAGS += -undefined dynamic_lookup
endif

# warnings annoy the fuck outta me !!!!
CXXFLAGS += -Wno-deprecated-declarations

all: mobility.cpp mobility.hpp
	$(CXX) $(CXXFLAGS)

clean:
	rm -f mobility_cpp`python3-config --extension-suffix`
