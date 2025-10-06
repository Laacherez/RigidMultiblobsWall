# Use Homebrew g++-15 explicitly
CXX = g++-15

# Python configuration
PYTHON_VERSION = 3.13
PYTHON_DIR = /Users/juls/RigidMultiblobsWall-1/.venv/bin/python3
PYTHON_INCLUDE = /opt/homebrew/opt/python@$(PYTHON_VERSION)/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/include/python$(PYTHON_VERSION)
PYTHON_LIB = /opt/homebrew/opt/python@$(PYTHON_VERSION)/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/lib

# Boost configuration
BOOST_INC = /opt/homebrew/opt/boost/include
BOOST_LIB = /opt/homebrew/opt/boost/lib

# Eigen configuration (header-only)
EIGEN_INC = /opt/homebrew/opt/eigen/include/eigen3

# Compilation flags
CXX_FLAGS = -O3 -std=c++17 -fPIC -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -I$(EIGEN_INC) -D__FILENAME__='"$(shell pwd)"'

# Linking flags
LDFLAGS = -shared -dynamiclib -L$(BOOST_LIB) -L$(PYTHON_LIB) \
	-lpython$(PYTHON_VERSION) \
	-lboost_python$(subst .,,$(PYTHON_VERSION))


