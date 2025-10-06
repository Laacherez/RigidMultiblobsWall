# Use with boost 1.63 or newer.
#

# location of the Python header files
PYTHON_VERSION = 3.13
PYTHON_DIR = /opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)
PYTHON_INCLUDE = $(PYTHON_DIR)/include/python$(PYTHON_VERSION)
PYTHON_LIB = /opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/lib/python$(PYTHON_VERSION)/config-$(PYTHON_VERSION)-darwin/

# location of the Boost Python include files and library
BOOST_INC = /opt/homebrew/include
BOOST_LIB = /opt/homebrew/lib

# Name of the Boost Python library
LIBBOOST_PYTHON = boost_python313

# Name of the Boost Numpy library (updated to use boost_numpy313)
LIBBOOST_NUMPY = boost_numpy313

# compile mesh classes
TARGET = visit_writer_interface

# Use g++-15
CXX = g++-15

$(TARGET).so: $(TARGET).o
	$(CXX) -shared -dynamiclib $(TARGET).o -L$(BOOST_LIB) -l$(LIBBOOST_PYTHON) -l$(LIBBOOST_NUMPY) -L$(PYTHON_LIB) -lpython$(PYTHON_VERSION) -o $(TARGET).so

$(TARGET).o: $(TARGET).cc
	$(CXX) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).cc
	$(CXX) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c visit_writer.c

clean: 
	rm -f $(TARGET).o $(TARGET).so
