# ####################################################################
# 
#			   C/C++ Makefile
# 
# Adapted from
#  D.Floros' template and
#  http://www.cs.swarthmore.edu/~newhall/unixhelp/howto_makefiles.html
#  
# ####################################################################
# 
# 'make'        build executable file 'main'
# 'make lib'	build the libraries .a
# 'make clean'  removes all .o and executable files
#

# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = mpicc

# define compile-time flags
CFLAGS = -O3 -Wall

# define any directories containing header files
INCLUDES = -I../inc

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib specify
#   their path using -Lpath, something like:
LDFLAGS =

# define any libraries to link into executable
LIBS = -lm -lopenblas

# define the source file for the library
SRC = knnring

# define directories for source files and libraries
SRCDIR = src
LIBDIR = lib

# define vpath to look for missing source files
VPATH = src

# define the different possible executables
TYPES = sequential mpi_sync mpi_async

# define the executable file  name
MAIN = knnring

# call everytime
.PRECIOUS: %.a

all: $(addprefix $(MAIN)_, $(TYPES))

lib: $(addsuffix .a, $(addprefix $(SRC)_, $(TYPES)))

$(MAIN)_%: $(MAIN).c $(SRC)_%.a
	mkdir -p lib
	$(CC) $(CFLAGS) $(INCLUDES) -o $(SRCDIR)/$@ $(word 1,$^) $(LIBDIR)/$(word 2,$^) $(LDFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .cpp file) and $@: the name of the target of the rule (a .o file)

.o.a:
	mkdir -p lib
	ar rcs $(LIBDIR)/$@ $<

# (see the gnu make manual section about automatic variables)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<

clean:
	$(RM) $(SRCDIR)/*.o *~ $(addprefix $(SRCDIR)/$(MAIN)_, $(TYPES)) $(LIBDIR)/knnring_*.a
