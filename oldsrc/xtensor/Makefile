SHELL = /bin/sh
CC=g++
CPPFLAGS=-I. -Ixtensor/include -Ixtl/include -std=c++17
B_DIR =../build



ilp.so: ilp.cpp 
	mkdir -p $(B_DIR)
	$(CC) -fPIC $(CPPFLAGS) -Wall -c ilp.cpp -o ilp.o 
	$(CC) -o $(B_DIR)/ilp ilp.o
# 	$(CC) -shared $(CPPFLAGS) GraphicsGems.o bezier_fit.o stroke_group.o -o $(B_DIR)/bezier_fit.so 
	rm *.o