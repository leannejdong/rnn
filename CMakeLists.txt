cmake_minimum_required(VERSION 3.17)
project(rnn)

set(CMAKE_CXX_STANDARD 20)

add_executable(rnn TrainingExamples.h Layers.h NeuralNetwork.h Simulation.h main.cpp)