#ifndef GP_STRUCT_H
#define GP_STRUCT_H

#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <random>
#include <fstream>
#include "GPNodeStruct.h"
#include <algorithm>

class GPStruct {
  private:
    int seed;
    int maxDepth;
    int maxGenerations;
    double mutationRate;
    double crossoverRate;
    int populationSize;
    int tournamentSize;

    std::vector<GPNodeStruct*> population;

    std::vector<std::vector<double>> training;
    std::vector<std::vector<double>> testing;
    std::vector<std::vector<double>> validation;
    
    std::vector<std::string> validTerminals = {"double"};
    const std::vector<std::string> validOperators = {"+", "*", "-", "/", "max", "min"};
    const std::vector<std::string> validUnaryOperators = {"sigmoid", "sin", "cos", "log"};

    std::vector<std::string> colNames;

    std::vector<double> currPopFitness;
    
  public:
    GPStruct(int populationSize, std::vector<std::vector<double>> dataset, int gen, int depth, std::vector<double> aR, int tournamentSize, std::vector<std::string> colNamesm, int seed);
    ~GPStruct();

    // initial population
    void generateIndividual(GPNodeStruct* root, int maxDepth);
    std::string randomTerminal();
    std::string randomOperator();
    void cachePopulation(int run, bool TL = false);

    // training & testing
    void train(int run = 0, int gen = 0);
    double test(int run = 0, bool TL = false);
    double avgDepth();

    // transfer learning
    void transferLearning(std::vector<std::vector<double>> dataset, int gen, std::vector<double> aR, std::vector<std::string> additionalColNames, int topK);
    
    // selection method
    std::vector<GPNodeStruct*> tournamentSelection(bool TL = false);
    
    // genetic operators
    void mutation(const GPNodeStruct& tree);
    void crossover(const GPNodeStruct& tree1, const GPNodeStruct& tree2);
    
    // metrics
    GPNodeStruct* bestTree();
    double fitness(const GPNodeStruct& tree, const std::string& set, bool recal = false);
    double populationFitness();
    
    // misc
    GPNodeStruct* getIndividual(const int& index);
    int getIndex(const GPNodeStruct& tree);
    void updateFitness(const GPNodeStruct& tree);
    void updateColNames(const std::vector<std::string>& colNames);
    void vizTree(GPNodeStruct* tree);
    void appendToCSV(std::vector<std::string> input);
    void diversityCalc(std::vector<std::string> input);
    bool isUnary(std::string value);
};

#endif // GP_STRUCT_H
