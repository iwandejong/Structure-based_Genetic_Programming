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
    
    // add valid float terminals
    std::vector<std::string> validFloatTerminals = {"double"};
    // add valid boolean terminals
    std::vector<std::string> validBooleanTerminals;

    const std::vector<std::string> validOperators = {"+", "*", "-", "/", "max", "min"};
    const std::vector<std::string> validUnaryOperators = {"sigmoid", "sin", "cos", "log"};
    // for structure-based GP, we add conditional operators
    const std::vector<std::string> validConditionalOperators = {"if", "ifelse", "and", "or", "not"};
    // params:
    // if: if(condition, trueBranch, falseBranch)
    // ifelse: ifelse(condition, trueBranch, falseBranch)
    // and: and(condition1, condition2)
    // or: or(condition1, condition2)
    // not: not(condition)
    const std::vector<std::string> validComparisonOperators = {"<", ">", "<=", ">=", "==", "!="};

    std::vector<std::string> colNames;

    std::vector<double> currPopFitness;
    
  public:
    GPStruct(int populationSize, std::vector<std::vector<double>> dataset, int gen, int depth, std::vector<double> aR, int tournamentSize, std::vector<std::pair<std::string, int>> columnTypes, int seed = 0);
    ~GPStruct();

    // initial population
    void generateIndividual(GPNodeStruct* root, int maxDepth, std::string parentType);
    std::string randomTerminal(std::string parentType);
    std::string randomOperator();
    void cachePopulation(int run, bool TL = false);

    // training & testing
    void train(int run = 0, int gen = 0);
    double test(int run = 0, bool TL = false);
    double avgDepth();
    
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
    void vizTree(GPNodeStruct* tree);
    void appendToCSV(std::vector<std::string> input);
    bool isUnary(std::string value);
};

#endif // GP_STRUCT_H
