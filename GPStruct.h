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
    std::vector<std::string> validFloatTerminals = {"double"}; // [is a float]
    // add valid boolean terminals
    std::vector<std::string> validBooleanTerminals = {}; // [is a boolean]

    const std::vector<std::string> validOperators = {"+", "*", "-", "/", "max", "min"}; // [returns float]
    const std::vector<std::string> validUnaryOperators = {"sigmoid", "sin", "cos", "log"}; // [returns float]
    // for structure-based GP, we add conditional operators
    const std::vector<std::string> validLogicalOperators = {"and", "or", "not"}; // [returns boolean]
    const std::vector<std::string> validConditionalOperators = {"if"}; // [returns float/boolean]
    // params:
    // if: if(condition, trueBranch, falseBranch) [3]
    // and: and(condition1, condition2) [2]
    // or: or(condition1, condition2) [2]
    // not: not(condition) [1]
    const std::vector<std::string> validComparisonOperators = {"<", ">", "<=", ">=", "==", "!="}; // [returns boolean]

    std::vector<std::string> colNames;

    std::vector<double> currPopFitness;
    
  public:
    GPStruct(int populationSize, std::vector<std::vector<double>> dataset, int gen, int depth, std::vector<double> aR, int tournamentSize, std::vector<std::pair<std::string, int>> columnTypes, int seed = 0);
    ~GPStruct();

    // initial population
    void generateIndividual(GPNodeStruct* origin, GPNodeStruct* root, int maxDepth, bool logical);
    std::string randomTerminal(bool parentRequiresBoolean);
    std::string randomOperator(bool isConditional = false);
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
    void appendToCSV(std::vector<std::string> input);
    int requiredOperands(std::string value);
    bool isBooleanParent(std::string value);
    int nodeLevel(GPNodeStruct* root, GPNodeStruct* targetNode);
    bool isBooleanTerminal(std::string value);
};

#endif // GP_STRUCT_H
