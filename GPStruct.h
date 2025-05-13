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

    // for structure-based GP
    int globalThreshold = 3;
    int localThreshold = 3;
    int cutoffDepth = 4;
    bool isGlobalSearch = true;
    bool isNodeAboveCutoff(const GPNodeStruct& node);

    std::vector<GPNodeStruct*> population;

    std::vector<std::vector<double>> training;
    std::vector<std::vector<double>> testing;
    std::vector<std::vector<double>> validation;
    
    // add valid float terminals
    std::vector<std::string> validFloatTerminals = {"double"}; // [is a float]
    // add valid boolean terminals
    std::vector<std::string> validBooleanTerminals = {}; // [is a boolean]

    const std::vector<std::string> validOperators = {"+", "*", "-", "/", "max", "min"}; // [returns float]
    const std::vector<std::string> validUnaryOperators = {"tanh", "sin", "cos", "log"}; // [returns float]
    // for structure-based GP, we add conditional operators
    // const std::vector<std::string> validLogicalOperators = {"and", "or", "not"}; // [returns boolean]
    // params:
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
    void generateIndividual(GPNodeStruct* root, int maxDepth, bool logical);
    std::string randomTerminal(bool parentRequiresBoolean);
    std::string randomOperator(bool isConditional = false);
    void cachePopulation(int run);

    // training & testing
    void train(int run = 0, bool structureBased = false);
    double test(int run = 0);
    
    // structure-based GP
    void setParameters(int globalThreshold, int localThreshold, int cutoffDepth);
    int globalIndex(GPNodeStruct* tree);
    int computeGlobalSimilarity(GPNodeStruct* tree1, GPNodeStruct* tree2, int currentDepth = 0);
    int computeLocalSimilarity(GPNodeStruct* tree1, GPNodeStruct* tree2, int currentDepth = 0);
    int localIndex(GPNodeStruct* tree);
    
    // selection method
    std::vector<GPNodeStruct*> tournamentSelection(bool inverse = false);
    
    // genetic operators
    void mutation(const GPNodeStruct& tree);
    void crossover(GPNodeStruct* tree1, GPNodeStruct* tree2);
    
    // metrics
    GPNodeStruct* bestTree();
    double fitness(const GPNodeStruct& tree, const std::string& set, bool recal = false);
    double populationFitness();
    
    // misc
    double avgDepth();
    GPNodeStruct* getIndividual(const int& index);
    int getIndex(const GPNodeStruct& tree);
    void updateFitness(const GPNodeStruct& tree);
    void appendToCSV(std::vector<std::string> input);
    int requiredOperands(std::string value);
    bool isBooleanParent(std::string value);
    int nodeLevel(GPNodeStruct* root, GPNodeStruct* targetNode);
    bool isBooleanTerminal(std::string value);
    void printTree(const GPNodeStruct* root, const GPNodeStruct* origin = nullptr, int depth = 0);
    void printTree(const GPNodeStruct& root, int depth = 0);
    void printPopulation();
};

#endif // GP_STRUCT_H
