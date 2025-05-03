#ifndef GP_H
#define GP_H

#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <random>
#include <fstream>
#include "GPNode.h"
#include <algorithm>

class GP {
  private:
    int seed;
    int maxDepth;
    int maxGenerations;
    double mutationRate;
    double crossoverRate;
    int populationSize;
    int tournamentSize;

    std::vector<GPNode*> population;

    std::vector<std::vector<double>> training;
    std::vector<std::vector<double>> testing;
    std::vector<std::vector<double>> validation;
    
    std::vector<std::string> validTerminals = {"double"};
    const std::vector<std::string> validOperators = {"+", "*", "-", "/", "max", "min"};
    const std::vector<std::string> validUnaryOperators = {"sigmoid", "sin", "cos", "log"};

    std::vector<std::string> colNames;

    std::vector<double> currPopFitness;
    
  public:
    GP(int populationSize, std::vector<std::vector<double>> dataset, int gen, int depth, std::vector<double> aR, int tournamentSize, std::vector<std::string> colNamesm, int seed);
    ~GP();

    // initial population
    void generateIndividual(GPNode* root, int maxDepth);
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
    std::vector<GPNode*> tournamentSelection(bool TL = false);
    
    // genetic operators
    void mutation(const GPNode& tree);
    void crossover(const GPNode& tree1, const GPNode& tree2);
    
    // metrics
    GPNode* bestTree();
    double fitness(const GPNode& tree, const std::string& set, bool recal = false);
    double populationFitness();
    
    // misc
    GPNode* getIndividual(const int& index);
    int getIndex(const GPNode& tree);
    void updateFitness(const GPNode& tree);
    void updateColNames(const std::vector<std::string>& colNames);
    void vizTree(GPNode* tree);
    void appendToCSV(std::vector<std::string> input);
    void diversityCalc(std::vector<std::string> input);
    bool isUnary(std::string value);
};

#endif // GP_H
