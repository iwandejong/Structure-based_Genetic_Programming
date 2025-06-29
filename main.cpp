#include <iostream>
#include <fstream>
#include <random>
#include "GPStruct.h"
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstdlib>

class Dataset {
  public:
    std::vector<std::vector<double>> data;
    // 0: boolean, 1: float
    std::vector<std::pair<std::string, int>> columnTypes;
};

Dataset* fetchDataset(std::string datasetName) {
  if (datasetName.empty()) {
    std::cerr << "No dataset name provided" << std::endl;
    return nullptr;
  }

  std::ifstream file(datasetName);
  try {
    if (!file.is_open()) {
      std::cerr << "Failed to open dataset: " << datasetName << std::endl;
      return nullptr;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error opening file: " << e.what() << std::endl;
    return nullptr;
  }

  std::string line;

  std::getline(file, line);
  std::vector<std::pair<std::string, int>> columnTypes;
  std::stringstream ss(line);
  std::string token;

  std::vector<std::string> booleanColumns = {"sex", "antivirals", "fatigue", "malaise", "anorexia", "histology"};

  while (std::getline(ss, token, '\t')) {
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    if (token.empty() || token == "target") continue;

    
    // Check if token is a boolean column
    int type = (std::find(booleanColumns.begin(), booleanColumns.end(), token) != booleanColumns.end()) ? 0 : 1;
    columnTypes.push_back({token, type});
  }

  std::vector<std::vector<double>> fullDataset;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string token;
    std::vector<double> rowData;

    while (std::getline(ss, token, '\t')) {
      try {
        rowData.push_back(std::stod(token));
      } catch (const std::invalid_argument&) {
        rowData.push_back(0.0);  // In case of an invalid number, default to 0
      }
    }

    fullDataset.push_back(rowData);
  }
  file.close();
  
  Dataset* ds = new Dataset();
  ds->data = fullDataset;
  ds->columnTypes = columnTypes;

  return ds;
}

void run() {
  Dataset* dataset = fetchDataset("hepatitis_cleaned.tsv");

  if (!dataset) {
    std::cerr << "Failed to load datasets" << std::endl;
    return;
  }

  // std::ofstream file("outputs.csv", std::ios::trunc);
  // if (!file.is_open()) {
  //   std::cerr << "Failed to open outputs.csv" << std::endl;
  // }
  // file << "run,generation,populationFitness,bestTree,action,structured" << std::endl;
  // file.close();

  std::vector<std::string> columnNames;
  for (const auto& name : dataset->columnTypes) {
    columnNames.push_back(name.first);
  }

  // setup for normal GP
  int populationSize = 35;
  int maxDepth = 7; // initial depth, can grow indefinitely
  int maxGenerations = 80;
  std::vector<double> applicationRates = {0.6, 0.25}; // crossoverRate, mutationRate
  int tournamentSize = 7;

  // setup for structure-based GP
  int populationSizeStruct = 35;
  int maxDepthStruct = 6; // initial depth, can grow indefinitely
  int maxGenerationsStruct = 110;
  std::vector<double> applicationRatesStruct = {0.5, 0.25}; // crossoverRate, mutationRate
  int tournamentSizeStruct = 7;

  int runs = 1; // only 1 run for submission purposes
  std::vector<GPStruct*> gps;
  std::vector<GPStruct*> gp_structs;

  gps.resize(runs);
  gp_structs.resize(runs);

  std::vector<double> bestFitness(runs);
  std::vector<double> bestFitnessStruct(runs);
  std::vector<double> avgDuration(runs);
  std::vector<double> avgDurationStruct(runs);

  for (int i = 0; i < runs; i++) {
    std::srand(i);
    
    // normal GP
    auto start = std::chrono::high_resolution_clock::now();
    gps[i] = new GPStruct(populationSize, dataset->data, maxGenerations, maxDepth, applicationRates, tournamentSize, dataset->columnTypes, i);
    gps[i]->train(i);
    bestFitness[i] = gps[i]->test(i);
    auto end = std::chrono::high_resolution_clock::now();

    // reset the seed
    std::srand(i);
    
    // structure-based GP
    auto start_struct = std::chrono::high_resolution_clock::now();
    gp_structs[i] = new GPStruct(populationSizeStruct, dataset->data, maxGenerationsStruct, maxDepthStruct, applicationRatesStruct, tournamentSizeStruct, dataset->columnTypes, i);
    gp_structs[i]->train(i, true);
    bestFitnessStruct[i] = gp_structs[i]->test(i);
    auto end_struct = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::chrono::duration<double> elapsed2 = end_struct - start_struct;
    std::cout << "Normal GP for run " << i+1 << " completed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Structure-based GP for run " << i+1 << " completed in " << elapsed2.count() << " seconds" << std::endl;
    std::cout << "___________________" << std::endl;
    
    avgDuration[i] = elapsed.count();
    avgDurationStruct[i] = elapsed2.count();
  }

  // print the results
  std::cout << "Testing Results:" << std::endl;
  std::cout << "Run\tBest BACC\tStruct-BACC\tTime\t\tStructTime" << std::endl;
  for (int i = 0; i < runs; i++) {
    std::cout << i+1 << "\t" << std::to_string(bestFitness[i]) << "\t" << std::to_string(bestFitnessStruct[i]) << "\t" << std::to_string(avgDuration[i]) << "\t" << std::to_string(avgDurationStruct[i]) << std::endl;
  }
  std::cout << "___________________" << std::endl;
  std::cout << "Average Best BACC: " << std::accumulate(bestFitness.begin(), bestFitness.end(), 0.0) / runs << std::endl;
  std::cout << "Average Struct-BACC: " << std::accumulate(bestFitnessStruct.begin(), bestFitnessStruct.end(), 0.0) / runs << std::endl;
  std::cout << "Average Duration: " << std::accumulate(avgDuration.begin(), avgDuration.end(), 0.0) / runs << " seconds (Total duration: " << std::accumulate(avgDuration.begin(), avgDuration.end(), 0.0) << " seconds)" << std::endl;
  std::cout << "Average Struct Duration: " << std::accumulate(avgDurationStruct.begin(), avgDurationStruct.end(), 0.0) / runs << " seconds (Total duration: " << std::accumulate(avgDurationStruct.begin(), avgDurationStruct.end(), 0.0) << " seconds)" << std::endl;
  std::cout << "___________________" << std::endl;

  for (int i = 0; i < runs; i++) {
    delete gps[i];
  }
}

int main() {
  run();
  return 0;
}