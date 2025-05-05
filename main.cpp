#include <iostream>
#include <fstream>
#include <random>
#include "GP.h"
// #include "GPStruct.h"
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <numeric>

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
    if (std::find(booleanColumns.begin(), booleanColumns.end(), token) != booleanColumns.end()) {
      columnTypes.push_back({token, 0});
    } else {
      columnTypes.push_back({token, 1});
    }
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

  std::cout << "Dataset size: " << ds->data.size() << " x " << ds->data[0].size() << std::endl;
  std::cout << "Column names: ";
  std::cout << std::endl;
  for (const auto& name : ds->columnTypes) {
    std::cout << name.first << " (" << (name.second == 0 ? "boolean" : "float") << "), " << std::endl;
  }
  std::cout << "___________________" << std::endl;

  std::cout << "Dataset '" << datasetName << "' loaded!" << std::endl;

  return ds;
}

int main() {
  Dataset* dataset = fetchDataset("preprocessing/hepatitis_cleaned.tsv");

  if (!dataset) {
    std::cerr << "Failed to load datasets" << std::endl;
    return 1;
  }

  std::ofstream file("outputs.csv", std::ios::trunc);
  if (!file.is_open()) {
    std::cerr << "Failed to open outputs.csv" << std::endl;
  }
  file << "prediction,target" << std::endl;
  file.close();

  // std::ofstream file2("diversity.csv", std::ios::trunc);
  // if (!file2.is_open()) {
  //   std::cerr << "Failed to open diversity.csv" << std::endl;
  // }
  // file2 << "generation,individual,TL" << std::endl;
  // file2.close();

  // still store column names
  std::vector<std::string> columnNames;
  for (const auto& name : dataset->columnTypes) {
    columnNames.push_back(name.first);
  }

  int populationSize = 1;
  int maxDepth = 3;
  int maxGenerations = 0;
  std::vector<double> applicationRates = {0.85, 0.05}; // crossoverRate, mutationRate
  int tournamentSize = 4;
  int runs = 1; // each run includes transfer learning
  std::vector<GP*> gps;

  gps.resize(runs);

  // across-run-stats
  std::vector<double> bestFitness(runs);
  std::vector<double> avgDuration(runs);

  for (int i = 0; i < runs; i++) {
    // start chrono
    auto start = std::chrono::high_resolution_clock::now();
    gps[i] = new GP(populationSize, dataset->data, maxGenerations, maxDepth, applicationRates, tournamentSize, columnNames, i);
    gps[i]->cachePopulation(i);
    std::srand(i);
    gps[i]->train(i);
    bestFitness[i] = gps[i]->test(i, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Run " << i+1 << "/" << runs << " completed in " << elapsed.count() << " seconds" << std::endl;
    std::cout << "___________________" << std::endl;

    // store the duration and TL duration
    avgDuration[i] = elapsed.count();
  }

  // print the results
  std::cout << "Results:" << std::endl;
  std::cout << "Run\tBest MSE\tDuration" << std::endl;
  for (int i = 0; i < runs; i++) {
    std::cout << i+1 << "\t" << bestFitness[i] << "\t" 
              << avgDuration[i] << "s" << std::endl;
  }
  std::cout << "___________________" << std::endl;
  std::cout << "Average Best MSE: " << std::accumulate(bestFitness.begin(), bestFitness.end(), 0.0) / runs << std::endl;
  std::cout << "Average Duration: " << std::accumulate(avgDuration.begin(), avgDuration.end(), 0.0) / runs << " seconds (Total duration: " << std::accumulate(avgDuration.begin(), avgDuration.end(), 0.0) << " seconds)" << std::endl;
  std::cout << "___________________" << std::endl;

  for (int i = 0; i < runs; i++) {
    delete gps[i];
  }

  return 0;
}