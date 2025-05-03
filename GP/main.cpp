#include <iostream>
#include <fstream>
#include <random>
#include "GP.h"
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <numeric>

class Dataset {
  public:
    std::vector<std::vector<double>> data;
    std::vector<std::string> columnNames;
};

Dataset* fetchDataset(std::string datasetName) {
  if (datasetName.empty()) {
    std::cerr << "No dataset name provided" << std::endl;
    return nullptr;
  }

  std::ifstream file(datasetName);
  std::string line;

  std::getline(file, line);
  std::vector<std::string> columnNames;
  std::stringstream ss(line);
  std::string token;

  while (std::getline(ss, token, '\t')) {
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    columnNames.push_back(token);
  }

  int numCols = columnNames.size();

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
  ds->columnNames = columnNames;

  std::cout << "Dataset size: " << ds->data.size() << " x " << ds->data[0].size() << std::endl;
  std::cout << "Column names: ";
  for (const auto& name : ds->columnNames) {
    std::cout << name << "\t";
  }
  std::cout << std::endl;

  std::cout << "Dataset '" << datasetName << "' loaded!" << std::endl;

  return ds;
}

int main() {
  Dataset* dataset = fetchDataset("227_cpu_small_cleaned.tsv");
  Dataset* datasetTL = fetchDataset("197_cpu_act_cleaned.tsv");

  std::ofstream file("outputs.csv", std::ios::trunc);
  if (!file.is_open()) {
    std::cerr << "Failed to open outputs.csv" << std::endl;
  }
  file << "run,generation,populationFitness,bestTree,action" << std::endl;
  file.close();

  std::ofstream file2("diversity.csv", std::ios::trunc);
  if (!file2.is_open()) {
    std::cerr << "Failed to open diversity.csv" << std::endl;
  }
  file2 << "generation,individual,TL" << std::endl;
  file2.close();

  std::vector<std::string> colNamesTL = datasetTL->columnNames;
  std::vector<std::string> colNames = dataset->columnNames;
  std::vector<std::string> uniqueColNames;

  std::sort(colNamesTL.begin(), colNamesTL.end());
  std::sort(colNames.begin(), colNames.end());

  std::set_difference(
    colNamesTL.begin(), colNamesTL.end(),
    colNames.begin(), colNames.end(),
    std::back_inserter(uniqueColNames)
  );

  std::cout << "Unique column names in datasetTL: ";
  for (const auto& name : uniqueColNames) {
    std::cout << name << "\t";
  }
  std::cout << std::endl;

  int populationSize = 50;
  int maxDepth = 4;
  int maxGenerations = 50;
  std::vector<double> applicationRates = {0.65, 0.05}; // crossoverRate, mutationRate
  int tournamentSize = 7;
  int runs = 10; // each run includes transfer learning
  std::vector<GP*> gps;

  gps.resize(runs);
  
  // for transfer learning, we increase the application rates and decrease the max generations
  applicationRates = {0.70, 0.05}; // crossoverRate, mutationRate
  int tlGenerations = 50;
  int topK = populationSize * 0.25; // top K individuals to keep for transfer learning

  // across-run-stats
  std::vector<double> bestFitness(runs);
  std::vector<double> bestTLFitness(runs);
  std::vector<double> avgDuration(runs);
  std::vector<double> avgTLDuration(runs);

  // std::vector<std::thread> threads;

  // auto cache_start = std::chrono::high_resolution_clock::now();
  // for (int i = 0; i < runs; i++) {
  //   threads.emplace_back([i, &gps, populationSize, dataset, maxGenerations, maxDepth, applicationRates, tournamentSize]() {
  //       std::srand(i);
  //     });
  // }
      
  // for (auto& t : threads) {
  //   t.join();
  // }
  // auto cache_end = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> cache_duration = cache_end - cache_start;
      
  for (int i = 0; i < runs; i++) {
    // start chrono
    auto start = std::chrono::high_resolution_clock::now();
    gps[i] = new GP(populationSize, dataset->data, maxGenerations, maxDepth, applicationRates, tournamentSize, dataset->columnNames, i);
    gps[i]->cachePopulation(i);
    std::srand(i);
    gps[i]->train(i);
    bestFitness[i] = gps[i]->test(i, false);
    auto startTL = std::chrono::high_resolution_clock::now();
    std::cout << "Performing transfer learning..." << std::endl;
    gps[i]->transferLearning(datasetTL->data, tlGenerations, applicationRates, uniqueColNames, topK); // changes the current GP setup to perform transfer learning
    gps[i]->cachePopulation(i, true);
    gps[i]->train(i,maxGenerations); // train again with transfer learning
    bestTLFitness[i] = gps[i]->test(i, true);
    auto end = std::chrono::high_resolution_clock::now();
    auto endTL = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::chrono::duration<double> elapsedTL = endTL - startTL;
    std::cout << "Run " << i+1 << "/" << runs << " completed in " << elapsed.count() << " seconds (" << elapsedTL.count() << " seconds for TL)" << std::endl;
    std::cout << "___________________" << std::endl;

    // store the duration and TL duration
    avgDuration[i] = elapsed.count();
    avgTLDuration[i] = elapsedTL.count();
  }

  // print the results
  std::cout << "Results:" << std::endl;
  std::cout << "Run\tBest MSE\tBest TL MSE\tMSE Diff\tDuration\tTL Duration" << std::endl;
  for (int i = 0; i < runs; i++) {
    std::cout << i+1 << "\t" << bestFitness[i] << "\t" << bestTLFitness[i] << "\t" << bestFitness[i] - bestTLFitness[i] << "\t" 
              << avgDuration[i] << "s" << "\t" << avgTLDuration[i] << "s" << std::endl;
  }
  // std::cout << "___________________" << std::endl;
  // std::cout << "Cache Duration for " << runs << " runs: " << cache_duration.count() << " seconds" << std::endl;
  std::cout << "___________________" << std::endl;
  std::cout << "Average Best MSE: " << std::accumulate(bestFitness.begin(), bestFitness.end(), 0.0) / runs << std::endl;
  std::cout << "Average Best TL MSE: " << std::accumulate(bestTLFitness.begin(), bestTLFitness.end(), 0.0) / runs << std::endl;
  std::cout << "Average Fitness Improvement: " << (std::accumulate(bestFitness.begin(), bestFitness.end(), 0.0) - std::accumulate(bestTLFitness.begin(), bestTLFitness.end(), 0.0)) / runs << std::endl;
  std::cout << "Average Duration: " << std::accumulate(avgDuration.begin(), avgDuration.end(), 0.0) / runs << " seconds (Total duration: " << std::accumulate(avgDuration.begin(), avgDuration.end(), 0.0) << " seconds)" << std::endl;
  std::cout << "Average TL Duration: " << std::accumulate(avgTLDuration.begin(), avgTLDuration.end(), 0.0) / runs << " seconds (Total TL duration: " << std::accumulate(avgTLDuration.begin(), avgTLDuration.end(), 0.0) << " seconds)" << std::endl;
  std::cout << "___________________" << std::endl;

  for (int i = 0; i < runs; i++) {
    delete gps[i];
  }

  return 0;
}