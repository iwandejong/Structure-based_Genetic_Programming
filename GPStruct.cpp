#include "GPStruct.h"

GPStruct::GPStruct(int populationSize, std::vector<std::vector<double>> dataset, int gen, int depth, std::vector<double> aR, int tournamentSize, std::vector<std::pair<std::string, int>> columnTypes, int seed) {
  this->seed = seed;

  this->validFloatTerminals = std::vector<std::string>(columnTypes.size());
  this->validBooleanTerminals = std::vector<std::string>(columnTypes.size());
  for (int i = 0; i < columnTypes.size(); i++) {
    if (columnTypes[i].second == 0) {
      this->validBooleanTerminals.push_back(columnTypes[i].first);
    } else {
      this->validFloatTerminals.push_back(columnTypes[i].first);
    }
  }
  
  population = std::vector<GPNodeStruct*>(populationSize);

  // update colnames
  this->colNames = std::vector<std::string>(columnTypes.size());
  for (int i = 0; i < columnTypes.size(); i++) {
    this->colNames[i] = columnTypes[i].first;
  }

  double trainSplit = 0.8;
  // double validationSplit = 0.1;
  int numSamples = dataset.size();
  int trainSize = static_cast<int>(numSamples * trainSplit);
  int testSize = numSamples - trainSize;

  // Create and shuffle index vector
  std::vector<int> indices(numSamples);
  for (int i = 0; i < numSamples; ++i) {
    indices[i] = i;
  }
  std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

  this->training = std::vector<std::vector<double>>(trainSize);
  this->testing = std::vector<std::vector<double>>(testSize);

  for (int i = 0; i < trainSize; ++i) {
    this->training[i] = dataset[indices[i]];
  }

  for (int i = 0; i < testSize; ++i) {
    this->testing[i] = dataset[indices[trainSize + i]];
  }

  this->populationSize = populationSize;
  this->maxGenerations = gen;
  this->maxDepth = depth;
  this->crossoverRate = aR[0];
  this->mutationRate = aR[1];
  // reproduction rate is the remaining part of the application rate (1 - crossoverRate - mutationRate)
  this->tournamentSize = tournamentSize;

  for (int i = 0; i < populationSize; i++) {
    population[i] = new GPNodeStruct();
    generateIndividual(population[i], maxDepth);
    // vizTree(population[i]);
  }
}

GPStruct::~GPStruct() {
  for (int i = 0; i < populationSize; i++) {
    delete population[i];
  }
}

void GPStruct::cachePopulation(int run, bool TL) {
  this->currPopFitness = std::vector<double>(populationSize);
  double mse_sum = 0.0;
  for (int i = 0; i < populationSize; i++) {
    double tF = fitness(*population[i], "train", true);
    // double tFtest = fitness(population[i], "test", true);
    double tFtest = 0.0;
    currPopFitness[i] = tF;
    std::cout << "\033[90m" "Caching Individual " << i+1 << "/" << populationSize << " [MSE: " << std::to_string(tF) << "]" << std::endl << "\033[0m";
    mse_sum += tF;
  }
  std::cout << "\033[31m" << "Initial Generation Complete [Average MSE: " << std::to_string(mse_sum/populationSize) << "]" << std::endl << "\033[0m";
}

// initial population
void GPStruct::generateIndividual(GPNodeStruct* root, int maxDepth, std::string parentType) {
  if (maxDepth == 0) {
    root->value = randomTerminal(parentType);
    root->isLeaf = true;
    root->left = nullptr;
    root->right = nullptr;
    
    // Generate a random double value between -1 and 1
    // double newDouble = static_cast<double>(std::rand()) / RAND_MAX * 1.0;
    double newDouble = static_cast<double>(std::rand()) / RAND_MAX * 1.0 - 0.5;
    if (root->value == "double") {
      root->value = std::to_string(newDouble);
    }
    return;
  }
  
  root->value = randomOperator();
  root->isLeaf = false;

  if (isUnary(root->value)) {
    root->left = new GPNodeStruct();
    root->right = nullptr;
    float probability = static_cast<double>(maxDepth) / this->maxDepth; // Decreases as tree grows
    bool growLeft = (static_cast<double>(std::rand()) / RAND_MAX) < probability;
    
    if (growLeft) {
      generateIndividual(root->left, maxDepth - 1);
    } else {
      generateIndividual(root->left, 0);
    }
  } else {
    root->left = new GPNodeStruct();
    root->right = new GPNodeStruct();
    float probability = static_cast<double>(maxDepth) / this->maxDepth; // Decreases as tree grows
    bool growLeft = (static_cast<double>(std::rand()) / RAND_MAX) < probability;
    bool growRight = (static_cast<double>(std::rand()) / RAND_MAX) < probability;
    
    if (growLeft) {
      generateIndividual(root->left, maxDepth - 1);
    } else {
      generateIndividual(root->left, 0);
    }
    
    if (growRight) {
      generateIndividual(root->right, maxDepth - 1);
    } else {
      generateIndividual(root->right, 0);
    }
  }
  
}

std::string GPStruct::randomTerminal(std::string parentType) {
  if (parentType == "double") {
    return validFloatTerminals[std::rand() % validFloatTerminals.size()];
  } else if (parentType == "boolean") {
    return validBooleanTerminals[std::rand() % validBooleanTerminals.size()];
  } else {
    return validFloatTerminals[std::rand() % validFloatTerminals.size()];
  }
}

std::string GPStruct::randomOperator() {
  int randomIndex = std::rand() % (validOperators.size() + validUnaryOperators.size() + validConditionalOperators.size() + validComparisonOperators.size());
  if (randomIndex < validOperators.size()) {
    return validOperators[randomIndex];
  } else if (randomIndex < validOperators.size() + validUnaryOperators.size()) {
    return validUnaryOperators[randomIndex - validOperators.size()];
  } else if (randomIndex < validOperators.size() + validUnaryOperators.size() + validConditionalOperators.size()) {
    return validConditionalOperators[randomIndex - validOperators.size() - validUnaryOperators.size()];
  } else {
    return validComparisonOperators[randomIndex - validOperators.size() - validUnaryOperators.size() - validConditionalOperators.size()];
  }
}

bool GPStruct::isUnary(std::string value) {
  for (int i = 0; i < validUnaryOperators.size(); i++) {
    if (validUnaryOperators[i] == value) return true;
  }
  return false;
}

// training
void GPStruct::train(int run, int gen) {
  // 1: initial parents selection
  std::vector<GPNodeStruct*> parents = tournamentSelection();
  double summedFitness = 0.0;
  for (int i = 0; i < maxGenerations; i++) {
    // 2: operators
    double aR = static_cast<double>(std::rand()) / RAND_MAX;
    int action = 0;
    if (aR < crossoverRate) {
      crossover(*parents[0], *parents[1]);
      action = 1;
    } else if (aR < crossoverRate + mutationRate) {
      mutation(std::rand() % 2 == 0 ? *parents[0] : *parents[1]);
      action = 2;
    }
    // 4: print results & see if improved
    double popFitness = populationFitness();
    double bTF = fitness(*bestTree(), "train");
    // double bTFtest = fitness(bestTree(), "test", true);
    double bTFtest = 0.0;
    appendToCSV({std::to_string(run),std::to_string(i+gen),std::to_string(popFitness),std::to_string(bTF),std::to_string(action)});

    std::string colorAction = action == 0 ? "\033[91m" : action == 1 ? "\033[92m" : "\033[93m";
    std::string printAction = action == 0 ? "Reproduction" : action == 1 ? "Crossover" : "Mutation";
    std::cout << "Generation " << i+gen+1 << "/" << maxGenerations+gen << " [" << colorAction << printAction << "\033[0m" << "]" << std::endl;

    // 5 : select parents for next generation
    parents = tournamentSelection(gen != 0);
    // repeat

    // Checks to see if the tree structures are updated/changed
    // std::cout << population.size() << std::endl; // * check
    // std::cout << std::to_string(avgDepth()) << std::endl; // * check
  }
  // std::cout << "###########" << std::endl;
}

double GPStruct::avgDepth() {
  double sumDepth = 0.0;
  for (int i = 0; i < populationSize; i++) {
    sumDepth += population[i]->calcDepth();
  }
  return sumDepth / populationSize;
}

double GPStruct::test(int run, bool TL) {
  GPNodeStruct* tree = bestTree();
  if (!tree) {
    std::cerr << "No valid tree found!" << std::endl;
    return -1.0;
  }
  std::cout << "Best Tree Formula" << std::endl << tree->formula() << std::endl;
  double treeFitness = fitness(*tree, "test", true);
  appendToCSV({std::to_string(run),std::to_string(TL ? -2 : -1),std::to_string(-1),std::to_string(treeFitness),std::to_string(-1)});
  // vizTree(tree);
  std::cout << "Testing Results" << std::endl << "MSE: " << std::to_string(treeFitness) << std::endl;

  return treeFitness;
}

// selection method
std::vector<GPNodeStruct*> GPStruct::tournamentSelection(bool TL) {
  std::vector<GPNodeStruct*> newPopulation(tournamentSize);
  
  for (int p = 0; p < 2; p++) {
    std::vector<GPNodeStruct*> tempPopulation(tournamentSize);
    int randomIndividual = std::rand() % populationSize; 
    int initialRandom = randomIndividual;
    for (int x = 0; x < tournamentSize; x++) {
      tempPopulation[x] = population[randomIndividual];
      randomIndividual = std::rand() % populationSize;
    }
    
    GPNodeStruct* winner = tempPopulation[0];
    double wF = currPopFitness[initialRandom];
    for (int i = 0; i < tournamentSize; i++) {
      double tF = fitness(*tempPopulation[i], "train");
      // if (tF > wF) { // * inverse selection because of steady-state control model (not used, only experimental)
      if (tF < wF) {
      // if (TL ? tF < wF : tF > wF) { // * inverse selection because of steady-state control model (not used, only experimental)
        winner = tempPopulation[i];
        wF = tF;
      }
    }
    newPopulation[p] = winner;
  }

  return newPopulation;
}

// genetic operators
void GPStruct::mutation(const GPNodeStruct& tree) {
  // if (static_cast<double>(std::rand()) / RAND_MAX < mutationRate) {
    // std::cout << "Performing Mutation" << std::endl;
    // vizTree(tree);
    int mutationPoint = std::rand() % tree.treeSize();
    GPNodeStruct* temp = tree.traverseToNth(mutationPoint);
    delete temp->left;
    delete temp->right;
    generateIndividual(temp, std::rand() % maxDepth);
    // vizTree(tree);

    // recalculate fitness for the mutated tree
    updateFitness(tree);
  // }
}

void GPStruct::crossover(const GPNodeStruct& tree1, const GPNodeStruct& tree2) {
  // std::cout << "Performing Crossover" << std::endl;
  // std::cout << "################" << std::endl;
  // std::cout << (tree1 == tree2) << std::endl;
  // vizTree(tree1);
  // vizTree(tree2);
  int x = tree1.treeSize();
  int y = tree2.treeSize();
  
  int t1CP = std::rand() % x;
  int t2CP = std::rand() % y;
  
  GPNodeStruct* temp1 = tree1.traverseToNth(t1CP);
  GPNodeStruct* temp2 = tree2.traverseToNth(t2CP);
  // vizTree(temp1);
  // vizTree(temp2);

  GPNodeStruct* tempParent1 = tree1.findParent(temp1);
  GPNodeStruct* tempParent2 = tree2.findParent(temp2);

  if (!tempParent1 || !tempParent2 || temp1 == temp2) return;
  
  if (tempParent1) {
    if (tempParent1->left == temp1) {
      tempParent1->left = temp2;
    } else {
      tempParent1->right = temp2;
    }
  }

  if (tempParent2) {
    if (tempParent2->left == temp2) {
      tempParent2->left = temp1;
    } else {
      tempParent2->right = temp1;
    }
  }
  // vizTree(tree1);
  // std::cout << "################" << std::endl;
}

// metrics
GPNodeStruct* GPStruct::bestTree() {
  double bestFitness = INFINITY; // * minimize fitness
  GPNodeStruct* currBestTree = nullptr;
  for (int i = 0; i < populationSize; i++) {
    double tF = fitness(*population[i], "train");
    if (tF < bestFitness) { // * minimize fitness
      bestFitness = tF;
      currBestTree = population[i];
    }
    // std::cout << "Individual " << i+1 << "/" << populationSize << " [MSE: " << std::to_string(tF) << "]" <<  std::endl;
  }
  return currBestTree;
}

double GPStruct::fitness(const GPNodeStruct& tree, const std::string& set, bool recal) {
  double F1 = 0.0;
  double threshold = 0.5;
  double confusionMatrix[2][2] = {0.0}; // TP, TN, FP, FN

  std::vector<std::vector<double>> dataset;
  if (set == "train") {
    dataset = this->training;
  } else if (set == "test") {
    dataset = this->testing;
  } else {
    std::cerr << "Invalid dataset" << std::endl;
    return 1.0;
  }

  for (int i = 0; i < dataset.size(); i++) {
    double treeFitness = 0.0;
    if (recal) {
      // treeFitness = tree->fitness(dataset[i], colNames);
      treeFitness = tree.fitness(dataset[i], colNames);
    } else {
      // perform a simple lookup to get the value of the tree (saves a lot of time)
      treeFitness = currPopFitness[getIndex(tree)];
    }

    if (std::isnan(treeFitness) || std::isinf(treeFitness)) {
      return 0.0; // bad F1-score
    }

    double actual = dataset[i][dataset[0].size() - 1];

    // clip with a function to avoid overflow, we want the output to be normalised as well...
    treeFitness = 1 / (1 + exp(-treeFitness));

    // use threshold to determine if the prediction is correct
    if (treeFitness >= threshold) {
      treeFitness = 1.0;
    } else {
      treeFitness = 0.0;
    }

    // determine the confusion matrix
    if (treeFitness == 1.0 && actual == 1.0) {
      confusionMatrix[0][0] += 1.0;
    } else if (treeFitness == 1.0 && actual == 0.0) {
      confusionMatrix[0][1] += 1.0;
    } else if (treeFitness == 0.0 && actual == 1.0) {
      confusionMatrix[1][0] += 1.0;
    } else if (treeFitness == 0.0 && actual == 0.0) {
      confusionMatrix[1][1] += 1.0;
    }
  }

  if (confusionMatrix[0][0] + confusionMatrix[0][1] == 0 || confusionMatrix[0][0] + confusionMatrix[1][0] == 0) {
    return 0.0; // avoid division by zero
  }
  
  // calculate the precision and recall
  double precision = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1]);
  double recall = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]);
  double f1 = 2 * (precision * recall) / (precision + recall);

  return f1; // * maximize fitness
}

double GPStruct::populationFitness() {
  double totalFitness = 0.0;  
  for (int j = 0; j < populationSize; j++) {
    totalFitness += fitness(*population[j], "train");
  }
  return totalFitness / populationSize;
}

// misc
GPNodeStruct* GPStruct::getIndividual(const int& index) {
  return population[index];
}

int GPStruct::getIndex(const GPNodeStruct& tree) {
  for (int i = 0; i < populationSize; i++) {
    if (population[i] == &tree) {
      return i;
    }
  }
  return -1;
}

void GPStruct::updateFitness(const GPNodeStruct& tree) {
  int index = getIndex(tree);
  if (index != -1) {
    currPopFitness[index] = fitness(tree, "train", true); // recalculate fitness
  }
}

void GPStruct::vizTree(GPNodeStruct* tree) {
  // std::cout << "Has X: " << (tree->hasX() ? "Yes" : "No") << std::endl;
  std::cout << "f(x)=" << tree->formula() << std::endl;
  tree->print();
}

void GPStruct::appendToCSV(std::vector<std::string> input) {
  std::ofstream file("outputs.csv", std::ios::app);
  
  if (!file.is_open()) {
    std::cerr << "Failed to open outputs.csv" << std::endl;
    return;
  }

  for (const auto& value : input) {
    file << value;
    if (&value != &input.back()) {
      file << ",";
    }
  }
  file << std::endl;
  
  file.close();
}