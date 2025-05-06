#include "GPStruct.h"

GPStruct::GPStruct(int populationSize, std::vector<std::vector<double>> dataset, int gen, int depth, std::vector<double> aR, int tournamentSize, std::vector<std::pair<std::string, int>> columnTypes, int seed) {
  this->seed = seed;

  for (int i = 0; i < columnTypes.size(); i++) {
    if (columnTypes[i].second == 0) {
      this->validBooleanTerminals.push_back(columnTypes[i].first);
    } else if (columnTypes[i].second == 1) {
      this->validFloatTerminals.push_back(columnTypes[i].first);
    }
  }

  // print float and boolean
  std::cout << "\033[33m" "Valid Float Terminals: ";
  for (const auto& terminal : validFloatTerminals) {
    std::cout << terminal << ", ";
  }
  std::cout << "\033[0m" << std::endl;
  std::cout << "\033[31m" "Valid Boolean Terminals: ";
  for (const auto& terminal : validBooleanTerminals) {
    std::cout << terminal << ", ";
  }
  std::cout << "\033[0m" << std::endl;
  
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
    generateIndividual(population[i], population[i], maxDepth, std::rand() % 2 == 0);
    // vizTree(population[i]);
    std::cout << "___________________" << std::endl;
  }
}

GPStruct::~GPStruct() {
  for (int i = 0; i < populationSize; i++) {
    delete population[i];
  }
}

void GPStruct::cachePopulation(int run, bool TL) {
  this->currPopFitness = std::vector<double>(populationSize);
  double F1_sum = 0.0;
  for (int i = 0; i < populationSize; i++) {
    double tF = fitness(*population[i], "train", true);
    // double tFtest = fitness(population[i], "test", true);
    double tFtest = 0.0;
    currPopFitness[i] = tF;
    std::cout << "\033[90m" "Caching Individual " << i+1 << "/" << populationSize << " [F1: " << std::to_string(tF) << "]" << std::endl << "\033[0m";
    F1_sum += tF;
  }
  std::cout << "\033[31m" << "Initial Generation Complete [Average F1: " << std::to_string(F1_sum/populationSize) << "]" << std::endl << "\033[0m";
}

// initial population
void GPStruct::generateIndividual(GPNodeStruct* origin, GPNodeStruct* root, int maxDepth, bool logical) {
  if (maxDepth == 0) {
    root->value = randomTerminal(logical);
    root->isLeaf = true;
    root->children = {};

    // Generate a random double value between 0 and 1
    if (root->value == "double") {
      root->value = std::to_string(static_cast<double>(std::rand()) / RAND_MAX * 1.0);
    }

    if (origin != nullptr) {
      std::string tabs = "";
      std::string closingTabs = "";
      for (int i = 0; i < nodeLevel(origin, root); i++) {
        tabs += "\t";
        if (i != 0) {
          closingTabs += "\t";

        }
      }
      // find if boolean, then print the colour
      if (isBooleanTerminal(root->value)) {
        std::cout << "\033[31m" << tabs << root->value << std::endl << "\033[0m";
      } else {
        std::cout << "\033[33m" << tabs << root->value << std::endl << "\033[0m";
      }
    }
    return;
  }
  
  root->value = randomOperator(logical);
  root->isLeaf = false;

  if (origin != nullptr) {
    std::string tabs = "";
    for (int i = 0; i < nodeLevel(origin, root); i++) {
      tabs += "\t";
    }
    std::cout << tabs << root->value << std::endl;
  }
  
  int required = requiredOperands(root->value);
  // std::cout << root->value << ";" << required << std::endl;

  // if it's an if statement, we need to make sure the first child is a boolean
  if (required == 3) {
    root->children.push_back(new GPNodeStruct());
    // std::cout << "Parent " << root->value << " [isLogical: " << true << "] with operator " << root->children[0]->value << std::endl;
    generateIndividual(origin, root->children[0], (maxDepth - 1 == 0 ? 1 : maxDepth - 1), true); // true because we want a boolean as result
    required--;
  }

  for (int i = 0; i < required; i++) {
    root->children.push_back(new GPNodeStruct());
    float probability = static_cast<double>(maxDepth) / this->maxDepth; // Decreases as tree grows
    bool grow = (static_cast<double>(std::rand()) / RAND_MAX) < probability;
    bool isLogical = isBooleanParent(root->value);
    // std::cout << "Parent " << root->value << " [isLogical: " << isLogical << "]" << std::endl;
    if (grow) {
      generateIndividual(origin, root->children[i], maxDepth - 1, isLogical); 
    } else {
      generateIndividual(origin, root->children[i], 0, isLogical);
    }
  }

  // just a dead end after this
}

std::string GPStruct::randomTerminal(bool parentRequiresBoolean) {
  if (parentRequiresBoolean) {
    return validBooleanTerminals[std::rand() % validBooleanTerminals.size()];
  }
  return validFloatTerminals[std::rand() % validFloatTerminals.size()];
}

std::string GPStruct::randomOperator(bool requiresBoolean) {
  if (requiresBoolean) {
      std::vector<std::string> booleanOperators;
      booleanOperators.insert(booleanOperators.end(), validLogicalOperators.begin(), validLogicalOperators.end());
      booleanOperators.insert(booleanOperators.end(), validComparisonOperators.begin(), validComparisonOperators.end());

      return booleanOperators[std::rand() % booleanOperators.size()];
  }
  
  // Add standard arithmetic operators
  std::vector<std::string> floatOperators;
  floatOperators.insert(floatOperators.end(), validOperators.begin(), validOperators.end());
  floatOperators.insert(floatOperators.end(), validUnaryOperators.begin(), validUnaryOperators.end());
  floatOperators.insert(floatOperators.end(), validConditionalOperators.begin(), validConditionalOperators.end());
  
  return floatOperators[std::rand() % floatOperators.size()];
}

int GPStruct::requiredOperands(std::string value) {
  if (value == "if") return 3;
  if (value == "not") return 1;
  for (int i = 0; i < validUnaryOperators.size(); i++) {
    if (validUnaryOperators[i] == value) return 1;
  }

  // check if not a terminal
  for (int i = 0; i < validFloatTerminals.size(); i++) {
    if (validFloatTerminals[i] == value) return 0;
  }
  for (int i = 0; i < validBooleanTerminals.size(); i++) {
    if (validBooleanTerminals[i] == value) return 0;
  }

  // catch-all for all other operators
  return 2;
}

bool GPStruct::isBooleanParent(std::string value) {
  for (int i = 0; i < validLogicalOperators.size(); i++) {
    if (validLogicalOperators[i] == value) return true;
  }
  return false;
}

bool GPStruct::isBooleanTerminal(std::string value) {
  for (int i = 0; i < validBooleanTerminals.size(); i++) {
    if (validBooleanTerminals[i] == value) return true;
  }
  return false;
}

int GPStruct::nodeLevel(GPNodeStruct* root, GPNodeStruct* targetNode) {
  if (!root || !targetNode) {
      return -1;  // Invalid input
  }
  
  // Iterative approach using BFS
  std::queue<std::pair<GPNodeStruct*, int>> nodeQueue;
  nodeQueue.push({root, 0});  // Root is at level 0
  
  while (!nodeQueue.empty()) {
      auto [currentNode, level] = nodeQueue.front();
      nodeQueue.pop();
      
      // If this is the target node, return its level
      if (currentNode == targetNode) {
          return level;
      }
      
      // Add all children to the queue with incremented level
      for (auto* child : currentNode->children) {
          if (child) {
              nodeQueue.push({child, level + 1});
          }
      }
  }
  
  return -1;  // Node not found in the tree
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
  double treeFitness = fitness(*tree, "test", true);
  // vizTree(tree);
  std::cout << "Testing Results" << std::endl << "F1: " << std::to_string(treeFitness) << std::endl;

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
      if (tF > wF) { // * maximize fitness
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
    generateIndividual(nullptr, temp, std::rand() % maxDepth, isBooleanParent(temp->value));
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
  
  // swop all the children of the parents
  for (int i = 0; i < tempParent1->children.size(); i++) {
    if (tempParent1->children[i] == temp1) {
      tempParent1->children[i] = temp2;
    }
  }
  for (int i = 0; i < tempParent2->children.size(); i++) {
    if (tempParent2->children[i] == temp2) {
      tempParent2->children[i] = temp1;
    }
  }
}

// metrics
GPNodeStruct* GPStruct::bestTree() {
  double bestFitness = -INFINITY; // * maximize fitness
  GPNodeStruct* currBestTree = nullptr;
  for (int i = 0; i < populationSize; i++) {
    double tF = fitness(*population[i], "train");
    if (tF > bestFitness) { // * maximize fitness
      bestFitness = tF;
      currBestTree = population[i];
    }
    // std::cout << "Individual " << i+1 << "/" << populationSize << " [F1: " << std::to_string(tF) << "]" <<  std::endl;
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
      std::cerr << "Invalid fitness value" << std::endl;
      return 0.0; // bad F1-score
    }

    double actual = dataset[i][dataset[0].size() - 1];

    // clip with a function to avoid overflow, we want the output to be normalised as well...
    treeFitness = 1 / (1 + exp(-treeFitness));

    double diff = treeFitness - actual;
    // std::cout << "TF: " << std::to_string(treeFitness) << " | Actual: " << std::to_string(actual) << " | Diff: " << std::to_string(diff) << std::endl;


    // use threshold to determine if the prediction is correct. If the tree returns 0.0 (false), then it will remain false with the threshold
    if (treeFitness > threshold) {
      treeFitness = 1.0;
    } else {
      treeFitness = 0.0;
    }

    // determine the confusion matrix
    if (treeFitness == 1.0 && actual == 0.0) {
      confusionMatrix[0][0] += 1.0; // TP
    } else if (treeFitness == 1.0 && actual == 1.0) {
      confusionMatrix[0][1] += 1.0; // FP
    } else if (treeFitness == 0.0 && actual == 0.0) {
      confusionMatrix[1][0] += 1.0; // FN
    } else if (treeFitness == 0.0 && actual == 1.0) {
      confusionMatrix[1][1] += 1.0; // TN
    }
  }

  if (confusionMatrix[0][0] + confusionMatrix[0][1] == 0 || confusionMatrix[0][0] + confusionMatrix[1][0] == 0) {
    // std::cerr << "Invalid confusion matrix" << std::endl;
    return 0.0; // avoid division by zero
  }
  
  // calculate the precision and recall
  double precision = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[0][1]);
  double recall = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]);

  if (precision + recall == 0) {
    // std::cerr << "Invalid precision and recall" << std::endl;
    return 0.0; // avoid division by zero
  }

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