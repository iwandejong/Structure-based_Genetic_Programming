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
  
  population = std::vector<GPNodeStruct*>(populationSize);

  // update colnames
  this->colNames = std::vector<std::string>(columnTypes.size());
  for (int i = 0; i < columnTypes.size(); i++) {
    this->colNames[i] = columnTypes[i].first;
  }

  double trainSplit = 0.8;
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
    generateIndividual(population[i], maxDepth, std::rand() % booleanRandomizer == 0);
  }
}

GPStruct::~GPStruct() {
  for (int i = 0; i < populationSize; i++) {
    delete population[i];
  }
}

// initial population
void GPStruct::generateIndividual(GPNodeStruct* root, int maxDepth, bool logical) {
  if (maxDepth == 0) {
    root->value = randomTerminal(logical);
    root->isLeaf = true;
    root->children = {};

    // Generate a random double value between 0 and 1
    if (root->value == "double") {
      root->value = std::to_string(static_cast<double>(std::rand()) / RAND_MAX * 1.0 - 0.5); // Random float between -0.5 and 0.5
    }
    return;
  }
  
  root->value = randomOperator(logical);
  root->isLeaf = false;
  
  int required = requiredOperands(root->value);

  for (int i = 0; i < required; i++) {
    root->children.push_back(new GPNodeStruct());
    float probability = static_cast<double>(maxDepth) / this->maxDepth; // Decreases as tree grows
    bool grow = (static_cast<double>(std::rand()) / RAND_MAX) < probability;
    if (grow) {
      generateIndividual(root->children[i], maxDepth - 1, std::rand() % booleanRandomizer == 0);
    } else {
      generateIndividual(root->children[i], 0, std::rand() % booleanRandomizer == 0);
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
    // booleanOperators.insert(booleanOperators.end(), validLogicalOperators.begin(), validLogicalOperators.end());
    booleanOperators.insert(booleanOperators.end(), validComparisonOperators.begin(), validComparisonOperators.end());
    
    return booleanOperators[std::rand() % booleanOperators.size()];
  }
  
  // Add standard arithmetic operators
  std::vector<std::string> floatOperators;
  floatOperators.insert(floatOperators.end(), validOperators.begin(), validOperators.end());
  floatOperators.insert(floatOperators.end(), validUnaryOperators.begin(), validUnaryOperators.end());
  // floatOperators.insert(floatOperators.end(), validComparisonOperators.begin(), validComparisonOperators.end());
  
  return floatOperators[std::rand() % floatOperators.size()];
}

int GPStruct::requiredOperands(std::string value) {
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
void GPStruct::train(int run, bool structureBased) {
  // initial parents selection
  std::vector<GPNodeStruct*> parents = tournamentSelection();
  std::vector<GPNodeStruct*> losers = tournamentSelection(true);
  double summedFitness = 0.0;
  for (int i = 0; i < maxGenerations; i++) {
    // operators
    double aR = static_cast<double>(std::rand()) / RAND_MAX;
    int action = 0;

    std::cout << "\033[90m" "Winner 1: " << fitness(*parents[0], "train") << ", Winner 2: " << fitness(*parents[1], "train") << std::endl;

    // // print population
    // std::cout << "\033[34m" "Population: " << std::endl << "\033[0m";
    // for (int j = 0; j < populationSize; j++) {
    //   if (population[j] == parents[0] || population[j] == parents[1]) {
    //     std::cout << "\033[92m" << population[j] << " " "\033[0m";
    //   } else if (population[j] == losers[0] || population[j] == losers[1]) {
    //     std::cout << "\033[93m" << population[j] << " " "\033[0m";
    //   } else {
    //     std::cout << population[j] << " ";
    //   }
    // }
    // std::cout << std::endl << "\033[0m";

    // make deep copy of parents
    GPNodeStruct* offspring1 = new GPNodeStruct(*parents[0]);
    GPNodeStruct* offspring2 = new GPNodeStruct(*parents[1]);

    if (aR < crossoverRate) {
      crossover(offspring1, offspring2);
      action = 1;
    } else if (aR < crossoverRate + mutationRate) {
      mutation(*offspring1); // offspring1 is the winner
      action = 2;
    }

    int l1 = getIndex(*losers[0]);
    int l2 = getIndex(*losers[1]);
    std::cout << "\033[90m" "Offspring 1: " << fitness(*offspring1, "train") << ", Offspring 2: " << fitness(*offspring2, "train") << std::endl;
    std::cout << "\033[90m" "Loser 1: " << fitness(*losers[0], "train") << ", Loser 2: " << fitness(*losers[1], "train") << std::endl;

    if (structureBased) {
      if (isGlobalSearch) {
        int GI1 = globalIndex(offspring1);
        int GI2 = globalIndex(offspring2);
        std::cout << "\033[90m" "GI1: " << GI1 << ", GI2: " << GI2 << std::endl << "\033[0m";

        // ! Global search
        if (GI1 < globalThreshold) {
          population[l1] = offspring1;
          std::cout << "\033[35m" "APPROVED P1 (Global)" << std::endl << "\033[0m";
          
          // Start local search if promising individual found
          isGlobalSearch = false;
        }
        
        if (GI2 < globalThreshold) {
          population[l2] = offspring2;
          std::cout << "\033[35m" "APPROVED P2 (Global)" << std::endl << "\033[0m";
          
          // Start local search if promising individual found
          isGlobalSearch = false;
        }
      } else {
        int LI1 = localIndex(offspring1);
        int LI2 = localIndex(offspring2);
        std::cout << "\033[90m" "LI1: " << LI1 << ", LI2: " << LI2 << std::endl;

        // ! Local search
        if (LI1 < localThreshold) {
          population[l1] = offspring1;
          std::cout << "\033[35m" "APPROVED P1 (Local)" << std::endl << "\033[0m";
        } else {
          // If local search is not productive, switch back to global
          isGlobalSearch = true;
        }
        
        if (LI2 < localThreshold) {
          population[l2] = offspring2;
          std::cout << "\033[35m" "APPROVED P2 (Local)" << std::endl << "\033[0m";
        } else {
          // If local search is not productive, switch back to global
          isGlobalSearch = true;
        }
      }
    } else {
      population[l1] = offspring1;
      population[l2] = offspring2;
    }

    // std::cout << "\033[34m" "Population: " << std::endl << "\033[0m";
    // for (int j = 0; j < populationSize; j++) {
    //   if (population[j] == parents[0] || population[j] == parents[1]) {
    //     std::cout << "\033[92m" << population[j] << " " "\033[0m";
    //   } else if (population[j] == offspring1 || population[j] == offspring2) {
    //     std::cout << "\033[91m" << population[j] << " " "\033[0m";
    //   } else {
    //     std::cout << population[j] << " ";
    //   }
    // }
    // std::cout << std::endl << "\033[0m";

    // print results & see if improved
    double popFitness = populationFitness();
    std::cout << "\033[90m" "Population Fitness: " << std::to_string(popFitness) << std::endl << "\033[0m";
    double bTF = fitness(*bestTree(), "train");
    double bTFtest = 0.0;
    appendToCSV({std::to_string(run),std::to_string(i),std::to_string(popFitness),std::to_string(bTF),std::to_string(action),structureBased ? "1" : "0"});

    std::string colorAction = action == 0 ? "\033[91m" : action == 1 ? "\033[92m" : "\033[93m";
    std::string printAction = action == 0 ? "Reproduction" : action == 1 ? "Crossover" : "Mutation";
    std::cout << "Generation " << i+1 << "/" << maxGenerations << " [" << colorAction << printAction << "\033[0m" << "]" << std::endl;

    // 5 : select parents for next generation
    parents = tournamentSelection();
    losers = tournamentSelection(true);
  }
}

void GPStruct::setParameters(int globalThreshold, int localThreshold, int cutoffDepth) {
  this->globalThreshold = globalThreshold;
  this->localThreshold = localThreshold;
  this->cutoffDepth = cutoffDepth;
}

bool GPStruct::isNodeAboveCutoff(const GPNodeStruct& node) {
  if (node.isLeaf) return false;

  int depth = node.calcDepth();
  if (depth >= cutoffDepth) {
    return true;
  }

  for (const auto& child : node.children) {
    if (isNodeAboveCutoff(*child)) {
      return true;
    }
  }
  return false;
}

int GPStruct::globalIndex(GPNodeStruct* tree) {
  if (!tree) return 0;

  int GI = 0;
  for (int i = 0; i < populationSize; ++i) {
    if (population[i] != nullptr && population[i] != tree) {
      int treeSimilarity = computeGlobalSimilarity(tree, population[i]);
      if (treeSimilarity > GI) { // * maximum tree similarity
        GI = treeSimilarity;
      }
    }
  }
  return GI;
}

int GPStruct::localIndex(GPNodeStruct* tree) {
  if (!tree) return 0;

  int GI = 0;
  for (int i = 0; i < populationSize; ++i) {
    if (population[i] != nullptr && population[i] != tree) {
      int treeSimilarity = computeLocalSimilarity(tree, population[i]);
      if (treeSimilarity > GI) { // * maximum tree similarity
        GI = treeSimilarity;
      }
    }
  }
  return GI;
}

int GPStruct::computeGlobalSimilarity(GPNodeStruct* tree1, GPNodeStruct* tree2, int currentDepth) {
  if (!tree1 || !tree2) return 0; // invalid trees
  if (tree1->isLeaf || tree2->isLeaf) return 0; // global search can't contain terminals
  if (currentDepth >= cutoffDepth) return 0; // cutoff depth reached

  int similarity = (tree1->value == tree2->value) ? 1 : 0;

  int minChildren = std::min(tree1->children.size(), tree2->children.size());
  for (int i = 0; i < minChildren; ++i) {
    similarity += computeGlobalSimilarity(tree1->children[i], tree2->children[i], currentDepth + 1);
  }

  return similarity;
}


int GPStruct::computeLocalSimilarity(GPNodeStruct* tree1, GPNodeStruct* tree2, int currentDepth) {
  if (!tree1 || !tree2) return 0; // invalid trees

  int similarity = 0;
  if (currentDepth >= cutoffDepth && tree1->value == tree2->value) {
    similarity = 1; // match on value (function or terminal)
  }

  int minChildren = std::min(tree1->children.size(), tree2->children.size());
  for (int i = 0; i < minChildren; ++i) {
    similarity += computeLocalSimilarity(tree1->children[i], tree2->children[i], currentDepth + 1);
  }

  return similarity;
}


double GPStruct::avgDepth() {
  double sumDepth = 0.0;
  for (int i = 0; i < populationSize; i++) {
    sumDepth += population[i]->calcDepth();
  }
  return sumDepth / populationSize;
}

double GPStruct::test(int run) {
  GPNodeStruct* tree = bestTree();
  if (!tree) {
    std::cerr << "No valid tree found!" << std::endl;
    return -1.0;
  }
  double treeFitness = fitness(*tree, "test");
  std::cout << "Testing Results" << std::endl << "BACC: " << std::to_string(treeFitness) << std::endl;

  return treeFitness;
}

// selection method
std::vector<GPNodeStruct*> GPStruct::tournamentSelection(bool inverse) {
  std::vector<GPNodeStruct*> newPopulation(tournamentSize);
  std::vector<int> selectedIndices;

  for (int p = 0; p < 2; p++) {
    std::vector<GPNodeStruct*> tempPopulation(tournamentSize);
    int randomIndividual = std::rand() % populationSize; 
    int initialRandom = randomIndividual;
    selectedIndices.push_back(randomIndividual);
    for (int x = 0; x < tournamentSize; x++) {
      tempPopulation[x] = population[randomIndividual];
      randomIndividual = std::rand() % populationSize;
      while (std::find(selectedIndices.begin(), selectedIndices.end(), randomIndividual) != selectedIndices.end()) {
        randomIndividual = std::rand() % populationSize;
      }
      selectedIndices.push_back(randomIndividual);
    }
    
    GPNodeStruct* winner = nullptr;
    double wF = inverse ? INFINITY : -INFINITY;
    for (int i = 0; i < tournamentSize; i++) {
      double tF = fitness(*tempPopulation[i], "train");
      if (inverse ? tF < wF : tF > wF) {
        winner = tempPopulation[i];
        wF = tF;
      }
    }
    newPopulation[p] = winner;
    
    // swop if p-1 is less fit than p
    if (p > 0 && fitness(*newPopulation[p], "train") < fitness(*newPopulation[p-1], "train")) {
      std::swap(newPopulation[p], newPopulation[p-1]);
    }
  }

  return newPopulation;
}

// genetic operators
void GPStruct::mutation(const GPNodeStruct& tree) {
  int treeSize = tree.treeSize();
  int mutationPoint = std::rand() % treeSize;
  GPNodeStruct* node = tree.traverseToNth(mutationPoint);

  if (!isGlobalSearch && isNodeAboveCutoff(*node)) {
    std::cout << "\033[31m" "Mutation skipped (cutoff depth)" << std::endl << "\033[0m";
    return; // no mutation
  }
  
  if (std::rand() % 2 == 0) {
    // ! point mutation
    if (!node->isLeaf) {
      node->value = randomOperator(std::rand() % booleanRandomizer == 0);
    } else {
      GPNodeStruct* parent = tree.findParent(node);
      node->value = randomTerminal(std::rand() % booleanRandomizer == 0);
    }
  } else {
    // ! subtree mutation
    int maxSubtreeDepth = 4;
    generateIndividual(node, std::rand() % maxSubtreeDepth, std::rand() % booleanRandomizer == 0);
  }
}

void GPStruct::crossover(GPNodeStruct* tree1, GPNodeStruct* tree2) {
  int x = tree1->treeSize();
  int y = tree2->treeSize();
  
  if (tree1 == tree2 || x == 0 || y == 0) return; // no crossover
  
  for (int attempt = 0; attempt < 5; attempt++) {
    int t1CP = std::rand() % x;
    int t2CP = std::rand() % y;
    
    if (t1CP == 0 || t2CP == 0) continue; // skip root node (prevents pointers from being swapped)
    
    GPNodeStruct* temp1 = tree1->traverseToNth(t1CP);
    GPNodeStruct* temp2 = tree2->traverseToNth(t2CP);
    
    bool temp1IsBoolean = isBooleanTerminal(temp1->value);
    bool temp2IsBoolean = isBooleanTerminal(temp2->value);
      
    if (temp1IsBoolean == temp2IsBoolean) {
      GPNodeStruct* tempParent1 = tree1->findParent(temp1);
      GPNodeStruct* tempParent2 = tree2->findParent(temp2);
          
      if (!tempParent1 || !tempParent2 || temp1 == temp2) continue;
          
      if (!isGlobalSearch && (isNodeAboveCutoff(*temp1) || isNodeAboveCutoff(*temp2))) {
        std::cout << "\033[31m" "Crossover error: above cutoff depth when in local search. Retrying... " << std::endl << "\033[0m";
        continue;
      }

      GPNodeStruct* temp1Copy = new GPNodeStruct(*temp1);
      GPNodeStruct* temp2Copy = new GPNodeStruct(*temp2);
          
      // Replace the subtrees in the respective parents
      for (int i = 0; i < tempParent1->children.size(); i++) {
        if (tempParent1->children[i] == temp1) {
          delete tempParent1->children[i];
          tempParent1->children[i] = temp2Copy;
          break;
        }
      }
          
      for (int i = 0; i < tempParent2->children.size(); i++) {
        if (tempParent2->children[i] == temp2) {
          delete tempParent2->children[i];
          tempParent2->children[i] = temp1Copy;
          break;
        }
      }
          
      return; // successful crossover
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
    // std::cout << "Individual " << i+1 << "/" << populationSize << " [BACC: " << std::to_string(tF) << "]" <<  std::endl;
  }
  return currBestTree;
}

double GPStruct::fitness(const GPNodeStruct& tree, const std::string& set) {
  double BACC = 0.0;
  double threshold = 0.0;
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
    // if (recal) {
    treeFitness = tree.fitness(dataset[i], colNames);
    // } else {
      // perform a simple lookup to get the value of the tree (saves a lot of time)
      // treeFitness = currPopFitness[getIndex(tree)];
    // }

    if (std::isnan(treeFitness) || std::isinf(treeFitness)) {
      std::cerr << "Invalid fitness value" << std::endl;
      return 0.0; // bad BACC-score
    }

    double actual = dataset[i][dataset[0].size() - 1];

    // clip with a function to avoid overflow, we want the output to be normalised as well...
    treeFitness = tanh(treeFitness); // * normalize to [-1, 1], intersecting at 0.0

    // use threshold to determine if the prediction is correct. If the tree returns 0.0 (false), then it will remain false with the threshold
    if (treeFitness > threshold) {
      treeFitness = 1.0;
    } else {
      treeFitness = 0.0;
    }

    // determine the confusion matrix
    if (treeFitness == 1.0 && actual == 1.0) {
      confusionMatrix[0][0] += 1.0; // TP
    } else if (treeFitness == 1.0 && actual == 0.0) {
      confusionMatrix[0][1] += 1.0; // FP
    } else if (treeFitness == 0.0 && actual == 1.0) {
      confusionMatrix[1][0] += 1.0; // FN
    } else if (treeFitness == 0.0 && actual == 0.0) {
      confusionMatrix[1][1] += 1.0; // TN
    }
  }

  if (confusionMatrix[1][1] + confusionMatrix[0][1] == 0 || confusionMatrix[0][0] + confusionMatrix[1][0] == 0) {
    // std::cerr << "Invalid confusion matrix" << std::endl;
    return 0.0; // avoid division by zero
  }

  // calculate BACC
  double specificity = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]); // TN / (TN + FP)
  double sensitivity = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]); // TP / (TP + FN)
  double bacc = (sensitivity + specificity) / 2.0;
  return bacc; // * maximize fitness
}

double GPStruct::populationFitness() {
  double totalFitness = 0.0;  
  for (int i = 0; i < populationSize; i++) {
    totalFitness += fitness(*population[i], "train");
  }
  return totalFitness / static_cast<double>(populationSize);
}

// misc
GPNodeStruct* GPStruct::getIndividual(const int& index) {
  return population[index];
}

int GPStruct::getIndex(const GPNodeStruct& tree) {
  const GPNodeStruct* treePtr = &tree;
  for (int i = 0; i < populationSize; i++) {
    if (population[i] == treePtr) {
      return i;
    }
  }
  return -1;
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

void GPStruct::printTree(const GPNodeStruct* root, const GPNodeStruct* origin, int depth) {
  if (root == nullptr) {
      return;
  }
  
  // Create indentation based on depth
  std::string tabs = "";
  for (int i = 0; i < depth; i++) {
      tabs += "\t";
  }
  
  // Print the node value with appropriate color
  if (root->isLeaf) {
      // Terminal nodes
      if (isBooleanTerminal(root->value)) {
          // Boolean terminals in red
          std::cout << "\033[31m" << tabs << root->value << "\033[0m" << std::endl;
      } else {
          // Other terminals in yellow
          std::cout << "\033[33m" << tabs << root->value << "\033[0m" << std::endl;
      }
  } else {
      // Operator nodes (no color)
      std::cout << tabs << root->value << std::endl;
  }
  
  // Recursively print all children
  for (const GPNodeStruct* child : root->children) {
      printTree(child, origin, depth + 1);
  }
}

void GPStruct::printTree(const GPNodeStruct& root, int depth) {
  printTree(&root, nullptr, depth);
}

void GPStruct::printPopulation() {
  std::cout << "\033[36m" "Population:" << std::endl << "\033[0m";
  for (int i = 0; i < populationSize; i++) {
    std::cout << "\033[36m" "Individual " << i+1 << "/" << populationSize << " [GI : " << globalIndex(population[i]) << "; LI : " << localIndex(population[i]) << "]" << std::endl << "\033[0m";
    printTree(*population[i]);
  }
}