#ifndef GP_NODE_STRUCT_H
#define GP_NODE_STRUCT_H

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

class GPNodeStruct {
  public:
    std::string value;
    bool isLeaf = false;
    std::vector<GPNodeStruct*> children = {};
    GPNodeStruct() {};
    ~GPNodeStruct() {};
    GPNodeStruct(const GPNodeStruct& other);
    
    double fitness(const std::vector<double>& inputs, const std::vector<std::string>& colNames) const;
    
    GPNodeStruct* traverseToNth(int& n) const;
    GPNodeStruct* findParent(int& n) const;
    GPNodeStruct* findParent(GPNodeStruct* child) const;
    GPNodeStruct* findParentHelper(int& n, GPNodeStruct* parent) const;
    int treeSize() const;
    int calcDepth() const;
    double protectedDiv(const double& a, const double& b) const;
};

#endif // GP_NODE_STRUCT_H
