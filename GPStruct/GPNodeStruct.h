#ifndef GP_NODE_STRUCT_H
#define GP_NODE_STRUCT_H

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

class GPNodeStruct {
  public:
    std::string value;
    bool isLeaf = false;
    GPNodeStruct* left;
    GPNodeStruct* right;
    GPNodeStruct() {};
    ~GPNodeStruct();
    GPNodeStruct(const GPNodeStruct& other);
    
    double fitness(const std::vector<double>& inputs, const std::vector<std::string>& colNames) const;
    
    GPNodeStruct* traverseToNth(int& n) const;
    GPNodeStruct* findParent(int& n) const;
    GPNodeStruct* findParent(GPNodeStruct* child) const;
    GPNodeStruct* findParentHelper(int& n, GPNodeStruct* parent) const;
    int treeSize() const;
    int calcDepth() const;
    double protectedDiv(const double& a, const double& b) const;

    void print(const std::string& prefix = "", bool isLeft = true);
    std::string formula();
};

#endif // GP_NODE_STRUCT_H
