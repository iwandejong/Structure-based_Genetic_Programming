#ifndef GP_NODE_H
#define GP_NODE_H

#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

class GPNode {
  public:
    std::string value;
    bool isLeaf = false;
    GPNode* left;
    GPNode* right;
    GPNode() {};
    ~GPNode();
    GPNode(const GPNode& other);
    
    double fitness(const std::vector<double>& inputs, const std::vector<std::string>& colNames) const;
    
    GPNode* traverseToNth(int& n) const;
    GPNode* findParent(int& n) const;
    GPNode* findParent(GPNode* child) const;
    GPNode* findParentHelper(int& n, GPNode* parent) const;
    int treeSize() const;
    int calcDepth() const;
    double protectedDiv(const double& a, const double& b) const;

    void print(const std::string& prefix = "", bool isLeft = true);
    std::string formula();
};

#endif // GP_NODE_H
