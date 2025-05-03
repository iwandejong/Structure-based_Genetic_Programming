#include "GPNodeStruct.h"

GPNodeStruct::~GPNodeStruct() {
  if (left != nullptr) {
    delete left;
    left = nullptr;
  }

  if (right != nullptr) {
    delete right;
    right = nullptr;
  }
}

GPNodeStruct::GPNodeStruct(const GPNodeStruct& other) {
  value = other.value;
  isLeaf = other.isLeaf;
  left = (other.left != nullptr) ? new GPNodeStruct(*other.left) : nullptr;
  right = (other.right != nullptr) ? new GPNodeStruct(*other.right) : nullptr;
}

void GPNodeStruct::print(const std::string& prefix, bool isLeft) {
  if (right != NULL) {
    right->print(prefix + (isLeft ? "│   " : "    "), false);
  }

  std::cout << prefix + (isLeft ? "└── " : "┌── ") + value << std::endl;

  if (left != NULL) {
    left->print(prefix + (isLeft ? "    " : "│   "), true);
  }
}

std::string GPNodeStruct::formula() {
  if (!left && !right) {
      return value; // Return value if it's a leaf node
  }

  std::string leftFormula = (left) ? left->formula() : "";
  std::string rightFormula = (right) ? right->formula() : "";

  return "(" + leftFormula + " " + value + " " + rightFormula + ")";
}

double GPNodeStruct::fitness(const std::vector<double>& inputs, const std::vector<std::string>& colNames) const {
  if (isLeaf) {
    try {
      return std::stod(value);
    } catch (const std::invalid_argument&) {
      for (int i = 0; i < colNames.size(); i++) {
        if (colNames[i] == value) {
          return inputs[i];
        }
      }
    }
    return INFINITY; // can't be calculated, therefore a bad fitness
  }
  
  double leftValue = left->fitness(inputs, colNames);
  if (right) {
    double rightValue = right->fitness(inputs, colNames);
    
    if (value == "+") {
      // return leftValue + rightValue;
      return std::max(-1.0, std::min(1.0, leftValue + rightValue));
    } else if (value == "-") {
      // return leftValue - rightValue;
      return std::max(-1.0, std::min(1.0, leftValue - rightValue));
    } else if (value == "*") {
      // return leftValue * rightValue;
      return std::max(-1.0, std::min(1.0, leftValue * rightValue));
    }else if (value == "/") {
      // return protectedDiv(leftValue, rightValue);
      return std::max(-1.0, std::min(1.0, protectedDiv(leftValue, rightValue)));
    } else if (value == "max") {
      return std::max(leftValue, rightValue);
    } else if (value == "min") {
      return std::min(leftValue, rightValue);
    }
    return INFINITY; // can't be calculated, therefore a bad fitness
  } else {
    if (value == "exp") {
      return std::max(-1.0, std::min(1.0, exp(leftValue)));
    } else if (value == "sin") {
      return sin(leftValue);
    } else if (value == "cos") {
      return cos(leftValue);
    } else if (value == "log") {
      if (leftValue <= -1.0) {
        return -1.0; // log(<=-1.0) is undefined since -1.0+1.0=0.0, return a large negative value
      }
      return 0.5 * log(leftValue + 1.0) + 0.5;
    } else if (value == "sigmoid") {
      return 1.0 / (1.0 + exp(-leftValue));
    }
    return INFINITY; // can't be calculated, therefore a bad fitness
  }
}

// DFS
GPNodeStruct* GPNodeStruct::traverseToNth(int& n) const {
  if (n == 0) {
    return const_cast<GPNodeStruct*>(this);
  }
  n--;

  if (left) {
    GPNodeStruct* leftResult = left->traverseToNth(n);
    if (leftResult) return leftResult;
  }

  if (right) {
    GPNodeStruct* rightResult = right->traverseToNth(n);
    if (rightResult) return rightResult;
  }

  return nullptr;
}

GPNodeStruct* GPNodeStruct::findParent(int& n) const {
  if (n == 0) {
    return nullptr; // No parent for the root node
  }
  n--;

  GPNodeStruct* parent = findParentHelper(n, nullptr);
  return parent;
}

GPNodeStruct* GPNodeStruct::findParent(GPNodeStruct* child) const {
  if (child == nullptr) {
    return nullptr; // No parent for null child
  }

  if (left == child || right == child) {
    return const_cast<GPNodeStruct*>(this); // Return this node as the parent
  }
  GPNodeStruct* parent = nullptr;
  if (left) {
    parent = left->findParent(child);
    if (parent) return parent;
  }
  if (right) {
    parent = right->findParent(child);
    if (parent) return parent;
  }
  return nullptr; // No parent found
}

GPNodeStruct* GPNodeStruct::findParentHelper(int& n, GPNodeStruct* parent) const {
  if (n == 0) {
    return parent;
  }
  n--;

  if (left) {
    GPNodeStruct* leftResult = left->findParentHelper(n, const_cast<GPNodeStruct*>(this));
    if (leftResult) return leftResult;
  }

  if (right) {
    GPNodeStruct* rightResult = right->findParentHelper(n, const_cast<GPNodeStruct*>(this));
    if (rightResult) return rightResult;
  }

  return nullptr;
}

int GPNodeStruct::treeSize() const {
  if (left) return 1 + left->treeSize();
  if (right) return 1 + right->treeSize();
  return 1;
}

int GPNodeStruct::calcDepth() const {
  if (isLeaf) {
      return 0;
  }

  int leftDepth = (left) ? left->calcDepth() : 0;
  int rightDepth = (right) ? right->calcDepth() : 0;

  return 1 + std::max(leftDepth, rightDepth);
}

double GPNodeStruct::protectedDiv(const double& a, const double& b) const {
  return (std::abs(b) < 1e-6) ? a : a / b;
}