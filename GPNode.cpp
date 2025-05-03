#include "GPNode.h"

GPNode::~GPNode() {
  if (left != nullptr) {
    delete left;
    left = nullptr;
  }

  if (right != nullptr) {
    delete right;
    right = nullptr;
  }
}

GPNode::GPNode(const GPNode& other) {
  value = other.value;
  isLeaf = other.isLeaf;
  left = (other.left != nullptr) ? new GPNode(*other.left) : nullptr;
  right = (other.right != nullptr) ? new GPNode(*other.right) : nullptr;
}

void GPNode::print(const std::string& prefix, bool isLeft) {
  if (right != NULL) {
    right->print(prefix + (isLeft ? "│   " : "    "), false);
  }

  std::cout << prefix + (isLeft ? "└── " : "┌── ") + value << std::endl;

  if (left != NULL) {
    left->print(prefix + (isLeft ? "    " : "│   "), true);
  }
}

std::string GPNode::formula() {
  if (!left && !right) {
      return value; // Return value if it's a leaf node
  }

  std::string leftFormula = (left) ? left->formula() : "";
  std::string rightFormula = (right) ? right->formula() : "";

  return "(" + leftFormula + " " + value + " " + rightFormula + ")";
}

double GPNode::fitness(const std::vector<double>& inputs, const std::vector<std::string>& colNames) const {
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
GPNode* GPNode::traverseToNth(int& n) const {
  if (n == 0) {
    return const_cast<GPNode*>(this);
  }
  n--;

  if (left) {
    GPNode* leftResult = left->traverseToNth(n);
    if (leftResult) return leftResult;
  }

  if (right) {
    GPNode* rightResult = right->traverseToNth(n);
    if (rightResult) return rightResult;
  }

  return nullptr;
}

GPNode* GPNode::findParent(int& n) const {
  if (n == 0) {
    return nullptr; // No parent for the root node
  }
  n--;

  GPNode* parent = findParentHelper(n, nullptr);
  return parent;
}

GPNode* GPNode::findParent(GPNode* child) const {
  if (child == nullptr) {
    return nullptr; // No parent for null child
  }

  if (left == child || right == child) {
    return const_cast<GPNode*>(this); // Return this node as the parent
  }
  GPNode* parent = nullptr;
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

GPNode* GPNode::findParentHelper(int& n, GPNode* parent) const {
  if (n == 0) {
    return parent;
  }
  n--;

  if (left) {
    GPNode* leftResult = left->findParentHelper(n, const_cast<GPNode*>(this));
    if (leftResult) return leftResult;
  }

  if (right) {
    GPNode* rightResult = right->findParentHelper(n, const_cast<GPNode*>(this));
    if (rightResult) return rightResult;
  }

  return nullptr;
}

int GPNode::treeSize() const {
  if (left) return 1 + left->treeSize();
  if (right) return 1 + right->treeSize();
  return 1;
}

int GPNode::calcDepth() const {
  if (isLeaf) {
      return 0;
  }

  int leftDepth = (left) ? left->calcDepth() : 0;
  int rightDepth = (right) ? right->calcDepth() : 0;

  return 1 + std::max(leftDepth, rightDepth);
}

double GPNode::protectedDiv(const double& a, const double& b) const {
  return (std::abs(b) < 1e-6) ? a : a / b;
}