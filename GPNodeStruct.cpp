#include "GPNodeStruct.h"

GPNodeStruct::GPNodeStruct(const GPNodeStruct& other) {
  value = other.value;
  isLeaf = other.isLeaf;
  for (const auto& child : other.children) {
    children.push_back(new GPNodeStruct(*child));
  }
}

// booleans will return either 0 or 1
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

  double result = 0.0;
  if (children.size() == 0) {
    return INFINITY; // no children, can't be calculated
  } else if (children.size() == 1) {
    result = children[0]->fitness(inputs, colNames);
  } else if (children.size() == 2) {
    double left = children[0]->fitness(inputs, colNames);
    double right = children[1]->fitness(inputs, colNames);

    if (value == "+") {
      result = left + right;
    } else if (value == "-") {
      result = left - right;
    } else if (value == "*") {
      result = left * right;
    } else if (value == "/") {
      result = protectedDiv(left, right);
    } else if (value == "max") {
      result = std::max(left, right);
    } else if (value == "min") {
      result = std::min(left, right);
    } else if (value == "tanh") {
      // result = 1.0 / (1.0 + std::exp(-left));
      result = std::tanh(left);
    } else if (value == "sin") {
      result = std::sin(left);
    } else if (value == "cos") {
      result = std::cos(left);
    } else if (value == "log") {
      if (left <= 0) {
        return 0.0;
      }
      result = std::log(left);
    } else if (value == "and") {
      result = static_cast<double>(static_cast<bool>(left) && static_cast<bool>(right));
    } else if (value == "or") {
      result = static_cast<double>(static_cast<bool>(left) || static_cast<bool>(right));
    } else if (value == "not") {
      result = static_cast<double>(!static_cast<bool>(left));
    } else if (value == "<") {
      result = static_cast<double>(left < right);
    } else if (value == ">") {
      result = static_cast<double>(left > right);
    } else if (value == "<=") {
      result = static_cast<double>(left <= right);
    } else if (value == ">=") {
      result = static_cast<double>(left >= right);
    } else if (value == "==") {
      result = static_cast<double>(left == right);
    } else if (value == "!=") {
      result = static_cast<double>(left != right);
    }
  }
  return result;
}

// DFS
GPNodeStruct* GPNodeStruct::traverseToNth(int& n) const {
  if (n == 0) {
    return const_cast<GPNodeStruct*>(this);
  }
  n--;

  for (const auto& child : children) {
    GPNodeStruct* result = child->traverseToNth(n);
    if (result) {
      return result;
    }
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

  for (const auto& c : children) {
    if (c == child) {
      return const_cast<GPNodeStruct*>(this); // Return this node as the parent
    }
  }
  GPNodeStruct* parent = nullptr;
  for (const auto& c : children) {
    parent = c->findParent(child);
    if (parent) return parent;
  }
  return nullptr; // No parent found
}

GPNodeStruct* GPNodeStruct::findParentHelper(int& n, GPNodeStruct* parent) const {
  if (n == 0) {
    return parent;
  }
  n--;

  for (const auto& c : children) {
    GPNodeStruct* childResult = c->findParentHelper(n, const_cast<GPNodeStruct*>(this));
    if (childResult) return childResult;
  }

  return nullptr;
}

double GPNodeStruct::protectedDiv(const double& a, const double& b) const {
  return (std::abs(b) < 1e-6) ? a : a / b;
}

int GPNodeStruct::treeSize() const {
  if (children.empty()) {
    return 1;
  }
  int size = 1; // Count this node
  for (const auto& child : children) {
    size += child->treeSize();
  }
  return size;
}

int GPNodeStruct::calcDepth() const {
  if (isLeaf) {
      return 0;
  }

  int depth0 = 0;
  int depth1 = 0;
  int depth2 = 0;

  for (const auto& child : children) {
    if (child->children.size() == 0) {
      depth0 = std::max(depth0, 1);
    } else if (child->children.size() == 1) {
      depth1 = std::max(depth1, child->calcDepth());
    } else if (child->children.size() == 2) {
      depth2 = std::max(depth2, child->calcDepth());
    }
  }

  return 1 + std::max(depth0, std::max(depth1, depth2));
}