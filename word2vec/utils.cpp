#include "utils.h"
#include <iostream>

int main() {
    CorpusParser<std::string> *c = new CorpusParser<std::string>();
    
    std::vector<int> X;
    std::vector<std::vector<int>> Y;
    
    tie(X, Y) = c->get_train_x_and_y();

    std::cout << X.size() << std::endl;

    for(int i=0; i<X.size(); i++) std::cout << X[i] << " ";
    std::cout << std::endl;

    return 0;
}
