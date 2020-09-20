#include <fstream>
#include <vector>
#include <unordered_set>
#include <set>
#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include <algorithm>
#include <iterator>

#define WORD_CORPUS_FILE_LOC "/Users/ruppala/Documents/personal/word_embeddings/data/word_corpus.txt"
#define WINDOW_SIZE 3

template<class T>
class CorpusParser {
    public:
        std::vector<T> tokenize(std::vector<T> raw_words) {
            std::vector<T> tokens;
            for(T word: raw_words) {
                bool has_letter_in_word = false;
                for(int i=0; i<word.size(); i++) {
                    if(isalpha(word.at(i))) has_letter_in_word = true;
                }
                if(has_letter_in_word) tokens.push_back(word);
            }
            return tokens;
        }

        std::tuple<std::map<T, int>, std::map<int, T>> mapping(std::vector<T> tokens) { 
            std::set<T> unique_tokens(tokens.begin(), tokens.end());
            int index = 0;
            std::set<std::string>::iterator it = unique_tokens.begin();
            std::map<T, int> word_to_id;
            std::map<int, T> id_to_word;
            while(it != unique_tokens.end()) {
                word_to_id[*it] = index;
                id_to_word[index] = *it;
                index++;
                it++;
            }
            return { word_to_id, id_to_word };
        }

        std::tuple<std::vector<int>, std::vector<int>> generate_training_data(std::vector<T> tokens, std::map<T, int> word_to_id, int window_size) {
            int N = tokens.size();
            std::vector<int> X, Y;
            for(int i=0; i<N; i++) {
                for(int j = std::max(i - window_size, 0); j < std::min(N -1, i + window_size); j++) {
                    X.push_back(word_to_id[tokens[i]]);
                    Y.push_back(word_to_id[tokens[j]]);
                }
            }
            return { X, Y };
        }

        void print_vec(std::vector<int> v) {
            for(auto v1: v) {
                std::cout << v1 << " ";
            }
            std::cout << std::endl;
        }

        std::tuple<int, std::vector<int>, std::vector<std::vector<double>>> get_train_x_and_y() {
            std::ifstream train_file;
            train_file.open(WORD_CORPUS_FILE_LOC);
            if(!train_file.is_open()) {
                throw std::runtime_error("Could not open training file");
            }

            std::vector<T> raw_words;
            std::string word;
            while(train_file >> word) { raw_words.push_back(word); }

            std::vector<std::string> tokens = tokenize(raw_words);   
            
            std::map<std::string, int> word_to_id;
            std::map<int, std::string> id_to_word;
            tie(word_to_id, id_to_word) = mapping(tokens);

            std::vector<int> X, Y;
            tie(X, Y) = generate_training_data(tokens, word_to_id, WINDOW_SIZE);

            int n = id_to_word.size(), m = Y.size();
            std::vector<std::vector<double> > y_one_hot(n, std::vector<double>(m, 0.0));
                
            for(int j=0; j<m; j++) y_one_hot[Y[j]][j] = 1.0;

            return {n, X, y_one_hot };
        }
};



