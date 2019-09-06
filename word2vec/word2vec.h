#include "utils.h"
#include <cmath>
#include <numeric>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>

typedef std::vector<double> VS;
typedef std::vector<std::vector<double>> VD;
typedef std::map<std::string, std::vector<std::vector<double>>> MP;
typedef std::pair<std::string, std::vector<std::vector<double>>> PAIR;

class Word2Vec {
    private:
        VD word_vec_layer, dense_layer;
        MP cache, gradients;

    public:
        Word2Vec(int vocab_size, int embedding_size) {
            word_vec_layer.resize(vocab_size, VS(embedding_size, 0.01));
            dense_layer.resize(vocab_size, VS(embedding_size, 0.01));
        }

        template <class T>
        T get_map_key(MP map_obj, std::string key) {
            auto itr = map_obj.find(key);
            T result;
            if(itr != map_obj.end()) result = itr->second;
            return result;
        }

        template <typename T>
        std::vector<std::vector<T>> slice(std::vector<std::vector<T>> input, std::vector<int> indices) {
            std::vector<std::vector<T>> result;
            for(int i=0; i<indices.size(); i++) result.push_back(input[indices[i]]);
            return result;
        }

        // Softmax
        VD softmax(VD inp) {
            VD softmax_out(inp.size(), VS(inp[0].size(), 0.0));
            for(int i=0; i<inp.size(); i++) {
                std::vector<double>::iterator max_x = std::max_element(inp[i].begin(), inp[i].end());
                double total_sum = 0.0;
                for(auto value: inp[i]) total_sum += exp(value) - *max_x;
                for(int j=0; j<inp[i].size(); j++) {
                    softmax_out[i][j] = exp(inp[i][j] - *max_x) * 1.0 / total_sum;
                }
            }
            return softmax_out;
        }

        // Forward propagation
        VD forward(std::vector<int> X) {
            // K * M ( where K = last_index - first_index, M = embedding size )
            VD word_vec = slice(word_vec_layer, X);
            cache.insert(PAIR("word_vec", word_vec));

            // N * K ( where N = size of window )
            VD z = matmul<double>(dense_layer, transpose<double>(word_vec));
            cache.insert(PAIR("z", z));
            // N * K
            VD softmax_out = softmax(z);
            cache.insert(PAIR("softmax_out", softmax_out));

            return softmax_out;
        }

        // Softmax backward
        VD softmax_backward(VD softmax_out, VD Y) {
            VD result(softmax_out.size(), VS(softmax_out[0].size(), 0.0));
            for(int i=0; i<softmax_out.size(); i++) {
                for(int j=0; j<softmax_out[i].size(); j++) {
                    result[i][j] = softmax_out[i][j] - Y[i][j];
                }
            }
            return result;
        }
        
        // dense layer backward
        std::tuple<VD, VD> dense_backward(VD dl_dz) {
            VD dl_dense, dl_word_vec;
            VD word_vec = get_map_key<VD>(cache, "word_vec");

            // Let's try this without 1 / m 
            dl_dense = matmul<double>(dl_dz, word_vec);

            dl_word_vec = matmul<double>(transpose<double>(dense_layer), dl_dz);

            return { dl_dense, dl_word_vec };
        }

        // Backward Propogation
        void backward(VD Y_batch) {
            VD dl_dz = softmax_backward(get_map_key<VD>(cache, "softmax_out"), Y_batch);

            VD dl_dense;
            VD dl_word_vec;
            tie(dl_dense, dl_word_vec) = dense_backward(dl_dz);

            gradients.insert(PAIR("dl_dz", dl_dz));
            gradients.insert(PAIR("dl_dense", dl_dense));
            gradients.insert(PAIR("dl_word_vec", dl_word_vec));
        }

        VS subtract_single(VS A, VS B) {
            VS result;
            for(int i=0; i<A.size(); i++) result.push_back(A[i] - B[i]);
            return result;
        }

        void word_vec_layer_gradient(std::vector<int> X, VD dl_word_vec, double learning_rate) {
            for(int word_index=0; word_index<X.size(); word_index++) {
                VS result;
                for(double value: dl_word_vec[X[word_index]]) result.push_back(value * learning_rate);
                word_vec_layer[X[word_index]] = subtract_single(word_vec_layer[X[word_index]], result);
            }
        }

        // Updating the parameters
        void update_parameters(std::vector<int> X, double learning_rate) {
            // Update the word embedding layer
            VD dl_word_vec = get_map_key<VD>(gradients, "dl_word_vec");
            word_vec_layer_gradient(X, transpose<double>(dl_word_vec), learning_rate);
            // Update the dense layer
            VD dl_dense = get_map_key<VD>(gradients, "dl_dense");
            dense_layer = subtract<double>(dense_layer, multiply_matrix_with_scalar<double>(dl_dense, learning_rate));
        }

        // Cross Entropy
        double cross_entropy(VD softmax_out, VD y) {
            double cost = 0.0;
            for(int i=0; i<softmax_out.size(); i++) {
                for(int j=0; j<softmax_out[i].size(); j++) {
                    cost += y[i][j] * log(softmax_out[i][j] + 0.0001);
                }
            }
            return - (1.0 / softmax_out[0].size()) * cost;
        }

        // training
        void skipgram_training(std::vector<int> X,  VD Y, double learning_rate, int epochs, int batch_size) {
            for(int epoch = 0; epoch < epochs; epoch++) {
                double epoch_cost = 0.0;
                int index = 0;
                while(index < X.size()) {
                    int first_index = index, last_index = std::min((int) X.size(), index + batch_size);
                    std::vector<int> X_batch(&X[first_index], &X[last_index]);
                    // Forward propagation
                    VD softmax_out = forward(X_batch);
                    // Backward
                    VD Y_batch = transpose<double>(slice(transpose<double>(Y), X_batch));
                    backward(Y_batch);
                    // Updating the parameters
                    update_parameters(X_batch, learning_rate);
                    // Cross Entropy calculation
                    epoch_cost += cross_entropy(softmax_out, Y_batch);
                    index = index + batch_size;
                }
                if(epoch % 100 == 0) {
                    std::cout << "loss after " << epoch << " epochs: " << epoch_cost << std::endl;
                }
            }
        }
};
