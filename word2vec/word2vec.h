#include "utils.h"
#include <cmath>
#include <numeric>
#define EMBEDDING_SIZE 10

class Word2Vec {
    private:
        std::vector<std::vector<double>> word_vec_layer, dense_layer;
        std::map<std::string, std::vector<std::vector<double>> cache, gradients;

    public:
        Word2Vec(int vocab_size, int embedding_size) {
            word_vec_layer.resize(vocab_size, std::vector<double>(embedding_size, 0.01));
            dense_layer.resize(vocab_size, std::vector<double>(embedding_size, 0.01));
        }

        // Softmax
        std::vector<std::vector<double>> softmax(std::vector<std::vector<double>> inp) {
            std::vector<std::vector<double>> sotmax_out(inp.size(), std::vector<double>(inp[0].size(), 0.0));
            for(int i=0; i<inp.size(); i++) {
                double *max_x = std::max_element(inp[i].begin(), inp[i].end());
                double total_sum = std::accumulate(inp[i].begin(). inp[i].end(), 0.0);
                for(int j=0; j<inp[i].size(); j++) {
                    softmax_out[i][j] = exp(inp[i][j] - *max_x) * 1.0 / total_sum;
                }
            }
            return softmax_out;
        }

        // Forward propogation
        void forward(int first_index, int last_index) {
            // K * M ( where K = last_index - first_index, M = embedding size )
            std::vector<std::vector<double>> word_vec(&word_vec_layer[first_index], &word_vec_layer[last_index]); 
    
            // N * K ( where N = size of window )
            std::vector<std::vector<double>> z = matmul<double>(dense_layer, transpose<double>(word_vec));
            cache["z"] = z;
            
            // N * K
            std::vector<std::vector<double>> softmax_out = softmax(z);
            cache["softmax_out"] = softmax_out;    
        }
 



        // Softmax backward
        std::vector<std::vector<double>> softmax_backward(std::vector<std::vector<double>> softmax_out, std::vector<std::vector<double>> Y) {
            std::vector<std::vector<double>> result(softmax_out.size(), std::vector<softmax_out[0].size(), 0.0);
            for(int i=0; i<softmax_out.size(); i++) {
                for(int j=0; j<softmax_out[i].size(); j++) {
                    result[i][j] = softmax_out[i][j] - y[i][j];
                }
            }
            return result;
        }
        
        // dense layer backward
        std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> dense_backward(std::vector<std::vector<double>> dl_dz) {
            std::vector<std::vector<double>> dl_dense;
            std::vector<std::vector<double>> dl_word_vec;

            // Let's try this without 1 / m 
            dl_dense = matmul<double>(dl_dz, transpose<double>(word_vec));
            dl_word_vec = matmul<double>(transpose<double>(dense_layer), dl_dz);
            
            assert(shape(dense) == shape(dl_dense));
            assert(shape(word_vec) == shape(dl_word_vec));

            return { dl_dense, dl_word_vec };
        }


        // Backward Propogation
        void backward(std::vector<std::vector<int>> Y_batch) {
            std::vector<std::vector<double>> dl_dz = softmax_backward(cache['softmax_out'], Y_batch);
            
            std::vector<std::vector<double>> dl_dense;
            std::vector<std::vector<double>> dl_word_vec;
            tie(dl_dense, dl_word_vec) = dense_backward(dl_dz);

            gradients["dl_dz"] = dl_dz;
            gradients["dl_dense"] = dl_dense;
            gradients["dl_word_vec"] = dl_word_vec;
        }


        // Updating the parameters
        void update_parameters(int first_index, int last_index, double learning_rate) {
            // Update the word embedding layer
            for(int index = first_index; index < last_index; index++) {
                word_vec_layer[index] = subtract<double>(word_vec_layer[index], multiply_matrix_with_scalar<T>(transpose<double>(gradients["dl_word_vec"][index]), learning_rate));
            }
            // Update the dense layer
            dense_layer = subtract<double>(dense_layer, multiply_matrix_with_scalar<double>(gradients["dl_dense"], learning_rate));
        }

        // Cross Entropy
        double cross_entropy(std::vector<std::vector<double>> softmax_out, std::vector<std::vector<double>> y) {
            double cost = 0.0;
            for(int i=0; i<softmax_out.size(); i++) {
                for(int j=0; j<softmax_out[i].size(); j++) {
                    cost += y[i][j] * log(softmax_out[i][j] + 0.001);
                }
            }
            return - (1.0 / softmax_out.size()) * cost;
        }

        // training
        void skipgram_training(std::vector<int> X,  std::vector<int> Y, double learning_rate, int epochs, int batch_size) {
            for(int epoch = 0; epoch < epochs; epoch++) {
                double epoch_cost = 0.0;
                std::vector<pair<int, int>> batch_indices;
                for(int index = 0; index < X.size(); index = index + batch_size) {
                    int first_index = index, last_index = index + batch_size;

                    // Forward Propogation
                    forward(first_index, last_index);

                    // Backward 
                    std::vector<std::vector<int>> Y_batch(&X[first_index], &X[last_index]);
                    backward(Y_batch);

                    // Updating the parameters
                    update_parameters(first_index, last_index, learning_rate);

                    // Cross Entropy calculation
                    epoch_cost += cross_entropy();
                }
                if(epoch % 100 == 0) {
                    std::cout << "loss after " << epoch << " epochs: " << epoch_cost << std::endl;
                }
            }
        }
}
