#include "word2vec.h"
#include "parser.h"

// Main function
int main(int argc, char *argv[]) {
    // Read the data from data parser
    CorpusParser<std::string> *c = new CorpusParser<std::string>();
    
    std::vector<int> X;
    std::vector<std::vector<double>> Y;
    int vocab_size;

    tie(vocab_size, X, Y) = c->get_train_x_and_y();

    int embedding_size = 50, epochs = 5000, batch_size = 128;
    double learning_rate = 0.05;

    // Train the model
    Word2Vec *word2vec = new Word2Vec(vocab_size, embedding_size);
    word2vec->skipgram_training(X, Y, learning_rate, epochs, batch_size);

    // Predict using the trained model
    std::vector<std::vector<double>> word2vec_embeddings = word2vec->forward(X);
    
    return 0;
}
