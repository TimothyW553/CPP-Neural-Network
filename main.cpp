#include <bits/stdc++.h>

using namespace std;

const string path = R"(C:\Users\Timothy Wang\Documents\Neural-Network\)";
const string training_images = "MNIST_train.txt";

ifstream file{path + training_images};

const int samples = 60'000;

const int width = 28;
const int height = 28;

const int input_neurons = width * height;
const int hidden_neurons_layer_1 = 128;
const int output_neurons = 10;

const int epochs = 512;
const double learning_rate = 1e-3;
const double epsilon = 1e-3;
const double momentum = 0.9;

mt19937_64 g(chrono::steady_clock::now().time_since_epoch().count());
double randf(double l, double r){return uniform_real_distribution<double>(l, r)(g);}

vector<vector<double>> image(width + 1, vector<double>(height + 1));
vector<double> input;
vector<double> input_layer(input_neurons + 1);
vector<double> expected(output_neurons + 1);

/*
 * Adjacency matrix that describes the edges between nodes
 */
vector<vector<double>> w1(input_neurons + 1, vector<double>(hidden_neurons_layer_1 + 1));
vector<vector<double>> w2(hidden_neurons_layer_1 + 1, vector<double>(output_neurons + 1));

/*
 * Activation values of each node for each layer
 */
vector<double> hidden_layer_1(hidden_neurons_layer_1 + 1);
vector<double> output_layer(output_neurons + 1);

/*
 * These vectors holds the error values for the hidden layer neurons.
 * These error values are calculated during backpropagation and are used to
 * update the weights between the input layer and the hidden layer.
 * (or hidden and output)
 */
vector<double> delta_hidden_1(hidden_neurons_layer_1 + 1);
vector<double> delta_output(output_neurons + 1);

/*
 * These vectors holds the previous weight update values for the weights between
 * the input layer and the hidden layer. These values are used in the momentum
 * term of the weight update during backpropagation.
 */
vector<double> prev_delta_w1(input_neurons + 1, 0.0);
vector<double> prev_delta_w2(hidden_neurons_layer_1 + 1, 0.0);

/*
 * Initialize the weight arrays will random values at first
 */
void init_array(bool is_random) {
    ifstream weights;
    if (!is_random) {
        weights.open("model-neural-network.dat");
    }
    for (size_t i = 1; i <= input_neurons; ++i) {
        for (size_t j = 1; j <= hidden_neurons_layer_1; ++j) {
            if (is_random) w1[i][j] = randf(-.5, .5);
            else weights >> w1[i][j];
        }
    }
    for (size_t i = 1; i <= hidden_neurons_layer_1; ++i) {
        for (size_t j = 1; j <= output_neurons; ++j) {
            if (is_random) w2[i][j] = randf(-.5, .5);
            else weights >> w2[i][j];
        }
    }
}

void save_weights() {
    ofstream weights{"model-neural-network.dat"};
    for (size_t i = 1; i <= input_neurons; ++i) {
        for (size_t j = 1; j <= hidden_neurons_layer_1; ++j) {
            weights << w1[i][j] << ' ';
        }
        weights << '\n';
    }
    for (size_t i = 1; i <= hidden_neurons_layer_1; ++i) {
        for (size_t j = 1; j <= output_neurons; ++j) {
            weights << w2[i][j] << ' ';
        }
        weights << '\n';
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

/*
 * Computes the output of the neural network for the given input
 * using forward propagation.
 */
void forward_propagation() {
    // Hidden layer 1
    for (int i = 1; i <= hidden_neurons_layer_1; ++i) {
        double sum = 0.0;
        for (int j = 1; j <= input_neurons; ++j) {
            sum += input[j] * w1[j][i];
        }
        hidden_layer_1[i] = sigmoid(sum);
    }

    // Output layer
    for (int i = 1; i <= output_neurons; ++i) {
        double sum = 0.0;
        for (int j = 1; j <= hidden_neurons_layer_1; ++j) {
            sum += hidden_layer_1[j] * w2[j][i];
        }
        output_layer[i] = sigmoid(sum);
    }
}

/*
 * Computes the error terms of the output layer neurons and updates the weights
 * between the hidden layer and the output layer.
 */
void backpropagation_output_layer() {
    for (int i = 1; i <= output_neurons; ++i) {
        delta_output[i] = output_layer[i] * (1 - output_layer[i]) * (expected[i] - output_layer[i]);
    }

    for (int i = 1; i <= hidden_neurons_layer_1; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            double delta_w = learning_rate * delta_output[j] * hidden_layer_1[i] + momentum * prev_delta_w2[i];
            w2[i][j] += delta_w;
            prev_delta_w2[i] = delta_w;
        }
    }
}

/*
 * Computes the error terms of the hidden layer neurons and updates the weights
 * between the input layer and the hidden layer.
 */
void backpropagation_hidden_layer() {
    for (int i = 1; i <= hidden_neurons_layer_1; ++i) {
        double sum = 0;
        for (int j = 1; j <= output_neurons; ++j) {
            sum += delta_output[j] * w2[i][j];
        }
        delta_hidden_1[i] = hidden_layer_1[i] * (1 - hidden_layer_1[i]) * sum;
    }

    for (int i = 0; i <= input_neurons; ++i) {
        for (int j = 1; j <= hidden_neurons_layer_1; ++j) {
            double delta_w = learning_rate * delta_hidden_1[j] * input_layer[i] + momentum * prev_delta_w1[j];
            w1[i][j] += delta_w;
            prev_delta_w1[j] = delta_w;
        }
    }
}

void read_mnist() {
    if (!file.is_open()) return;

    input.clear();

    string line;
    getline(file, line);

    istringstream ss{line};
    string token;
    int answer;
    bool is_answer = true;
    while (getline(ss, token, ',')) {
        if (is_answer) {
            answer = stoi(token);
            is_answer = false;
            continue;
        }
        input.push_back(stod(token));
    }

    assert(input.size() == input_neurons);

    int index = 0;
    for (int i = 1; i <= height; ++i) {
        for (int j = 1; j <= width; ++j) {
            image[i][j] = input[index++];
            cout << (image[i][j] == 0 ? 0 : 1);
            int pos = i + (j - 1) * width;
        }
        cout << '\n';
    }
    fill(expected.begin(), expected.end(), 0.0);
    expected[answer + 1] = 1.0;
    cerr << "Expecting " << answer << '\n';
}

void train_network() {
    init_array(false);

    for (int e = 1; e <= epochs; ++e) {

        if (file.is_open()) file.close();
        file.open(path + training_images);

        double error = 0.0;

        for (int s = 1; s <= samples; ++s) {
            read_mnist();

            forward_propagation();

            // Compute network's prediction
            int prediction = 0;
            double max_output = 0.0;
            for (int i = 1; i <= output_neurons; ++i) {
                if (output_layer[i] > max_output) {
                    max_output = output_layer[i];
                    prediction = i - 1;
                }
            }
            cerr << "Sample " << s << ": Network prediction = " << prediction << endl;

            // Compute the error of the output layer
            for (int i = 1; i <= output_neurons; ++i) {
                double err = expected[i] - output_layer[i];
                delta_output[i] = err * sigmoid_derivative(output_layer[i]);
                error += 0.5 * err * err;
            }

            backpropagation_output_layer();
            backpropagation_hidden_layer();

            if (s % 1000 == 0) {
                save_weights();
            }
        }

        // Print out the error every 10 epochs
        if (e % 10 == 0) {
            cout << "Epoch " << e << " error: " << error << endl;
        }
    }
}

int main() {
    freopen("output.txt", "w", stdout);
    train_network();
}
