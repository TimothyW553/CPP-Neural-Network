#include "neuron.h"
#include "connection.h"

double Neuron::sigmoid( double x ) {
    return 1 / (1 + exp(-x));
}

double Neuron::sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double Neuron::random( void ) {
    return rand() / double(RAND_MAX);
}

// sumDOW determines the neuron error for the hidden layer
double Neuron::sumDOW( const Layer & next_layer ) const {
    double sum = 0.0;
    for (size_t i = 0; i < next_layer.size() - 1; ++i) {
        sum += output_weights[i].weight * next_layer[i].gradient;
    }
    return sum;
}

Neuron::Neuron( uint32 output_amount, uint32 index ) {
    this->index = index;
    output_weights.resize(output_amount);

    for (size_t i = 0; i < output_amount; ++i) {
        output_weights.push_back(Connection());
        output_weights.back().weight = random();
    }
}

void Neuron::set_output( double value ) {
    output = value;
}

double Neuron::get_output( void ) const {
    return output;
}

std::vector<Connection> Neuron::get_output( void ) const {
    return output_weights;
}

void Neuron::feed_forward( const Layer & prev_layer ) {
    double sum = 0.0;
    for (size_t i = 0; i < prev_layer.size(); ++i) {
        sum += prev_layer[i].get_output() * prev_layer[i].output_weights[index].weight;
    }
    output = Neuron::sigmoid(sum);
}

/*
 * Calculates the gradient for the output layer
 * In other words, the amount we need to change the weights for this connection
 * The larger the delta, the larger the gradient (need for change)
 * The smaller the delta, the smaller the gradient (less change is needed)
 *
 * Sigmoid derivative will tell us the sign of which direction to go
 */
void Neuron::calculate_output_gradients( double target ) {
    double delta = target - output;
    gradient = delta * Neuron::sigmoid_derivative(output);
}

/*
 * Calculates the gradient for the hidden layers
 * Similar to function above
 */
void Neuron::calculate_hidden_gradients( const Layer & next_layer ) {
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::sigmoid_derivative(output);
}

void Neuron::update_weights( Layer & prev_layer ) {
    for (size_t i = 0; i < prev_layer.size(); ++i) {
        double old_delta_weight = prev_layer[i].output_weights[index].delta_weight;

        double new_delta_weight = learning_rate * prev_layer[i].get_output() * gradient + alpha * old_delta_weight;

        prev_layer[i].output_weights[index].delta_weight = new_delta_weight;
        prev_layer[i].output_weights[index].weight += new_delta_weight;
    }
}
