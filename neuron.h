#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Connection;
class Neuron;

using Layer = std::vector<Neuron>;

class Neuron {
private:
    const double learning_rate = 0.2;
    const double alpha = 0.5;

    uint32 index;
    double gradient;
    double output;
    std::vector<Connection> output_weights;

    double sigmoid( double x );
    double sigmoid_derivative( double x );
    double random( void );
    double sumDOW( const Layer & next_layer ) const;
public:
    Neuron( uint32 output_amount, uint32 index );

    void set_output( double value );

    double get_output( void ) const ;
    std::vector<Connection> get_output_weights( void ) const;

    void feed_forward( const Layer & prev_layer );
    void calculate_output_gradients( double target );
    void calculate_hidden_gradients( const Layer & next_layer );
    void update_weights( Layer & prev_layer );
};

#endif // NEURON_H