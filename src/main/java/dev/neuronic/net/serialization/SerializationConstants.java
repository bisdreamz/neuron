package dev.neuronic.net.serialization;

/**
 * Constants for neural network serialization format.
 */
public final class SerializationConstants {
    
    // File format identification
    public static final int MAGIC_NUMBER = 0x4E4E4D44; // "NNMD" - Neural Network Model Data
    public static final int CURRENT_VERSION = 1;
    
    // Type IDs for neural network and layers
    public static final int TYPE_NEURAL_NET = 0;
    public static final int TYPE_DENSE_LAYER = 1;
    public static final int TYPE_SOFTMAX_CROSSENTROPY_OUTPUT = 2;
    public static final int TYPE_LINEAR_REGRESSION_OUTPUT = 3;
    public static final int TYPE_SIGMOID_BINARY_OUTPUT = 4;
    public static final int TYPE_MULTILABEL_SIGMOID_OUTPUT = 5;
    public static final int TYPE_INPUT_EMBEDDING_LAYER = 6;
    public static final int TYPE_GRU_LAYER = 7;
    public static final int TYPE_MIXED_FEATURE_INPUT_LAYER = 8;
    public static final int TYPE_SIMPLE_NET = 9;
    public static final int TYPE_SIMPLE_NET_FLOAT = 10;
    public static final int TYPE_SIMPLE_NET_STRING = 11;
    public static final int TYPE_SIMPLE_NET_MULTI_FLOAT = 12;
    public static final int TYPE_INPUT_SEQUENCE_EMBEDDING_LAYER = 13;
    public static final int TYPE_DROPOUT_LAYER = 14;
    public static final int TYPE_LAYER_NORM_LAYER = 15;
    public static final int TYPE_HUBER_REGRESSION_OUTPUT = 16;
    
    // Type IDs for activators (stateless singletons)
    public static final int TYPE_RELU_ACTIVATOR = 100;
    public static final int TYPE_SIGMOID_ACTIVATOR = 101;
    public static final int TYPE_TANH_ACTIVATOR = 102;
    public static final int TYPE_SOFTMAX_ACTIVATOR = 103;
    public static final int TYPE_LINEAR_ACTIVATOR = 104;
    public static final int TYPE_LEAKY_RELU_ACTIVATOR = 105;
    
    // Type IDs for optimizers
    public static final int TYPE_SGD_OPTIMIZER = 200;
    public static final int TYPE_ADAM_OPTIMIZER = 201;
    public static final int TYPE_ADAMW_OPTIMIZER = 202;
    // Future: TYPE_RMSPROP_OPTIMIZER = 203, etc.
    
    // Special type ID for custom registered types (followed by class name string)
    public static final int TYPE_CUSTOM = 999;
    
    // Weight initialization strategies
    public static final int WEIGHT_INIT_XAVIER = 1;
    public static final int WEIGHT_INIT_HE = 2;
    
    // File structure markers
    public static final int SECTION_METADATA = 0x1000;
    public static final int SECTION_LAYERS = 0x1001;
    public static final int SECTION_END = 0x1999;
    
    private SerializationConstants() {} // Prevent instantiation
}