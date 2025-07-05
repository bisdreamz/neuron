package dev.neuronic.net;

/**
 * Weight initialization strategies for neural network layers.
 * 
 * <p>The choice of initialization strategy significantly affects training performance
 * and convergence. Different strategies work better with different activation functions.
 */
public enum WeightInitStrategy {
    
    /**
     * Xavier/Glorot uniform initialization - the gold standard for sigmoid and tanh activations.
     * 
     * <p><strong>What it does:</strong>
     * Initializes weights uniformly in the range [-limit, +limit] where:
     * limit = sqrt(6 / (fanIn + fanOut))
     * 
     * <p><strong>When to use:</strong>
     * <ul>
     *   <li>With sigmoid activation functions (recommended default)</li>
     *   <li>With tanh activation functions (recommended default)</li>
     *   <li>With GRU, LSTM, or other recurrent layers (they use sigmoid/tanh gates)</li>
     *   <li>When training is unstable with He initialization</li>
     * </ul>
     * 
     * <p><strong>Why it works:</strong>
     * Keeps the variance of activations roughly equal across all layers by considering
     * both the number of inputs and outputs. This prevents vanishing or exploding
     * gradients in deep networks with sigmoid/tanh activations.
     * 
     * <p><strong>Technical note:</strong>
     * Uses uniform distribution U(-limit, +limit) rather than normal distribution
     * because it provides slightly better empirical results and is the most common
     * implementation in major frameworks.
     */
    XAVIER,
    
    /**
     * He initialization: w = random_gaussian * sqrt(2 / fanIn)
     * 
     * <p><strong>When to use:</strong>
     * <ul>
     *   <li>With ReLU activation functions</li>
     *   <li>With Leaky ReLU activation functions</li>
     *   <li>With any activation that zeros negative inputs</li>
     *   <li>In most modern deep learning architectures</li>
     * </ul>
     * 
     * <p><strong>Why it works:</strong>
     * Accounts for the fact that ReLU activations zero out half the neurons on average,
     * effectively reducing the variance by half. Uses only fanIn (not fanOut) and a
     * factor of 2 to compensate for this variance reduction.
     * 
     * <p><strong>Recommended default:</strong>
     * This is the most commonly used initialization in modern neural networks since
     * ReLU and its variants are the most popular activation functions.
     */
    HE,

    /**
     * He initialization plus uniform noise.
     *
     * <p><strong>When to use:</strong>
     * <ul>
     *   <li>When you suspect the model is getting stuck in a local minimum</li>
     *   <li>When you want to add a small amount of noise to the initialization</li>
     * </ul>
     *
     * <p><strong>Why it works:</strong>
     * Adds a small amount of uniform noise to the He initialization, which can help
     * break symmetries and prevent the model from getting stuck in a collapsed state.
     */
    HE_PLUS_UNIFORM_NOISE
}