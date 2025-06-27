package dev.neuronic.net.layers;

import dev.neuronic.net.optimizers.Optimizer;

/**
 * Base implementation for Layer specifications that provides common functionality
 * for optimizer management and learning rate scaling.
 * 
 * <p><b>What is this?</b> When building neural networks, different layers often need
 * different learning rates. This base class makes it easy to configure layers with
 * custom learning speeds without complex setup.
 * 
 * <p><b>Key features:</b>
 * <ul>
 *   <li>Automatic optimizer inheritance from the network builder</li>
 *   <li>Learning rate scaling per layer</li>
 *   <li>Clean builder pattern for configuration</li>
 * </ul>
 * 
 * <p><b>Example usage:</b>
 * <pre>{@code
 * // Network with default optimizer for all layers
 * NeuralNet net = NeuralNet.newBuilder()
 *     .setDefaultOptimizer(new AdamOptimizer(0.001f))  // Default for all layers
 *     .input(784)
 *     .layer(Layers.hiddenDenseRelu(256))              // Uses default Adam(0.001f)
 *     .layer(Layers.inputEmbedding(10000, 128)
 *         .learningRateRatio(0.1f))                    // Uses Adam(0.0001f) - 10x slower
 *     .output(Layers.outputSoftmaxCrossEntropy(10));
 * }</pre>
 * 
 * @param <T> the concrete spec type for fluent API chaining
 */
public abstract class BaseLayerSpec<T extends BaseLayerSpec<T>> implements Layer.Spec {
    
    protected Optimizer optimizer;
    protected float learningRateRatio = 1.0f;
    protected final int outputSize;
    
    /**
     * Creates a new layer specification.
     * 
     * @param outputSize the number of outputs from this layer
     * @param optimizer the optimizer for this layer (null to use network default)
     */
    protected BaseLayerSpec(int outputSize, Optimizer optimizer) {
        if (outputSize <= 0) {
            throw new IllegalArgumentException("Output size must be positive: " + outputSize);
        }
        this.outputSize = outputSize;
        this.optimizer = optimizer;
    }
    
    /**
     * Sets a specific optimizer for this layer, overriding the network default.
     * 
     * <p><b>What is this?</b> An optimizer controls how the network learns from its mistakes.
     * Different optimizers work better for different types of problems.
     * 
     * <p><b>When to use:</b> Only set this if you need a different optimizer than the
     * network default. Most layers should use the same optimizer.
     * 
     * <p><b>Common optimizers:</b>
     * <ul>
     *   <li><b>AdamOptimizer</b> - Excellent default choice with adaptive learning rates</li>
     *   <li><b>AdamWOptimizer</b> - Adam with better weight decay, great for large models</li>
     *   <li><b>SgdOptimizer</b> - Simple and fast, good for large datasets</li>
     * </ul>
     * 
     * @param optimizer the optimizer to use for this layer
     * @return this spec for method chaining
     */
    @SuppressWarnings("unchecked")
    public T optimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        return (T) this;
    }
    
    /**
     * Sets the learning rate ratio for this layer relative to the optimizer's base learning rate.
     * 
     * <p><b>What is this?</b> When training neural networks, different layers often learn best at 
     * different speeds. This ratio adjusts how fast this specific layer learns compared to others.
     * 
     * <p><b>Common values:</b>
     * <ul>
     *   <li><b>1.0</b> (default) - Same learning rate as other layers</li>
     *   <li><b>0.1 to 0.5</b> - Slower learning, common for embedding layers that need stable updates</li>
     *   <li><b>2.0 to 10.0</b> - Faster learning, rare but useful for layers that need rapid adaptation</li>
     * </ul>
     * 
     * <p><b>Example:</b> If your network uses Adam optimizer with learning rate 0.001, and you set
     * {@code learningRateRatio(0.1)}, this layer will use learning rate 0.0001 (10x slower).
     * 
     * <p><b>When to use:</b>
     * <ul>
     *   <li><b>Embedding layers:</b> Often use 0.1-0.5 to prevent large updates from rare words</li>
     *   <li><b>Pre-trained layers:</b> Use smaller ratios (0.01-0.1) to preserve learned features</li>
     *   <li><b>Output layers:</b> Sometimes use 2.0-5.0 for faster convergence</li>
     *   <li><b>Most other layers:</b> Keep default 1.0</li>
     * </ul>
     * 
     * <p><b>Why it matters:</b> In language models, rare words appear infrequently. When they do appear,
     * they can cause large updates that destabilize the embeddings. A smaller learning rate ratio
     * helps keep these updates controlled.
     * 
     * @param ratio multiplier for the learning rate (e.g., 0.1 = 10x slower, 2.0 = 2x faster)
     * @return this spec for method chaining
     * @throws IllegalArgumentException if ratio is not positive
     */
    @SuppressWarnings("unchecked")
    public T learningRateRatio(float ratio) {
        if (ratio <= 0) {
            throw new IllegalArgumentException("Learning rate ratio must be positive: " + ratio);
        }
        this.learningRateRatio = ratio;
        return (T) this;
    }
    
    @Override
    public int getOutputSize() {
        return outputSize;
    }
    
    @Override
    public Layer create(int inputSize, Optimizer defaultOptimizer) {
        return createLayer(inputSize, getEffectiveOptimizer(defaultOptimizer));
    }
    
    /**
     * Creates the actual layer instance with the effective optimizer.
     * Subclasses implement this method to create their specific layer type.
     * 
     * @param inputSize the input size for the layer
     * @param effectiveOptimizer the optimizer to use (after applying default and learning rate scaling)
     * @return the created layer
     */
    protected abstract Layer createLayer(int inputSize, Optimizer effectiveOptimizer);
    
    /**
     * Gets the effective optimizer for this layer, applying learning rate scaling if needed.
     * This method is called during layer creation to set up the actual optimizer.
     * 
     * @param defaultOptimizer the network's default optimizer (used if this layer has no specific optimizer)
     * @return the optimizer to use for this layer, potentially wrapped for learning rate scaling
     * @throws IllegalStateException if no optimizer is available (neither layer-specific nor default)
     */
    protected Optimizer getEffectiveOptimizer(Optimizer defaultOptimizer) {
        Optimizer baseOptimizer = (optimizer != null) ? optimizer : defaultOptimizer;
        
        if (baseOptimizer == null) {
            throw new IllegalStateException(
                "No optimizer available. Either set a default optimizer on the network builder " +
                "or specify an optimizer for this layer.");
        }
        
        // If learning rate ratio is 1.0, use the optimizer as-is
        if (learningRateRatio == 1.0f) {
            return baseOptimizer;
        }
        
        // Otherwise, wrap it to scale the learning rate
        return new LearningRateScaledOptimizer(baseOptimizer, learningRateRatio);
    }
    
    /**
     * Internal wrapper that scales the learning rate of another optimizer.
     * This allows different layers to learn at different speeds while using
     * the same underlying optimization algorithm.
     */
    private static class LearningRateScaledOptimizer implements Optimizer {
        private final Optimizer wrapped;
        private final float scale;
        
        LearningRateScaledOptimizer(Optimizer wrapped, float scale) {
            this.wrapped = wrapped;
            this.scale = scale;
        }
        
        @Override
        public void optimize(float[][] weights, float[] biases, float[][] weightGradients, float[] biasGradients) {
            // Scale gradients before passing to wrapped optimizer
            // This is mathematically equivalent to scaling the learning rate
            scaleGradients(weightGradients, scale);
            scaleGradients(biasGradients, scale);
            
            wrapped.optimize(weights, biases, weightGradients, biasGradients);
            
            // Restore gradients in case they're used elsewhere
            scaleGradients(weightGradients, 1.0f / scale);
            scaleGradients(biasGradients, 1.0f / scale);
        }
        
        @Override
        public void optimize(float[][] weights, float[] biases, float[][] weightGradients, float[] biasGradients, 
                           java.util.concurrent.ExecutorService executor) {
            // Scale gradients
            scaleGradients(weightGradients, scale);
            scaleGradients(biasGradients, scale);
            
            wrapped.optimize(weights, biases, weightGradients, biasGradients, executor);
            
            // Restore gradients
            scaleGradients(weightGradients, 1.0f / scale);
            scaleGradients(biasGradients, 1.0f / scale);
        }
        
        private void scaleGradients(float[][] gradients, float scale) {
            for (float[] row : gradients) {
                scaleGradients(row, scale);
            }
        }
        
        private void scaleGradients(float[] gradients, float scale) {
            for (int i = 0; i < gradients.length; i++) {
                gradients[i] *= scale;
            }
        }
        
        @Override
        public void setLearningRate(float learningRate) {
            // Delegate to the wrapped optimizer
            wrapped.setLearningRate(learningRate);
        }
    }
}