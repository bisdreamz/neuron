package dev.neuronic.net.optimizers;

import java.util.concurrent.ExecutorService;

/**
 * Optimizer interface for neural network parameter updates.
 * 
 * <h3>Thread Safety Requirements</h3>
 * Implementations MUST be thread-safe to support parallel training. Multiple threads
 * will call {@code optimize()} simultaneously during parallel training, and implementations
 * must handle concurrent weight updates correctly.
 * 
 * <h3>Parallel Training Approaches</h3>
 * <ul>
 *   <li><b>Lock-free (Hogwild!):</b> Allow race conditions on weight updates. 
 *       Mathematically proven to converge for SGD with sparse gradients.</li>
 *   <li><b>Synchronized:</b> Use synchronization for thread safety at the cost of some parallelism.</li>
 *   <li><b>Atomic operations:</b> Use lock-free atomic updates for individual parameters.</li>
 * </ul>
 * 
 * The current implementation uses the Hogwild! approach for maximum performance.
 */
public interface Optimizer {
    /**
     * Update weights and biases using their gradients.
     * 
     * <p><b>Thread Safety:</b> This method MUST be thread-safe. Multiple threads
     * may call this method concurrently during parallel training.
     *
     * @param weights the weight parameters to update
     * @param biases the bias parameters to update
     * @param weightGradients the gradients w.r.t. weights
     * @param biasGradients the gradients w.r.t. biases
     */
    void optimize(float[][] weights, float[] biases, float[][] weightGradients, float[] biasGradients);
    
    /**
     * Update weights and biases using their gradients with optional executor for internal parallelism.
     * 
     * <p><b>Thread Safety:</b> This method MUST be thread-safe. Multiple threads
     * may call this method concurrently during parallel training.
     *
     * @param weights the weight parameters to update
     * @param biases the bias parameters to update
     * @param weightGradients the gradients w.r.t. weights
     * @param biasGradients the gradients w.r.t. biases
     * @param executor optional executor service for internal parallelism
     */
    default void optimize(float[][] weights, float[] biases, float[][] weightGradients, 
                         float[] biasGradients, ExecutorService executor) {
        if (executor == null) {
            optimize(weights, biases, weightGradients, biasGradients);
            return;
        }
        
        // Smart default: submit to executor and wait for completion
        try {
            executor.submit(() -> {
                optimize(weights, biases, weightGradients, biasGradients);
                return null;
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel optimization failed", e);
        }
    }
    
    /**
     * Set the learning rate for this optimizer.
     * 
     * <p>This method allows dynamic learning rate adjustment during training,
     * enabling learning rate scheduling strategies like cosine annealing,
     * step decay, or reduce on plateau.
     * 
     * <p><b>Thread Safety:</b> Implementations should make this thread-safe
     * as it may be called during training.
     * 
     * @param learningRate the new learning rate to use
     */
    void setLearningRate(float learningRate);
    
    /**
     * Create a variant of this optimizer suitable for embeddings.
     * 
     * <p>Embeddings have different optimization needs than dense layers:
     * <ul>
     *   <li>Reduced or no weight decay (sparse features shouldn't be pulled to zero)</li>
     *   <li>Potentially different learning rates</li>
     *   <li>Different gradient clipping thresholds</li>
     * </ul>
     * 
     * <p>Default implementation returns the same optimizer (no changes).
     * Optimizers like AdamW should override to reduce weight decay.
     * 
     * @return an optimizer configured for embedding parameters
     */
    default Optimizer forEmbeddings() {
        return this;
    }
}

