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
     * Update 1D parameters using their gradients.
     * 
     * <p>This method supports optimization of 1D parameter arrays such as LayerNorm
     * gamma/beta, bias-only layers, or other vector parameters that don't fit the
     * traditional weight matrix + bias vector pattern.
     * 
     * <p><b>Thread Safety:</b> This method MUST be thread-safe. Multiple threads
     * may call this method concurrently during parallel training.
     *
     * @param parameters the 1D parameters to update
     * @param gradients the gradients w.r.t. parameters
     */
    default void optimize(float[] parameters, float[] gradients) {
        // Default implementation: treat as bias update using existing optimize method
        // This maintains backward compatibility while allowing optimizers to override
        // with more efficient implementations
        float[][] dummyWeights = new float[0][0];
        float[][] dummyWeightGrads = new float[0][0];
        optimize(dummyWeights, parameters, dummyWeightGrads, gradients);
    }
    
    /**
     * Update 1D parameters using their gradients with optional executor for internal parallelism.
     * 
     * <p><b>Thread Safety:</b> This method MUST be thread-safe. Multiple threads
     * may call this method concurrently during parallel training.
     *
     * @param parameters the 1D parameters to update
     * @param gradients the gradients w.r.t. parameters
     * @param executor optional executor service for internal parallelism
     */
    default void optimize(float[] parameters, float[] gradients, ExecutorService executor) {
        if (executor == null) {
            optimize(parameters, gradients);
            return;
        }
        
        // Smart default: submit to executor and wait for completion
        try {
            executor.submit(() -> {
                optimize(parameters, gradients);
                return null;
            }).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel optimization failed", e);
        }
    }
    
    /**
     * Update weights and biases using a stable key for stateful optimizers.
     * This is crucial for layers like embeddings where the weight/gradient arrays
     * may be temporary or represent a subset of the full parameter set.
     *
     * @param stateKey a stable object to key optimizer state (e.g., the full embedding table)
     * @param weights the weight parameters to update
     * @param biases the bias parameters to update
     * @param weightGradients the gradients w.r.t. weights
     * @param biasGradients the gradients w.r.t. biases
     * @param executor optional executor service for internal parallelism
     */
    default void optimize(Object stateKey, float[][] weights, float[] biases,
                         float[][] weightGradients, float[] biasGradients,
                         ExecutorService executor) {
        // Default implementation falls back to the original method for backward compatibility.
        // Stateful optimizers (Adam, AdamW) MUST override this to handle state correctly.
        optimize(weights, biases, weightGradients, biasGradients, executor);
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
     * Performs a sparse "lazy" update, ideal for embedding layers.
     * Only the optimizer states for the specified indices are updated, preventing state decay
     * for untouched parameters. This method must be implemented by all stateful optimizers.
     *
     * @param stateKey         A stable object to key the optimizer state (e.g., the full embedding table).
     * @param allWeights       The full weight matrix (e.g., the entire embedding table).
     * @param indicesToUpdate  An array of indices corresponding to the rows in `allWeights` that need updating.
     * @param gradients        An array of gradients, where `gradients[i]` corresponds to `indicesToUpdate[i]`.
     *                         Must have the same length as `indicesToUpdate`.
     * @param executor         Optional executor for parallelization.
     */
    void sparseOptimize(Object stateKey, float[][] allWeights, int[] indicesToUpdate,
                        float[][] gradients, ExecutorService executor);
    
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

