package dev.neuronic.net.layers;

/**
 * Interface for layers that support gradient accumulation for mini-batch training.
 * 
 * <p>Gradient accumulation allows true mini-batch training where gradients are
 * computed for multiple samples before updating weights. This improves training
 * stability and enables larger effective batch sizes.
 * 
 * <p>The typical flow is:
 * <ol>
 *   <li>Call {@code startAccumulation()} to begin accumulating gradients</li>
 *   <li>Process multiple samples, accumulating gradients for each</li>
 *   <li>Call {@code applyAccumulatedGradients()} to update weights</li>
 * </ol>
 * 
 * <p>Implementations must handle thread-safe accumulation since multiple threads
 * may be accumulating gradients simultaneously during parallel training.
 */
public interface GradientAccumulator {
    
    /**
     * Start accumulating gradients for a new batch.
     * Clears any previously accumulated gradients.
     */
    void startAccumulation();
    
    /**
     * Backward pass with gradient accumulation.
     * Accumulates gradients without updating weights.
     * 
     * @param stack the layer context stack from forward pass
     * @param stackIndex this layer's index in the stack
     * @param upstreamGradient gradient from the layer above
     * @return gradient to pass to the layer below
     */
    float[] backwardAccumulate(Layer.LayerContext[] stack, int stackIndex, float[] upstreamGradient);
    
    /**
     * Apply accumulated gradients to update weights.
     * This should be called after all samples in the batch have been processed.
     * 
     * @param batchSize the number of samples that were accumulated
     */
    void applyAccumulatedGradients(int batchSize);
    
    /**
     * Check if this layer is currently accumulating gradients.
     * 
     * @return true if accumulating, false otherwise
     */
    boolean isAccumulating();
}