package dev.neuronic.net.math;

/**
 * Interface for gradient clipping strategies.
 * 
 * <p><b>Purpose:</b> Prevent exploding gradients during training by limiting gradient magnitude.
 * Essential for training stability in RNNs, GRUs, LSTMs, and deep networks.
 * 
 * <p><b>Common Use Cases:</b>
 * <ul>
 *   <li><b>L2 Norm Clipping:</b> Most common - clip gradients when norm exceeds threshold</li>
 *   <li><b>Value Clipping:</b> Clip individual gradient values to [-max, max] range</li>
 *   <li><b>Adaptive Clipping:</b> Dynamically adjust clipping threshold based on gradient history</li>
 * </ul>
 * 
 * <p><b>Performance:</b> All implementations use vectorized operations for optimal performance.
 * 
 * <p><b>Thread Safety:</b> All clipping operations are thread-safe and stateless.
 */
public interface GradientClipper {
    
    /**
     * Clip gradients in-place to prevent exploding gradients.
     * 
     * @param gradients the gradient array to clip (modified in-place)
     * @return true if clipping was applied, false if gradients were within bounds
     */
    boolean clipInPlace(float[] gradients);
    
    /**
     * Check if gradients would be clipped without actually modifying them.
     * Useful for monitoring gradient behavior.
     * 
     * @param gradients the gradients to check
     * @return true if gradients exceed clipping threshold
     */
    boolean wouldClip(float[] gradients);
    
    /**
     * Get a description of this clipping strategy for logging/debugging.
     */
    String getDescription();
    
    // Factory methods for common clipping strategies
    
    /**
     * Create L2 norm gradient clipper - most commonly used.
     * Clips gradients when their L2 norm exceeds maxNorm.
     * 
     * @param maxNorm maximum allowed L2 norm (typically 1.0 to 5.0)
     * @return norm-based clipper
     */
    static GradientClipper byNorm(float maxNorm) {
        return new NormClipper(maxNorm);
    }
    
    /**
     * Create value-based gradient clipper.
     * Clips individual gradient values to [-maxValue, maxValue] range.
     * 
     * @param maxValue maximum absolute value for any gradient
     * @return value-based clipper
     */
    static GradientClipper byValue(float maxValue) {
        return new ValueClipper(maxValue);
    }
    
    /**
     * Create adaptive gradient clipper that adjusts threshold based on gradient history.
     * 
     * @param initialThreshold initial clipping threshold
     * @param adaptationRate how quickly to adapt (0.01 = 1% adjustment per update)
     * @return adaptive clipper
     */
    static GradientClipper adaptive(float initialThreshold, float adaptationRate) {
        return new AdaptiveClipper(initialThreshold, adaptationRate);
    }
    
    /**
     * No-op clipper that never clips - useful for disabling clipping conditionally.
     */
    static GradientClipper none() {
        return NoOpClipper.INSTANCE;
    }
}