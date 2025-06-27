package dev.neuronic.net.math;

import dev.neuronic.net.math.ops.VectorNorm;

/**
 * Adaptive gradient clipper that adjusts threshold based on gradient history.
 * 
 * <p><b>Algorithm:</b>
 * <ul>
 *   <li>Track moving average of gradient norms</li>
 *   <li>Adjust threshold based on recent gradient behavior</li>
 *   <li>More aggressive clipping when gradients are consistently large</li>
 *   <li>Less aggressive when gradients are well-behaved</li>
 * </ul>
 * 
 * <p><b>Benefits:</b>
 * <ul>
 *   <li>Automatically adapts to training dynamics</li>
 *   <li>Reduces need for manual threshold tuning</li>
 *   <li>Can improve convergence in unstable training scenarios</li>
 * </ul>
 * 
 * <p><b>Thread Safety:</b> Uses thread-local state for concurrent training.
 */
final class AdaptiveClipper implements GradientClipper {
    
    private final float initialThreshold;
    private final float adaptationRate;
    private final ThreadLocal<AdaptiveState> state;
    
    private static class AdaptiveState {
        float currentThreshold;
        float movingAverage;
        long updateCount;
        
        AdaptiveState(float initialThreshold) {
            this.currentThreshold = initialThreshold;
            this.movingAverage = initialThreshold;
            this.updateCount = 0;
        }
    }
    
    AdaptiveClipper(float initialThreshold, float adaptationRate) {
        if (initialThreshold <= 0) {
            throw new IllegalArgumentException("Initial threshold must be positive, got: " + initialThreshold);
        }
        if (adaptationRate <= 0 || adaptationRate >= 1) {
            throw new IllegalArgumentException("Adaptation rate must be in (0, 1), got: " + adaptationRate);
        }
        
        this.initialThreshold = initialThreshold;
        this.adaptationRate = adaptationRate;
        this.state = ThreadLocal.withInitial(() -> new AdaptiveState(initialThreshold));
    }
    
    @Override
    public boolean clipInPlace(float[] gradients) {
        if (gradients.length == 0) return false;
        
        AdaptiveState currentState = state.get();
        
        // Calculate current gradient norm
        float norm = (float) Math.sqrt(VectorNorm.computeL2Squared(gradients));
        
        // Update moving average and threshold
        updateThreshold(currentState, norm);
        
        // Apply clipping if needed
        if (norm > currentState.currentThreshold) {
            float scale = currentState.currentThreshold / norm;
            NetMath.elementwiseScaleInPlace(gradients, scale);
            return true;
        }
        
        return false;
    }
    
    private void updateThreshold(AdaptiveState currentState, float currentNorm) {
        // Update moving average of gradient norms
        if (currentState.updateCount == 0) {
            currentState.movingAverage = currentNorm;
        } else {
            currentState.movingAverage = (1 - adaptationRate) * currentState.movingAverage + 
                                       adaptationRate * currentNorm;
        }
        
        // Adapt threshold based on moving average
        // If recent norms are high, lower threshold (more aggressive clipping)
        // If recent norms are low, raise threshold (less aggressive clipping)
        float targetThreshold = Math.max(initialThreshold * 0.1f, 
                                       Math.min(initialThreshold * 2.0f, 
                                              currentState.movingAverage * 1.2f));
        
        currentState.currentThreshold = (1 - adaptationRate) * currentState.currentThreshold + 
                                      adaptationRate * targetThreshold;
        
        currentState.updateCount++;
    }
    
    @Override
    public boolean wouldClip(float[] gradients) {
        if (gradients.length == 0) return false;
        
        AdaptiveState currentState = state.get();
        float norm = (float) Math.sqrt(VectorNorm.computeL2Squared(gradients));
        return norm > currentState.currentThreshold;
    }
    
    @Override
    public String getDescription() {
        AdaptiveState currentState = state.get();
        return String.format("AdaptiveClipper(initial=%.3f, rate=%.3f, current=%.3f)", 
                           initialThreshold, adaptationRate, currentState.currentThreshold);
    }
    
    /**
     * Get the current threshold for this thread.
     */
    public float getCurrentThreshold() {
        return state.get().currentThreshold;
    }
    
    /**
     * Get the current moving average of gradient norms for this thread.
     */
    public float getMovingAverage() {
        return state.get().movingAverage;
    }
    
    /**
     * Reset the adaptive state for this thread.
     */
    public void reset() {
        state.set(new AdaptiveState(initialThreshold));
    }
    
    @Override
    public String toString() {
        return getDescription();
    }
}