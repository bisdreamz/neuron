package dev.neuronic.net.layers;

/**
 * Layer Normalization component for neural networks.
 * 
 * <p><b>What it does:</b> Normalizes inputs across features (not batch) to have 
 * zero mean and unit variance, then applies learnable scale and shift parameters.
 * 
 * <p><b>Why it helps:</b>
 * <ul>
 *   <li>Stabilizes training by reducing internal covariate shift</li>
 *   <li>Allows higher learning rates for faster convergence</li>
 *   <li>Acts as regularization, improving generalization</li>
 *   <li>Especially beneficial for RNNs and Transformers</li>
 * </ul>
 * 
 * <p><b>Layer Norm vs Batch Norm:</b>
 * <ul>
 *   <li><b>Layer Norm:</b> Normalizes across features for each example independently</li>
 *   <li><b>Batch Norm:</b> Normalizes across batch for each feature</li>
 *   <li>Layer Norm works with any batch size (even 1) and is preferred for RNNs</li>
 * </ul>
 * 
 * <p><b>Algorithm:</b>
 * <pre>
 * μ = mean(x)                      // Mean across features
 * σ² = variance(x)                 // Variance across features  
 * x̂ = (x - μ) / √(σ² + ε)         // Normalize
 * y = γ * x̂ + β                   // Scale and shift with learned parameters
 * </pre>
 */
public class LayerNorm {
    
    private final int size;
    private final float epsilon;
    private final float[] gamma;  // Scale parameters (learned)
    private final float[] beta;   // Shift parameters (learned)
    
    // Thread-local buffers for efficiency
    private final ThreadLocal<float[]> normalizedBuffer;
    private final ThreadLocal<Stats> statsBuffer;
    
    /**
     * Container for mean and variance statistics.
     */
    public static class Stats {
        public float mean;
        public float variance;
    }
    
    /**
     * Create a layer normalization component.
     * 
     * @param size the size of the feature dimension to normalize
     * @param epsilon small constant for numerical stability (typically 1e-5)
     */
    public LayerNorm(int size, float epsilon) {
        if (size <= 0) {
            throw new IllegalArgumentException("Size must be positive: " + size);
        }
        if (epsilon <= 0) {
            throw new IllegalArgumentException("Epsilon must be positive: " + epsilon);
        }
        
        this.size = size;
        this.epsilon = epsilon;
        
        // Initialize scale to 1 and shift to 0
        this.gamma = new float[size];
        this.beta = new float[size];
        for (int i = 0; i < size; i++) {
            gamma[i] = 1.0f;
            beta[i] = 0.0f;
        }
        
        // Thread-local buffers
        this.normalizedBuffer = ThreadLocal.withInitial(() -> new float[size]);
        this.statsBuffer = ThreadLocal.withInitial(Stats::new);
    }
    
    /**
     * Create with default epsilon (1e-5).
     */
    public LayerNorm(int size) {
        this(size, 1e-5f);
    }
    
    /**
     * Apply layer normalization forward pass.
     * 
     * @param input the input vector to normalize
     * @param output the output vector (can be same as input for in-place)
     * @return mean and variance for backward pass
     */
    public Stats forward(float[] input, float[] output) {
        if (input.length != size || output.length != size) {
            throw new IllegalArgumentException("Input/output size mismatch");
        }
        
        Stats stats = statsBuffer.get();
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += input[i];
        }
        stats.mean = sum / size;
        
        // Compute variance
        float varSum = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = input[i] - stats.mean;
            varSum += diff * diff;
        }
        stats.variance = varSum / size;
        
        // Normalize and apply scale/shift
        float stdInv = 1.0f / (float)Math.sqrt(stats.variance + epsilon);
        for (int i = 0; i < size; i++) {
            float normalized = (input[i] - stats.mean) * stdInv;
            output[i] = gamma[i] * normalized + beta[i];
        }
        
        return stats;
    }
    
    /**
     * Apply layer normalization forward pass (returns normalized values for backward).
     * 
     * @param input the input vector to normalize
     * @param output the output vector
     * @param normalized buffer to store normalized values (before scale/shift)
     * @return mean and variance for backward pass
     */
    public Stats forward(float[] input, float[] output, float[] normalized) {
        if (input.length != size || output.length != size || normalized.length != size) {
            throw new IllegalArgumentException("Size mismatch");
        }
        
        Stats stats = statsBuffer.get();
        
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += input[i];
        }
        stats.mean = sum / size;
        
        // Compute variance
        float varSum = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = input[i] - stats.mean;
            varSum += diff * diff;
        }
        stats.variance = varSum / size;
        
        // Normalize and apply scale/shift
        float stdInv = 1.0f / (float)Math.sqrt(stats.variance + epsilon);
        for (int i = 0; i < size; i++) {
            normalized[i] = (input[i] - stats.mean) * stdInv;
            output[i] = gamma[i] * normalized[i] + beta[i];
        }
        
        return stats;
    }
    
    /**
     * Compute gradients for layer normalization backward pass.
     * 
     * @param upstreamGradient gradient from upstream layer
     * @param normalized the normalized values from forward pass
     * @param stats the statistics from forward pass
     * @param inputGradient gradient to propagate to input (output)
     * @param gammaGradient gradient for gamma parameters (output)
     * @param betaGradient gradient for beta parameters (output)
     */
    public void backward(float[] upstreamGradient, float[] normalized, Stats stats,
                        float[] inputGradient, float[] gammaGradient, float[] betaGradient) {
        
        float stdInv = 1.0f / (float)Math.sqrt(stats.variance + epsilon);
        
        // Gradients for gamma and beta
        for (int i = 0; i < size; i++) {
            gammaGradient[i] = upstreamGradient[i] * normalized[i];
            betaGradient[i] = upstreamGradient[i];
        }
        
        // Gradient for input (more complex due to normalization)
        float[] dxNorm = normalizedBuffer.get();
        for (int i = 0; i < size; i++) {
            dxNorm[i] = upstreamGradient[i] * gamma[i];
        }
        
        // Compute gradients through normalization
        float dvar = 0.0f;
        float dmean = 0.0f;
        
        for (int i = 0; i < size; i++) {
            float xCentered = normalized[i] / stdInv; // Recover (x - mean)
            dvar += dxNorm[i] * xCentered * -0.5f * stdInv * stdInv * stdInv;
            dmean += dxNorm[i] * -stdInv;
        }
        
        dmean += dvar * -2.0f * stats.mean / size;
        
        // Final input gradients
        for (int i = 0; i < size; i++) {
            float xCentered = normalized[i] / stdInv;
            inputGradient[i] = dxNorm[i] * stdInv + dvar * 2.0f * xCentered / size + dmean / size;
        }
    }
    
    /**
     * Update the scale and shift parameters.
     * 
     * @param gammaGradient gradients for gamma
     * @param betaGradient gradients for beta  
     * @param learningRate learning rate for update
     */
    public void updateParameters(float[] gammaGradient, float[] betaGradient, float learningRate) {
        for (int i = 0; i < size; i++) {
            gamma[i] -= learningRate * gammaGradient[i];
            beta[i] -= learningRate * betaGradient[i];
        }
    }
    
    /**
     * Get the scale parameters (gamma).
     */
    public float[] getGamma() {
        return gamma;
    }
    
    /**
     * Get the shift parameters (beta).
     */
    public float[] getBeta() {
        return beta;
    }
    
    /**
     * Get the feature size.
     */
    public int getSize() {
        return size;
    }
    
    /**
     * Get epsilon value.
     */
    public float getEpsilon() {
        return epsilon;
    }
}