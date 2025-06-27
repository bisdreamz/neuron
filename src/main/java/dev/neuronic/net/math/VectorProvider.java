package dev.neuronic.net.math;

/**
 * Provider interface for vector operations.
 * This allows complete isolation of Vector API dependencies.
 */
public interface VectorProvider {
    
    /**
     * Check if this provider is available on the current platform.
     */
    boolean isAvailable();
    
    /**
     * Get the vector length (number of lanes).
     */
    int getVectorLength();
    
    /**
     * Get the loop bound for the given array length.
     */
    int getLoopBound(int length);
    
    /**
     * Perform vectorized dot product.
     */
    float dotProduct(float[] a, float[] b);
    
    /**
     * Perform vectorized element-wise addition.
     */
    void elementwiseAdd(float[] a, float[] b, float[] output);
    
    /**
     * Perform vectorized element-wise multiplication.
     */
    void elementwiseMultiply(float[] a, float[] b, float[] output);
    
    /**
     * Additional vector operations can be added here as needed.
     */
}