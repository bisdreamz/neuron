package dev.neuronic.net.math;

/**
 * Scalar fallback implementation of VectorProvider.
 * This is always available and contains no Vector API dependencies.
 */
public final class ScalarProvider implements VectorProvider {
    
    private static final ScalarProvider INSTANCE = new ScalarProvider();
    
    public static ScalarProvider getInstance() {
        return INSTANCE;
    }
    
    @Override
    public boolean isAvailable() {
        return true; // Scalar operations are always available
    }
    
    @Override
    public int getVectorLength() {
        return 1; // Scalar = 1 lane
    }
    
    @Override
    public int getLoopBound(int length) {
        return 0; // No vectorization
    }
    
    @Override
    public float dotProduct(float[] a, float[] b) {
        float sum = 0;
        int i = 0;
        int unrollBound = a.length - 3;
        
        // 4-way unrolled for performance
        for (; i < unrollBound; i += 4) {
            sum += a[i] * b[i] + 
                   a[i+1] * b[i+1] + 
                   a[i+2] * b[i+2] + 
                   a[i+3] * b[i+3];
        }
        
        // Handle remainder
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        
        return sum;
    }
    
    @Override
    public void elementwiseAdd(float[] a, float[] b, float[] output) {
        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] + b[i];
        }
    }
    
    @Override
    public void elementwiseMultiply(float[] a, float[] b, float[] output) {
        for (int i = 0; i < a.length; i++) {
            output[i] = a[i] * b[i];
        }
    }
    
    private ScalarProvider() {} // Singleton
}