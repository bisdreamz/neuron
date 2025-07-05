package dev.neuronic.net.common;

import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * Thread-safe pool for 3D float arrays of a specific size.
 * Designed for embedding gradient buffers in neural network layers.
 * 
 * <p>Similar to PooledFloatArray but for 3D arrays, this class manages
 * reusable float[][][] buffers to avoid allocation overhead in hot paths.
 * 
 * <p>Usage pattern:
 * <pre>{@code
 * PooledFloatArray3D pool = new PooledFloatArray3D(features, maxValues, embeddingDim);
 * float[][][] buffer = pool.getBuffer();
 * try {
 *     // Use buffer for computations
 * } finally {
 *     pool.releaseBuffer(buffer);
 * }
 * }</pre>
 */
public class PooledFloatArray3D {
    private final int dim1;
    private final int dim2;
    private final int dim3;
    private final ConcurrentLinkedDeque<float[][][]> bufferPool;
    
    /**
     * Create a pool for 3D float arrays of the specified dimensions.
     * 
     * @param dim1 first dimension size (e.g., number of features)
     * @param dim2 second dimension size (e.g., vocabulary size)
     * @param dim3 third dimension size (e.g., embedding dimension)
     */
    public PooledFloatArray3D(int dim1, int dim2, int dim3) {
        if (dim1 <= 0 || dim2 <= 0 || dim3 <= 0) {
            throw new IllegalArgumentException(String.format(
                "All dimensions must be positive: [%d, %d, %d]", dim1, dim2, dim3));
        }
        this.dim1 = dim1;
        this.dim2 = dim2;
        this.dim3 = dim3;
        this.bufferPool = new ConcurrentLinkedDeque<>();
    }
    
    /**
     * Get a 3D float array from the pool.
     * Returns a reused array if available, otherwise creates a new one.
     * All returned arrays are guaranteed to be zeroed.
     * 
     * @return a 3D float array of the configured dimensions, zeroed and ready for use
     */
    public float[][][] getBuffer() {
        return getBuffer(true);
    }
    
    /**
     * Get a 3D float array from the pool with optional zeroing.
     * Returns a reused array if available, otherwise creates a new one.
     * 
     * @param zero whether to zero the array (false for performance when immediately overwriting)
     * @return a 3D float array of the configured dimensions
     */
    public float[][][] getBuffer(boolean zero) {
        float[][][] buffer = bufferPool.poll();
        if (buffer != null) {
            if (zero) {
                // Zero out reused buffer to prevent contamination
                zeroBuffer(buffer);
            }
            return buffer;
        } else {
            // Create new 3D array - Java already initializes to zero
            return new float[dim1][dim2][dim3];
        }
    }
    
    /**
     * Return a 3D float array to the pool for reuse.
     * Only arrays of the correct dimensions are accepted.
     * 
     * @param buffer the array to return (null-safe, wrong-size arrays ignored)
     */
    public void releaseBuffer(float[][][] buffer) {
        if (buffer != null && 
            buffer.length == dim1 && 
            buffer[0].length == dim2 && 
            buffer[0][0].length == dim3) {
            bufferPool.offer(buffer);
        }
    }
    
    /**
     * Zero out all elements in the 3D buffer.
     */
    private void zeroBuffer(float[][][] buffer) {
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                java.util.Arrays.fill(buffer[i][j], 0.0f);
            }
        }
    }
    
    /**
     * Get the dimensions this pool manages.
     * 
     * @return array of dimensions [dim1, dim2, dim3]
     */
    public int[] getDimensions() {
        return new int[] { dim1, dim2, dim3 };
    }
    
    /**
     * Get the current number of buffers in the pool.
     * 
     * @return number of available buffers (may change concurrently)
     */
    public int getPoolSize() {
        return bufferPool.size();
    }
}