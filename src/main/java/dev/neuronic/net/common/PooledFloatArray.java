package dev.neuronic.net.common;

import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * Simple thread-safe pool for float arrays of a specific size.
 * Each instance manages buffers for exactly one array size.
 * 
 * Usage:
 * - Create one PooledFloatArray per buffer size needed
 * - Call getBuffer() to obtain an array
 * - Call releaseBuffer() when done to return it to the pool
 * - Arrays are reused to avoid allocation overhead
 */
public class PooledFloatArray {
    private final int bufferSize;
    private final ConcurrentLinkedDeque<float[]> bufferPool;

    /**
     * Create a pool for float arrays of the specified size.
     * 
     * @param bufferSize the size of arrays this pool will manage
     */
    public PooledFloatArray(int bufferSize) {
        if (bufferSize <= 0) {
            throw new IllegalArgumentException("Buffer size must be positive: " + bufferSize);
        }
        this.bufferSize = bufferSize;
        this.bufferPool = new ConcurrentLinkedDeque<>();
    }

    /**
     * Get a float array from the pool.
     * Returns a reused array if available, otherwise creates a new one.
     * All returned arrays are guaranteed to be zeroed.
     * 
     * @return a float array of the configured size, zeroed and ready for use
     */
    public float[] getBuffer() {
        return getBuffer(true);
    }

    /**
     * Get a float array from the pool with optional zeroing.
     * Returns a reused array if available, otherwise creates a new one.
     * 
     * @param zero whether to zero the array (false for performance when immediately overwriting)
     * @return a float array of the configured size
     */
    public float[] getBuffer(boolean zero) {
        float[] buffer = bufferPool.poll();
        if (buffer != null) {
            if (zero) {
                // Zero out reused buffer to prevent contamination
                java.util.Arrays.fill(buffer, 0.0f);
            }
            return buffer;
        } else {
            // New arrays are already zeroed by default
            return new float[bufferSize];
        }
    }

    /**
     * Return a float array to the pool for reuse.
     * Only arrays of the correct size are accepted.
     * 
     * @param buffer the array to return (null-safe, wrong-size arrays ignored)
     */
    public void releaseBuffer(float[] buffer) {
        if (buffer != null && buffer.length == bufferSize) {
            bufferPool.offer(buffer);
        }
    }

    /**
     * Get the buffer size this pool manages.
     * 
     * @return the size of arrays in this pool
     */
    public int getBufferSize() {
        return bufferSize;
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