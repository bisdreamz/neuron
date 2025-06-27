package dev.neuronic.net.math;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Central configuration for vector/SIMD support.
 * Checks availability once at startup and provides shared configuration.
 */
public final class Vectorization {
    
    private static final VectorSpecies<Float> SPECIES;
    private static final boolean AVAILABLE;
    private static final int VECTOR_LENGTH;
    private static int DEFAULT_VECTORIZATION_LIMIT = 8; // Default limit for shouldVectorizeLimited
    
    static {
        boolean available = false;
        VectorSpecies<Float> species = null;
        
        try {
            // Try to load Vector API and get preferred species
            Class.forName("jdk.incubator.vector.FloatVector");
            species = FloatVector.SPECIES_PREFERRED;
            available = true;
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            // Vector API not available
        }
        
        AVAILABLE = available;
        SPECIES = species;
        VECTOR_LENGTH = available ? species.length() : 1;
        
        if (available) {
            System.out.println("Vector API enabled with " + VECTOR_LENGTH + " lanes");
        } else {
            System.out.println("Vector API not available, using scalar operations");
        }
    }
    
    /**
     * @return true if Vector API is available on this platform
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }
    
    /**
     * @return the preferred vector species for float operations, or null if unavailable
     */
    public static VectorSpecies<Float> getSpecies() {
        return SPECIES;
    }
    
    /**
     * @return number of float lanes in the vector, or 1 if using scalar operations
     */
    public static int getVectorLength() {
        return VECTOR_LENGTH;
    }
    
    /**
     * @return the loop bound for vectorized operations (aligned to vector length)
     */
    public static int loopBound(int length) {
        return AVAILABLE && SPECIES != null ? SPECIES.loopBound(length) : 0;
    }
    
    /**
     * Check if the given length is worth vectorizing.
     * Small arrays may be faster with scalar operations due to overhead.
     * 
     * @param length array length to check
     * @return true if vectorization is recommended
     */
    public static boolean shouldVectorize(int length) {
        return AVAILABLE && length >= VECTOR_LENGTH * 2;
    }

    /**
     * Vectorize only if the length is between the minimum and provided ceiling limit
     * for ops that dont benefit from vector api during larger inputs
     * @param length length of array to check
     * @param limit limit of float lanes to limit passing criteria to
     * @return true if passes shouldVectorize and length <= limit * vectorLength
     */
    public static boolean shouldVectorizeLimited(int length, int limit) {
        return shouldVectorize(length) && length <= VECTOR_LENGTH * limit;
    }
    
    /**
     * Vectorize only if the length is between the minimum and the default ceiling limit.
     * Uses the configurable default limit for operations that benefit from limited vectorization.
     * @param length length of array to check
     * @return true if passes shouldVectorize and length <= defaultLimit * vectorLength
     */
    public static boolean shouldVectorizeLimited(int length) {
        return shouldVectorizeLimited(length, DEFAULT_VECTORIZATION_LIMIT);
    }
    
    /**
     * Get the default vectorization limit multiplier.
     * @return current default limit (array length threshold = limit * vectorLength)
     */
    public static int getDefaultVectorizationLimit() {
        return DEFAULT_VECTORIZATION_LIMIT;
    }
    
    /**
     * Set the default vectorization limit multiplier for shouldVectorizeLimited(length).
     * This affects operations that benefit from limited vectorization on small arrays.
     * @param limit new default limit (array length threshold = limit * vectorLength)
     */
    public static void setDefaultVectorizationLimit(int limit) {
        if (limit <= 0) {
            throw new IllegalArgumentException("Vectorization limit must be positive: " + limit);
        }
        DEFAULT_VECTORIZATION_LIMIT = limit;
    }
    
    /**
     * Get the optimal buffer size that is aligned to vector boundaries.
     * This ensures efficient vectorized operations by returning a size that's
     * a multiple of the vector length.
     * 
     * @param minSize minimum required size
     * @return size that is >= minSize and a multiple of vector length
     */
    public static int getOptimalBufferSize(int minSize) {
        if (!AVAILABLE || VECTOR_LENGTH <= 1) {
            return minSize;
        }
        // Round up to next multiple of vector length
        return ((minSize + VECTOR_LENGTH - 1) / VECTOR_LENGTH) * VECTOR_LENGTH;
    }
    
    /**
     * Check if an array is aligned for optimal vectorization.
     * Arrays allocated with getOptimalBufferSize() will always be aligned.
     * 
     * @param array the array to check
     * @return true if the array length is a multiple of vector length
     */
    public static boolean isAligned(float[] array) {
        if (!AVAILABLE || VECTOR_LENGTH <= 1) {
            return true;
        }
        return array.length % VECTOR_LENGTH == 0;
    }
    
    /**
     * Get the optimal batch size for parallel operations.
     * Returns a batch size that's both efficient for vectorization
     * and work distribution across threads.
     * 
     * @param suggestedSize the suggested batch size
     * @return optimized batch size (multiple of vector length)
     */
    public static int getOptimalBatchSize(int suggestedSize) {
        if (!AVAILABLE || VECTOR_LENGTH <= 1) {
            return suggestedSize;
        }
        // Ensure batch size is at least 4x vector length for efficiency
        int minBatch = VECTOR_LENGTH * 4;
        if (suggestedSize < minBatch) {
            return minBatch;
        }
        return getOptimalBufferSize(suggestedSize);
    }
    
    private Vectorization() {} // Prevent instantiation
}