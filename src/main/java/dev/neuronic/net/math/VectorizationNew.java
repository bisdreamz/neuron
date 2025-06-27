package dev.neuronic.net.math;

/**
 * Central configuration for vector/SIMD support using provider pattern.
 * This class has NO Vector API imports and can always be loaded.
 */
public final class VectorizationNew {
    
    private static final VectorProvider PROVIDER;
    private static final boolean AVAILABLE;
    private static final int VECTOR_LENGTH;
    
    static {
        VectorProvider provider = null;
        boolean available = false;
        
        try {
            // Try to load Vector API
            Class.forName("jdk.incubator.vector.FloatVector");
            
            // If successful, load the vector provider via reflection
            Class<?> implClass = Class.forName("dev.neuronic.net.math.VectorProviderImpl");
            provider = (VectorProvider) implClass.getMethod("getInstance").invoke(null);
            available = true;
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            // Vector API not available - use scalar fallback
        } catch (Exception e) {
            System.err.println("Failed to initialize Vector provider: " + e.getMessage());
        }
        
        // Fall back to scalar if vector not available
        if (provider == null) {
            provider = ScalarProvider.getInstance();
        }
        
        PROVIDER = provider;
        AVAILABLE = available;
        VECTOR_LENGTH = provider.getVectorLength();
        
        if (available) {
            System.out.println("VECTORIZATION ENABLED: Vector API with " + VECTOR_LENGTH + " lanes");
        } else {
            System.out.println("VECTORIZATION NOT ENABLED: Using scalar operations");
        }
    }
    
    /**
     * Get the vector provider for operations.
     */
    public static VectorProvider getProvider() {
        return PROVIDER;
    }
    
    /**
     * Check if Vector API is available.
     */
    public static boolean isAvailable() {
        return AVAILABLE;
    }
    
    /**
     * Get the vector length.
     */
    public static int getVectorLength() {
        return VECTOR_LENGTH;
    }
    
    /**
     * Get loop bound for vectorized operations.
     */
    public static int loopBound(int length) {
        return PROVIDER.getLoopBound(length);
    }
    
    /**
     * Check if vectorization is worthwhile for the given length.
     */
    public static boolean shouldVectorize(int length) {
        return AVAILABLE && length >= VECTOR_LENGTH * 2;
    }
    
    private VectorizationNew() {} // Prevent instantiation
}