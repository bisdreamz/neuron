package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Central dispatcher for vector operations.
 * This class determines whether to use vector or scalar implementations.
 * 
 * Pattern:
 * 1. Each operation has a scalar implementation in the main class
 * 2. Vector implementations are in the vector subpackage
 * 3. This dispatcher loads vector implementations dynamically
 */
public final class VectorDispatcher {
    
    // Cache of loaded vector implementation classes
    private static final java.util.Map<String, java.lang.reflect.Method> VECTOR_METHODS = new java.util.HashMap<>();
    
    static {
        // Only attempt to load vector implementations if Vector API is available
        if (Vectorization.isAvailable()) {
            // Pre-load common vector operations
            loadVectorMethod("ElementwiseAdd", "compute", float[].class, float[].class, float[].class);
            loadVectorMethod("ElementwiseMultiply", "compute", float[].class, float[].class, float[].class);
            loadVectorMethod("ElementwiseSubtract", "compute", float[].class, float[].class, float[].class);
            loadVectorMethod("ElementwiseScale", "compute", float[].class, float.class, float[].class);
            loadVectorMethod("DotProduct", "compute", float[].class, float[].class);
            // Add more as needed
        }
    }
    
    private static void loadVectorMethod(String className, String methodName, Class<?>... paramTypes) {
        try {
            Class<?> vectorClass = Class.forName(
                "com.nimbus.net.math.ops.vector." + className + "Vector");
            java.lang.reflect.Method method = vectorClass.getMethod(methodName, paramTypes);
            VECTOR_METHODS.put(className + "." + methodName, method);
        } catch (Exception e) {
            // Vector implementation not available for this operation
        }
    }
    
    /**
     * Check if a vector implementation is available for the given operation.
     */
    public static boolean hasVectorImpl(String className, String methodName) {
        return VECTOR_METHODS.containsKey(className + "." + methodName);
    }
    
    /**
     * Invoke a vector operation if available.
     * Returns true if successfully invoked, false if should fall back to scalar.
     */
    public static boolean invokeVector(String className, String methodName, Object... args) {
        java.lang.reflect.Method method = VECTOR_METHODS.get(className + "." + methodName);
        if (method != null) {
            try {
                method.invoke(null, args);
                return true;
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        return false;
    }
    
    /**
     * Invoke a vector operation that returns a value.
     */
    @SuppressWarnings("unchecked")
    public static <T> T invokeVectorWithReturn(String className, String methodName, Object... args) {
        java.lang.reflect.Method method = VECTOR_METHODS.get(className + "." + methodName);
        if (method != null) {
            try {
                return (T) method.invoke(null, args);
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        return null;
    }
    
    private VectorDispatcher() {} // Prevent instantiation
}