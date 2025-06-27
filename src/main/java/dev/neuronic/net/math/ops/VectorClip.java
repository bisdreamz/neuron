package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Vectorized clipping operations for gradient clipping.
 * 
 * <p><b>Performance:</b> Uses SIMD instructions when available for optimal speed.
 * Falls back to scalar operations when vectorization is not beneficial.
 */
public final class VectorClip {
    
    public interface Impl {
        boolean clipInPlace(float[] array, float minValue, float maxValue);
        boolean wouldClip(float[] array, float minValue, float maxValue);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public boolean clipInPlace(float[] array, float minValue, float maxValue) {
            boolean clipped = false;
            
            for (int i = 0; i < array.length; i++) {
                float original = array[i];
                if (original < minValue) {
                    array[i] = minValue;
                    clipped = true;
                } else if (original > maxValue) {
                    array[i] = maxValue;
                    clipped = true;
                }
            }
            
            return clipped;
        }
        
        @Override
        public boolean wouldClip(float[] array, float minValue, float maxValue) {
            for (float value : array) {
                if (value < minValue || value > maxValue) {
                    return true;
                }
            }
            return false;
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.VectorClipVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Clip array values to [minValue, maxValue] range in-place.
     * 
     * @param array the array to clip (modified in-place)
     * @param minValue minimum allowed value
     * @param maxValue maximum allowed value
     * @return true if any values were clipped
     */
    public static boolean clipInPlace(float[] array, float minValue, float maxValue) {
        if (array.length == 0) return false;
        
        return IMPL.clipInPlace(array, minValue, maxValue);
    }
    
    /**
     * Check if any values in array would be clipped without modifying the array.
     * 
     * @param array the array to check
     * @param minValue minimum allowed value
     * @param maxValue maximum allowed value
     * @return true if any values are outside [minValue, maxValue]
     */
    public static boolean wouldClip(float[] array, float minValue, float maxValue) {
        if (array.length == 0) return false;
        
        return IMPL.wouldClip(array, minValue, maxValue);
    }
    
    static boolean clipVectorized(float[] array, float minValue, float maxValue) {
        return IMPL.clipInPlace(array, minValue, maxValue);
    }
    
    static boolean clipScalar(float[] array, float minValue, float maxValue) {
        return new ScalarImpl().clipInPlace(array, minValue, maxValue);
    }
    
    static boolean wouldClipVectorized(float[] array, float minValue, float maxValue) {
        return IMPL.wouldClip(array, minValue, maxValue);
    }
    
    static boolean wouldClipScalar(float[] array, float minValue, float maxValue) {
        return new ScalarImpl().wouldClip(array, minValue, maxValue);
    }
    
    private VectorClip() {}
}