package dev.neuronic.net.math;

/**
 * No-op gradient clipper that never clips gradients.
 * 
 * <p><b>Use Cases:</b>
 * <ul>
 *   <li>Disabling clipping conditionally without changing code structure</li>
 *   <li>Default clipper when no clipping is desired</li>
 *   <li>Baseline for comparing clipping strategies</li>
 * </ul>
 * 
 * <p><b>Performance:</b> Zero overhead - all operations return immediately.
 */
final class NoOpClipper implements GradientClipper {
    
    static final NoOpClipper INSTANCE = new NoOpClipper();
    
    private NoOpClipper() {
        // Singleton
    }
    
    @Override
    public boolean clipInPlace(float[] gradients) {
        return false; // Never clips
    }
    
    @Override
    public boolean wouldClip(float[] gradients) {
        return false; // Would never clip
    }
    
    @Override
    public String getDescription() {
        return "NoOpClipper(disabled)";
    }
    
    @Override
    public String toString() {
        return getDescription();
    }
}