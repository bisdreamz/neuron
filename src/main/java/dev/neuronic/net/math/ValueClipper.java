package dev.neuronic.net.math;

import dev.neuronic.net.math.ops.VectorClip;

/**
 * Value-based gradient clipper that clips individual gradient values.
 * 
 * <p><b>Algorithm:</b> Clip each gradient value to [-maxValue, maxValue] range.
 * 
 * <p><b>Use Cases:</b>
 * <ul>
 *   <li>Simple gradient limiting when norm clipping is too aggressive</li>
 *   <li>Preventing individual parameters from receiving extreme updates</li>
 *   <li>Debugging gradient issues by limiting extreme values</li>
 * </ul>
 * 
 * <p><b>Performance:</b> Uses vectorized clipping operations following Vectorization guidelines.
 * Automatically selects vectorized or scalar implementation based on array size.
 */
final class ValueClipper implements GradientClipper {
    
    private final float maxValue;
    private final float minValue;
    
    ValueClipper(float maxValue) {
        if (maxValue <= 0) {
            throw new IllegalArgumentException("Max value must be positive, got: " + maxValue);
        }
        this.maxValue = maxValue;
        this.minValue = -maxValue;
    }
    
    @Override
    public boolean clipInPlace(float[] gradients) {
        return VectorClip.clipInPlace(gradients, minValue, maxValue);
    }
    
    @Override
    public boolean wouldClip(float[] gradients) {
        return VectorClip.wouldClip(gradients, minValue, maxValue);
    }
    
    @Override
    public String getDescription() {
        return String.format("ValueClipper(range=[%.3f, %.3f])", minValue, maxValue);
    }
    
    /**
     * Get the maximum absolute value threshold.
     */
    public float getMaxValue() {
        return maxValue;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof ValueClipper)) return false;
        ValueClipper other = (ValueClipper) obj;
        return Float.compare(maxValue, other.maxValue) == 0;
    }
    
    @Override
    public int hashCode() {
        return Float.hashCode(maxValue);
    }
    
    @Override
    public String toString() {
        return getDescription();
    }
}