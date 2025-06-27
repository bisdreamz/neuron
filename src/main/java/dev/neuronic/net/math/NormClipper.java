package dev.neuronic.net.math;

import dev.neuronic.net.math.ops.VectorNorm;

/**
 * L2 norm-based gradient clipper - the most commonly used clipping strategy.
 * 
 * <p><b>Algorithm:</b>
 * <pre>
 * norm = sqrt(sum(gradient[i]^2))
 * if (norm > maxNorm):
 *     scale = maxNorm / norm
 *     gradient[i] *= scale  // for all i
 * </pre>
 * 
 * <p><b>Why Norm Clipping:</b>
 * <ul>
 *   <li>Preserves gradient direction while limiting magnitude</li>
 *   <li>Most effective for preventing exploding gradients in RNNs</li>
 *   <li>Standard in modern deep learning frameworks</li>
 * </ul>
 * 
 * <p><b>Performance:</b> Uses vectorized operations for norm calculation and scaling.
 * Zero-allocation implementation for hot training paths.
 */
final class NormClipper implements GradientClipper {
    
    private final float maxNorm;
    private final float maxNormSquared;
    
    NormClipper(float maxNorm) {
        if (maxNorm <= 0) {
            throw new IllegalArgumentException("Max norm must be positive, got: " + maxNorm);
        }
        this.maxNorm = maxNorm;
        this.maxNormSquared = maxNorm * maxNorm;
    }
    
    @Override
    public boolean clipInPlace(float[] gradients) {
        if (gradients.length == 0) return false;
        
        // Calculate L2 norm using vectorized operations
        float normSquared = VectorNorm.computeL2Squared(gradients);
        
        if (normSquared <= maxNormSquared) {
            return false; // No clipping needed
        }
        
        // Apply clipping: gradients *= (maxNorm / norm)
        float norm = (float) Math.sqrt(normSquared);
        float scale = maxNorm / norm;
        
        NetMath.elementwiseScaleInPlace(gradients, scale);
        return true;
    }
    
    @Override
    public boolean wouldClip(float[] gradients) {
        if (gradients.length == 0) return false;
        return VectorNorm.computeL2Squared(gradients) > maxNormSquared;
    }
    
    @Override
    public String getDescription() {
        return String.format("NormClipper(maxNorm=%.3f)", maxNorm);
    }
    
    /**
     * Get the maximum norm threshold.
     */
    public float getMaxNorm() {
        return maxNorm;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof NormClipper)) return false;
        NormClipper other = (NormClipper) obj;
        return Float.compare(maxNorm, other.maxNorm) == 0;
    }
    
    @Override
    public int hashCode() {
        return Float.hashCode(maxNorm);
    }
    
    @Override
    public String toString() {
        return getDescription();
    }
}