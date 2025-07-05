package dev.neuronic.net.optimizers.adamw;

/**
 * Numerically stable bias correction for Adam/AdamW optimizers.
 * 
 * Prevents underflow issues when computing 1 - beta^t for large t.
 */
public final class BiasCorrection {
    
    /**
     * Compute numerically stable bias correction factor for Adam.
     * 
     * Standard formula: 1 / (1 - beta^t)
     * Problem: beta^t underflows to 0 for large t, making correction = 1.0
     * 
     * Solution: Use logarithms to avoid underflow
     * - If t * log(beta) > -10, use standard formula
     * - Otherwise, use approximation: correction ≈ 1.0 (since 1 - beta^t ≈ 1)
     * 
     * @param beta the decay rate (0 < beta < 1)
     * @param timeStep the current time step (t >= 1)
     * @return the bias correction factor
     */
    public static float computeCorrectionFactor(float beta, long timeStep) {
        if (timeStep <= 0) {
            throw new IllegalArgumentException("Time step must be positive: " + timeStep);
        }
        
        // For small time steps, use standard formula
        if (timeStep < 100) {
            return 1.0f / (1.0f - (float)Math.pow(beta, timeStep));
        }
        
        // For larger time steps, check if beta^t would underflow
        double logBeta = Math.log(beta);
        double tLogBeta = timeStep * logBeta;
        
        // If beta^t > 1e-7, we can safely compute it
        if (tLogBeta > -16.12) { // log(1e-7) ≈ -16.12
            double betaPowT = Math.exp(tLogBeta);
            return (float)(1.0 / (1.0 - betaPowT));
        }
        
        // For very large t where beta^t ≈ 0, the correction factor approaches 1.0
        // This is mathematically correct: lim(t→∞) 1/(1-beta^t) = 1/(1-0) = 1
        return 1.0f;
    }
    
    /**
     * Compute both momentum and velocity correction factors efficiently.
     * 
     * @param beta1 momentum decay rate
     * @param beta2 velocity decay rate  
     * @param timeStep current time step
     * @return array of [momentumCorrection, velocityCorrection]
     */
    public static float[] computeBothCorrectionFactors(float beta1, float beta2, long timeStep) {
        return new float[] {
            computeCorrectionFactor(beta1, timeStep),
            computeCorrectionFactor(beta2, timeStep)
        };
    }
}