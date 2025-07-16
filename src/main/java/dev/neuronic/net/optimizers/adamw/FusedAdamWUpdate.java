package dev.neuronic.net.optimizers.adamw;

import dev.neuronic.net.math.Vectorization;

/**
 * Fused AdamW update operation that combines all steps into a single pass.
 * 
 * This dramatically reduces memory bandwidth usage by fusing:
 * 1. Momentum EMA update: m = β₁ * m + (1 - β₁) * g
 * 2. Velocity EMA update: v = β₂ * v + (1 - β₂) * g²
 * 3. Bias correction: m̂ = m / (1 - β₁^t), v̂ = v / (1 - β₂^t)
 * 4. Parameter update: p = p - α * m̂ / (√v̂ + ε)
 * 5. Weight decay: p = p * (1 - λ)
 * 
 * All operations are performed in a single pass over the data.
 */
public final class FusedAdamWUpdate {
    
    public interface Impl {
        void compute(float[] params, float[] gradients, float[] momentum, float[] velocity,
                    float beta1, float beta2, float learningRate, float epsilon,
                    float weightDecay, float momentumCorrection, float velocityCorrection,
                    boolean applyWeightDecay);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public void compute(float[] params, float[] gradients, float[] momentum, float[] velocity,
                          float beta1, float beta2, float learningRate, float epsilon,
                          float weightDecay, float momentumCorrection, float velocityCorrection,
                          boolean applyWeightDecay) {
            
            float oneMinusBeta1 = 1.0f - beta1;
            float oneMinusBeta2 = 1.0f - beta2;
            
            for (int i = 0; i < params.length; i++) {
                float grad = gradients[i];
                
                // Update momentum: m = β₁ * m + (1 - β₁) * g
                momentum[i] = beta1 * momentum[i] + oneMinusBeta1 * grad;
                
                // Update velocity: v = β₂ * v + (1 - β₂) * g²
                velocity[i] = beta2 * velocity[i] + oneMinusBeta2 * grad * grad;
                
                // Bias correction
                float mHat = momentum[i] / momentumCorrection;
                float vHat = velocity[i] / velocityCorrection;
                
                // Parameter update: p = p - α * m̂ / (√v̂ + ε)
                params[i] -= learningRate * mHat / (float)(Math.sqrt(vHat) + epsilon);
                
                // Weight decay: p = p * (1 - α * λ)
                if (applyWeightDecay) {
                    params[i] *= (1.0f - learningRate * weightDecay);
                }
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.optimizers.adamw.vector.FusedAdamWUpdateVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    private FusedAdamWUpdate() {}
    
    /**
     * Perform fused AdamW update in a single pass.
     * 
     * @param params parameter array to update (modified in-place)
     * @param gradients gradient array
     * @param momentum momentum state array (modified in-place)
     * @param velocity velocity state array (modified in-place)
     * @param beta1 momentum decay rate
     * @param beta2 velocity decay rate
     * @param learningRate learning rate
     * @param epsilon small constant for numerical stability
     * @param weightDecay weight decay coefficient
     * @param momentumCorrection bias correction for momentum (1 - beta1^t)
     * @param velocityCorrection bias correction for velocity (1 - beta2^t)
     * @param applyWeightDecay whether to apply weight decay
     */
    public static void compute(float[] params, float[] gradients, float[] momentum, float[] velocity,
                             float beta1, float beta2, float learningRate, float epsilon,
                             float weightDecay, float momentumCorrection, float velocityCorrection,
                             boolean applyWeightDecay) {
        if (params.length != gradients.length || params.length != momentum.length || params.length != velocity.length) {
            throw new IllegalArgumentException("All arrays must have same length");
        }
        
        IMPL.compute(params, gradients, momentum, velocity, beta1, beta2, learningRate, epsilon,
                    weightDecay, momentumCorrection, velocityCorrection, applyWeightDecay);
    }
    
    static void computeVectorized(float[] params, float[] gradients, float[] momentum, float[] velocity,
                                float beta1, float beta2, float learningRate, float epsilon,
                                float weightDecay, float momentumCorrection, float velocityCorrection,
                                boolean applyWeightDecay) {
        IMPL.compute(params, gradients, momentum, velocity, beta1, beta2, learningRate, epsilon,
                    weightDecay, momentumCorrection, velocityCorrection, applyWeightDecay);
    }
    
    static void computeScalar(float[] params, float[] gradients, float[] momentum, float[] velocity,
                            float beta1, float beta2, float learningRate, float epsilon,
                            float weightDecay, float momentumCorrection, float velocityCorrection,
                            boolean applyWeightDecay) {
        new ScalarImpl().compute(params, gradients, momentum, velocity, beta1, beta2, learningRate, epsilon,
                               weightDecay, momentumCorrection, velocityCorrection, applyWeightDecay);
    }
}