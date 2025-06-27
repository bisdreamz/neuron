package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Huber loss implementation - a robust loss function that combines MSE and MAE.
 * 
 * <p>For |error| <= delta: loss = 0.5 * error^2 (quadratic, like MSE)
 * <p>For |error| > delta: loss = delta * |error| - 0.5 * delta^2 (linear, like MAE)
 * 
 * <p>This makes the loss less sensitive to outliers than MSE while still being
 * differentiable everywhere (unlike MAE which has undefined gradient at 0).
 * 
 * <p>Derivative:
 * <ul>
 *   <li>For |error| <= delta: error
 *   <li>For |error| > delta: delta * sign(error)
 * </ul>
 */
public final class HuberLoss {
    
    public interface Impl {
        float computeLoss(float[] predictions, float[] targets, float delta);
        void computeDerivatives(float[] predictions, float[] targets, float delta, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public float computeLoss(float[] predictions, float[] targets, float delta) {
            float sum = 0.0f;
            float halfDeltaSq = 0.5f * delta * delta;
            
            for (int i = 0; i < predictions.length; i++) {
                float error = predictions[i] - targets[i];
                float absError = Math.abs(error);
                
                if (absError <= delta)
                    sum += 0.5f * error * error;
                else
                    sum += delta * absError - halfDeltaSq;
            }
            
            return sum / predictions.length;
        }
        
        @Override
        public void computeDerivatives(float[] predictions, float[] targets, float delta, float[] output) {
            for (int i = 0; i < predictions.length; i++) {
                float error = predictions[i] - targets[i];
                
                if (Math.abs(error) <= delta)
                    output[i] = error;
                else
                    output[i] = error > 0 ? delta : -delta;
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.HuberLossVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute Huber loss for a batch of predictions.
     * 
     * @param predictions predicted values
     * @param targets true values
     * @param delta threshold parameter (typically 1.0)
     * @return average Huber loss
     */
    public static float computeLoss(float[] predictions, float[] targets, float delta) {
        return IMPL.computeLoss(predictions, targets, delta);
    }
    
    /**
     * Compute Huber loss derivatives.
     * 
     * @param predictions predicted values
     * @param targets true values
     * @param delta threshold parameter
     * @param output pre-allocated array for derivatives
     */
    public static void computeDerivatives(float[] predictions, float[] targets, float delta, float[] output) {
        IMPL.computeDerivatives(predictions, targets, delta, output);
    }
    
    private HuberLoss() {}
}