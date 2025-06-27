package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Mean Squared Error loss computation operations.
 * 
 * <p>Loss: MSE = (1/n) * sum((prediction[i] - target[i])^2)
 * <p>Derivative: dMSE/dprediction[i] = (2/n) * (prediction[i] - target[i])
 */
public final class MeanSquaredErrorLoss {
    
    public interface Impl {
        float computeLoss(float[] predictions, float[] targets);
        void computeDerivatives(float[] predictions, float[] targets, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public float computeLoss(float[] predictions, float[] targets) {
            float sum = 0.0f;
            for (int i = 0; i < predictions.length; i++) {
                float diff = predictions[i] - targets[i];
                sum += diff * diff;
            }
            return sum / predictions.length;
        }
        
        @Override
        public void computeDerivatives(float[] predictions, float[] targets, float[] output) {
            float scale = 2.0f / predictions.length;
            for (int i = 0; i < predictions.length; i++) {
                output[i] = scale * (predictions[i] - targets[i]);
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.MeanSquaredErrorLossVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    /**
     * Compute MSE loss between predictions and targets.
     * 
     * @param predictions predicted values
     * @param targets target values
     * @return MSE loss value
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static float computeLoss(float[] predictions, float[] targets) {
        if (predictions.length != targets.length)
            throw new IllegalArgumentException("Predictions and targets must have same length");
        
        return IMPL.computeLoss(predictions, targets);
    }
    
    /**
     * Compute MSE loss derivatives (gradients) w.r.t. predictions.
     * 
     * @param predictions predicted values
     * @param targets target values
     * @param output pre-allocated output buffer for derivatives
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static void computeDerivatives(float[] predictions, float[] targets, float[] output) {
        if (predictions.length != targets.length || predictions.length != output.length)
            throw new IllegalArgumentException("All arrays must have same length");
        
        IMPL.computeDerivatives(predictions, targets, output);
    }
    
    static float computeLossVectorized(float[] predictions, float[] targets) {
        return IMPL.computeLoss(predictions, targets);
    }
    
    static float computeLossScalar(float[] predictions, float[] targets) {
        return new ScalarImpl().computeLoss(predictions, targets);
    }
    
    static void computeDerivativesVectorized(float[] predictions, float[] targets, float[] output) {
        IMPL.computeDerivatives(predictions, targets, output);
    }
    
    static void computeDerivativesScalar(float[] predictions, float[] targets, float[] output) {
        new ScalarImpl().computeDerivatives(predictions, targets, output);
    }
    
    private MeanSquaredErrorLoss() {}
}