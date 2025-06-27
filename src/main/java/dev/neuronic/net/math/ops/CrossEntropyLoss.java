package dev.neuronic.net.math.ops;

import dev.neuronic.net.math.Vectorization;

/**
 * Cross-entropy loss computation for multi-class classification.
 * 
 * <p>CrossEntropy(y_true, y_pred) = -sum(y_true * log(y_pred))
 * 
 * <p>For one-hot encoded labels, this simplifies to: -log(y_pred[true_class])
 * 
 * <p>Cross-entropy loss is the standard loss function for multi-class classification
 * when paired with softmax activation. It penalizes predictions that are confident
 * but wrong more heavily than uncertain predictions.
 */
public final class CrossEntropyLoss {
    
    public interface Impl {
        float compute(float[] trueLabels, float[] predictions);
        void gradient(float[] trueLabels, float[] predictions, float[] output);
    }
    
    private static final class ScalarImpl implements Impl {
        @Override
        public float compute(float[] trueLabels, float[] predictions) {
            float sum = 0.0f;
            for (int i = 0; i < trueLabels.length; i++) {
                float pred = Math.max(predictions[i], 1e-7f);
                sum += trueLabels[i] * Math.log(pred);
            }
            return -sum;
        }
        
        @Override
        public void gradient(float[] trueLabels, float[] predictions, float[] output) {
            for (int i = 0; i < trueLabels.length; i++) {
                output[i] = predictions[i] - trueLabels[i];
            }
        }
    }
    
    private static final Impl IMPL;
    
    static {
        Impl impl = null;
        if (Vectorization.isAvailable()) {
            try {
                Class<?> vectorClass = Class.forName(
                        "dev.neuronic.net.math.ops.vector.CrossEntropyLossVector");
                impl = (Impl) vectorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                // Fall back to scalar
            }
        }
        IMPL = (impl != null) ? impl : new ScalarImpl();
    }
    
    private CrossEntropyLoss() {}
    
    /**
     * Compute cross-entropy loss between true labels and predictions.
     * 
     * @param trueLabels one-hot encoded true labels
     * @param predictions predicted probabilities (should sum to 1.0)
     * @return the cross-entropy loss value
     */
    public static float compute(float[] trueLabels, float[] predictions) {
        if (trueLabels.length != predictions.length)
            throw new IllegalArgumentException(
                "True labels and predictions must have same length: " +
                "labels=" + trueLabels.length + ", predictions=" + predictions.length
            );
        
        return IMPL.compute(trueLabels, predictions);
    }
    
    /**
     * Compute cross-entropy loss gradient with respect to predictions.
     * 
     * <p>Gradient: (predictions - trueLabels) / batchSize
     * 
     * <p>Note: This assumes predictions come from softmax, where the gradient
     * simplifies to this form when combined with cross-entropy loss.
     * 
     * @param trueLabels one-hot encoded true labels
     * @param predictions predicted probabilities
     * @param output output array for gradients
     */
    public static void gradient(float[] trueLabels, float[] predictions, float[] output) {
        if (trueLabels.length != predictions.length || predictions.length != output.length)
            throw new IllegalArgumentException(
                "All arrays must have same length: labels=" + trueLabels.length + 
                ", predictions=" + predictions.length + ", output=" + output.length
            );
        
        IMPL.gradient(trueLabels, predictions, output);
    }
    
    static float computeVectorized(float[] trueLabels, float[] predictions) {
        return IMPL.compute(trueLabels, predictions);
    }
    
    static float computeScalar(float[] trueLabels, float[] predictions) {
        return new ScalarImpl().compute(trueLabels, predictions);
    }
    
    static void gradientVectorized(float[] trueLabels, float[] predictions, float[] output) {
        IMPL.gradient(trueLabels, predictions, output);
    }
    
    static void gradientScalar(float[] trueLabels, float[] predictions, float[] output) {
        new ScalarImpl().gradient(trueLabels, predictions, output);
    }
}