package dev.neuronic.net.losses;

/**
 * Sparse Cross-Entropy loss for language models and other high-vocabulary tasks.
 * Accepts sparse targets (single integer index) instead of one-hot vectors.
 * 
 * <p>Memory efficient for large vocabularies:
 * - Standard: O(vocab_size) per example
 * - Sparse: O(1) per example
 */
public final class SparseCrossEntropyLoss implements Loss {
    
    public static final SparseCrossEntropyLoss INSTANCE = new SparseCrossEntropyLoss();
    
    private SparseCrossEntropyLoss() {} // Singleton
    
    @Override
    public float loss(float[] prediction, float[] labels) {
        // Check if this is a sparse target (single element = token index)
        if (labels.length == 1) {
            int targetIndex = (int) labels[0];
            if (targetIndex < 0 || targetIndex >= prediction.length) {
                throw new IllegalArgumentException(
                    "Target index " + targetIndex + " out of bounds [0, " + prediction.length + ")");
            }
            // Cross-entropy = -log(p[target])
            float prob = prediction[targetIndex];
            // Clamp to avoid log(0)
            prob = Math.max(prob, 1e-7f);
            return (float) -Math.log(prob);
        } else {
            // Fall back to dense cross-entropy for compatibility
            return CrossEntropyLoss.INSTANCE.loss(prediction, labels);
        }
    }
    
    @Override
    public float[] derivatives(float[] prediction, float[] labels) {
        float[] derivatives = new float[prediction.length];
        
        // Check if this is a sparse target
        if (labels.length == 1) {
            int targetIndex = (int) labels[0];
            if (targetIndex < 0 || targetIndex >= prediction.length) {
                throw new IllegalArgumentException(
                    "Target index " + targetIndex + " out of bounds [0, " + prediction.length + ")");
            }
            // For softmax + cross-entropy: gradient = predictions - one_hot(target)
            System.arraycopy(prediction, 0, derivatives, 0, prediction.length);
            derivatives[targetIndex] -= 1.0f;
        } else {
            // Fall back to dense cross-entropy
            return CrossEntropyLoss.INSTANCE.derivatives(prediction, labels);
        }
        
        return derivatives;
    }
}