package dev.neuronic.net.losses;

import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.activators.Activator;
import dev.neuronic.net.activators.SoftmaxActivator;

/**
 * Cross-Entropy loss function for multi-class classification.
 * 
 * <p>Loss: CrossEntropy = -sum(trueLabels[i] * log(predictions[i]))
 * <p>Gradient: dCrossEntropy/dpredictions[i] = predictions[i] - trueLabels[i]
 * 
 * <p><strong>When to Use Cross-Entropy:</strong>
 * <ul>
 * <li><strong>Multi-class classification</strong> - MNIST, CIFAR, ImageNet</li>
 * <li><strong>With Softmax activation</strong> - optimal mathematical pairing</li>
 * <li><strong>One-hot encoded labels</strong> - mutually exclusive classes</li>
 * <li><strong>Probability outputs</strong> - when model outputs probabilities</li>
 * </ul>
 * 
 * <p><strong>When NOT to Use Cross-Entropy:</strong>
 * <ul>
 * <li><strong>Regression tasks</strong> - use MSE instead</li>
 * <li><strong>Binary classification</strong> - use Binary Cross-Entropy instead</li>
 * <li><strong>Multi-label classification</strong> - use Binary Cross-Entropy for each label</li>
 * <li><strong>Non-probability outputs</strong> - ensure outputs are normalized probabilities</li>
 * </ul>
 * 
 * <p><strong>Example Usage:</strong>
 * <pre>
 * // MNIST digit classification
 * float[] trueLabels = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}; // One-hot for digit "2"
 * float[] predictions = {0.1, 0.1, 0.7, 0.05, 0.05}; // Softmax output
 * float loss = CrossEntropyLoss.INSTANCE.loss(predictions, trueLabels);
 * // loss = -log(0.7) ≈ 0.357
 * </pre>
 * 
 * <p><strong>Mathematical Properties:</strong>
 * <ul>
 * <li>Heavily penalizes confident wrong predictions</li>
 * <li>Gradient simplifies beautifully when paired with Softmax</li>
 * <li>Encourages probability calibration (confident when correct)</li>
 * <li>Convex loss function (single global minimum)</li>
 * </ul>
 */
public final class CrossEntropyLoss implements Loss, CombinedLossActivation {
    
    public static final CrossEntropyLoss INSTANCE = new CrossEntropyLoss();
    
    private CrossEntropyLoss() {} // Private constructor for singleton
    
    @Override
    public float loss(float[] prediction, float[] labels) {
        return NetMath.lossComputeCrossEntropy(labels, prediction);
    }
    
    @Override
    public float[] derivatives(float[] prediction, float[] labels) {
        float[] derivatives = new float[prediction.length];
        
        NetMath.lossGradientCrossEntropy(labels, prediction, derivatives);
        
        return derivatives;
    }
    
    @Override
    public Class<? extends Activator> getHandledActivator() {
        return SoftmaxActivator.class;
    }
}