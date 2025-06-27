package dev.neuronic.net.losses;

import dev.neuronic.net.activators.Activator;

/**
 * Marker interface for loss functions that handle their own activation derivatives.
 * 
 * When a loss function implements this interface, it indicates that:
 * 1) The loss function computes the combined derivative of loss + activation
 * 2) The final layer should NOT apply activation derivatives separately
 * 3) The gradient from derivatives() is ready to use directly
 * 
 * This prevents double-application of derivatives for cases like Softmax + CrossEntropy.
 */
public interface CombinedLossActivation {
    
    /**
     * Returns the activator that this loss function handles derivatives for.
     * The final layer will check this to skip separate activation derivative computation.
     */
    Class<? extends Activator> getHandledActivator();
    
    /**
     * Compute the pre-activation values (logits) before applying activation.
     * This allows the loss to work directly with logits for numerical stability.
     */
    default boolean requiresPreActivation() {
        return false;
    }
}