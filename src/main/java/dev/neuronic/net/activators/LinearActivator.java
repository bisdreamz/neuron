package dev.neuronic.net.activators;

/**
 * Linear (identity) activation function: f(x) = x
 * 
 * Use this for:
 * - Output layers that will be processed by loss functions (e.g., logits for CrossEntropy)
 * - Layers where you want raw weighted sums without transformation
 * 
 * The derivative is always 1.0, making gradient flow unchanged.
 */
public final class LinearActivator implements Activator {
    
    public static final LinearActivator INSTANCE = new LinearActivator();
    
    private LinearActivator() {} // Singleton pattern
    
    @Override
    public void activate(float[] input, float[] output) {
        // Identity function: output = input
        System.arraycopy(input, 0, output, 0, input.length);
    }
    
    @Override
    public void derivative(float[] input, float[] output) {
        // Derivative of identity is always 1.0
        for (int i = 0; i < output.length; i++) {
            output[i] = 1.0f;
        }
    }
}