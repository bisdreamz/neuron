package dev.neuronic.net.activators;

import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.math.Parallelization;
import dev.neuronic.net.math.ops.LeakyReLU;

import java.util.concurrent.ExecutorService;

/**
 * Leaky Rectified Linear Unit (Leaky ReLU) activation function.
 *
 * <p>LeakyReLU(x) = x if x > 0, else alpha * x
 *
 * <p>A variant of ReLU that allows a small gradient when the unit is not active.
 * This helps avoid "dying ReLU" problems where neurons become permanently inactive.
 *
 * <p>Derivative: 1 if x > 0, else alpha
 */
public final class LeakyReluActivator implements Activator {

    private final float alpha;
    
    /**
     * Create a Leaky ReLU with custom alpha value.
     * @param alpha the slope for negative inputs (typically 0.01 to 0.3)
     */
    public LeakyReluActivator(float alpha) {
        if (alpha <= 0 || alpha >= 1)
            throw new IllegalArgumentException("Alpha must be between 0 and 1, got: " + alpha);
        
        this.alpha = alpha;
    }
    
    /**
     * Get the alpha parameter value.
     */
    public float getAlpha() {
        return alpha;
    }
    
    /**
     * Create a Leaky ReLU with default alpha = 0.01
     */
    public static LeakyReluActivator createDefault() {
        return new LeakyReluActivator(0.01f);
    }
    
    /**
     * Create a Leaky ReLU with custom alpha
     */
    public static LeakyReluActivator create(float alpha) {
        return new LeakyReluActivator(alpha);
    }

    @Override
    public void activate(float[] input, float[] output) {
        checkLength(input, output);
        NetMath.activationLeakyRelu(input, alpha, output);
    }

    @Override
    public void derivative(float[] input, float[] output) {
        checkLength(input, output);
        NetMath.activationLeakyReluDerivative(input, alpha, output);
    }

    private static void checkLength(float[] input, float[] output) {
        if (input.length != output.length)
            throw new IllegalArgumentException(
                    "Input and output arrays must have same length: " +
                            "input=" + input.length + ", output=" + output.length
            );
    }
    
    @Override
    public void activate(float[] input, float[] output, ExecutorService executor) {
        if (Parallelization.shouldParallelize(input.length, executor))
            activateParallel(input, output, executor);
        else
            activate(input, output);
    }
    
    @Override
    public void derivative(float[] input, float[] output, ExecutorService executor) {
        if (Parallelization.shouldParallelize(input.length, executor))
            derivativeParallel(input, output, executor);
        else
            derivative(input, output);
    }
    
    private void activateParallel(float[] input, float[] output, ExecutorService executor) {
        checkLength(input, output);
        
        int numThreads = Parallelization.calculateOptimalThreads(input.length, executor);
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(input.length, numThreads);
        
        Runnable[] tasks = new Runnable[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            final Parallelization.WorkRange range = ranges[i];
            tasks[i] = () -> LeakyReLU.activateRange(input, alpha, output, range.start, range.end);
        }
        
        Parallelization.executeParallel(executor, tasks);
    }
    
    private void derivativeParallel(float[] input, float[] output, ExecutorService executor) {
        checkLength(input, output);
        
        int numThreads = Parallelization.calculateOptimalThreads(input.length, executor);
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(input.length, numThreads);
        
        Runnable[] tasks = new Runnable[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            final Parallelization.WorkRange range = ranges[i];
            tasks[i] = () -> LeakyReLU.derivativeRange(input, alpha, output, range.start, range.end);
        }
        
        Parallelization.executeParallel(executor, tasks);
    }
}