package dev.neuronic.net.activators;

import dev.neuronic.net.math.Vectorization;
import dev.neuronic.net.math.Parallelization;

import java.util.concurrent.ExecutorService;

/**
 * Softmax activation function for multi-class classification.
 * 
 * <p>Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)) for all j)
 * 
 * <p><strong>What Softmax Does:</strong>
 * <ul>
 * <li>Converts raw logits into probability distribution</li>
 * <li>All outputs sum to 1.0 (probability constraint)</li>
 * <li>Amplifies differences between inputs (exponential)</li>
 * <li>Always produces positive outputs in range [0, 1]</li>
 * </ul>
 * 
 * <p><strong>When to Use Softmax:</strong>
 * <ul>
 * <li><strong>Multi-class classification</strong> (MNIST, CIFAR, ImageNet)</li>
 * <li><strong>Output layer only</strong> - final layer of classification networks</li>
 * <li><strong>Mutually exclusive classes</strong> - one true class per sample</li>
 * <li><strong>With CrossEntropy loss</strong> - optimal pairing for classification</li>
 * </ul>
 * 
 * <p><strong>When NOT to Use Softmax:</strong>
 * <ul>
 * <li><strong>Regression tasks</strong> - use linear/no activation</li>
 * <li><strong>Binary classification</strong> - use Sigmoid instead</li>
 * <li><strong>Multi-label classification</strong> - use Sigmoid for independent probabilities</li>
 * <li><strong>Hidden layers</strong> - use ReLU/Tanh instead</li>
 * </ul>
 * 
 * <p><strong>Example:</strong>
 * <pre>
 * Input:  [2.0, 1.0, 3.0]
 * Output: [0.245, 0.090, 0.665]  // Probabilities summing to 1.0
 * Prediction: argmax = class 2 (highest probability)
 * </pre>
 * 
 * <p><strong>Numerical Stability:</strong>
 * Uses the "max trick" to prevent overflow: subtracts max(x) from all inputs
 * before computing exponentials.
 */
public final class SoftmaxActivator implements Activator {
    
    public static final SoftmaxActivator INSTANCE = new SoftmaxActivator();
    
    private SoftmaxActivator() {} // Private constructor for singleton
    
    @Override
    public void activate(float[] input, float[] output) {
        checkLength(input, output);
        
        if (Vectorization.shouldVectorize(input.length))
            activateVectorized(input, output);
        else
            activateScalar(input, output);
    }
    
    @Override
    public void derivative(float[] input, float[] output) {
        checkLength(input, output);
        
        // Softmax derivative is complex: output[i] * (1 - output[i]) for diagonal
        // and -output[i] * output[j] for off-diagonal elements
        // This simplified version assumes we're computing the derivative after
        // softmax has already been applied (input is already softmax output)
        
        if (Vectorization.shouldVectorize(input.length))
            derivativeVectorized(input, output);
        else
            derivativeScalar(input, output);
    }
    
    private static void checkLength(float[] input, float[] output) {
        if (input.length != output.length)
            throw new IllegalArgumentException(
                "Input and output arrays must have same length: " +
                "input=" + input.length + ", output=" + output.length
            );
    }
    
    void activateVectorized(float[] input, float[] output) {
        // Delegate to scalar implementation for now
        activateScalar(input, output);
    }
    
    void activateScalar(float[] input, float[] output) {
        // Step 1: Find maximum for numerical stability
        float maxVal = input[0];
        for (int i = 1; i < input.length; i++) {
            if (input[i] > maxVal)
                maxVal = input[i];
        }
        
        // Step 2: Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i] - maxVal);
            sum += output[i];
        }
        
        // Step 3: Normalize
        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }
    }
    
    void derivativeVectorized(float[] input, float[] output) {
        // Delegate to scalar implementation for now
        derivativeScalar(input, output);
    }
    
    void derivativeScalar(float[] input, float[] output) {
        // Simplified derivative: softmax_i * (1 - softmax_i)
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] * (1.0f - input[i]);
        }
    }
    
    private float findMaxVectorized(float[] array) {
        // Delegate to scalar implementation for now
        return findMaxScalar(array);
    }
    
    private float findMaxScalar(float[] array) {
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max)
                max = array[i];
        }
        return max;
    }
    
    @Override
    public void activate(float[] input, float[] output, ExecutorService executor) {
        // Note: Softmax requires global operations (max, sum) that don't parallelize well
        // For small arrays, the overhead of coordination would exceed benefits
        // We still provide the interface for consistency but use sequential implementation
        activate(input, output);
    }
    
    @Override
    public void derivative(float[] input, float[] output, ExecutorService executor) {
        // Softmax derivative can be parallelized since it's element-wise after softmax is computed
        if (Parallelization.shouldParallelize(input.length, executor))
            derivativeParallel(input, output, executor);
        else
            derivative(input, output);
    }
    
    private void derivativeParallel(float[] input, float[] output, ExecutorService executor) {
        checkLength(input, output);
        
        int numThreads = Parallelization.calculateOptimalThreads(input.length, executor);
        Parallelization.WorkRange[] ranges = Parallelization.splitWork(input.length, numThreads);
        
        Runnable[] tasks = new Runnable[ranges.length];
        for (int i = 0; i < ranges.length; i++) {
            final Parallelization.WorkRange range = ranges[i];
            tasks[i] = () -> {
                if (Vectorization.shouldVectorize(range.size))
                    derivativeVectorizedRange(input, output, range.start, range.end);
                else
                    derivativeScalarRange(input, output, range.start, range.end);
            };
        }
        
        Parallelization.executeParallel(executor, tasks);
    }
    
    private void derivativeVectorizedRange(float[] input, float[] output, int start, int end) {
        // Delegate to scalar implementation for now
        derivativeScalarRange(input, output, start, end);
    }
    
    private void derivativeScalarRange(float[] input, float[] output, int start, int end) {
        for (int i = start; i < end; i++) {
            output[i] = input[i] * (1.0f - input[i]);
        }
    }
}