package dev.neuronic.net.optimizers;

import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Stochastic Gradient Descent optimizer with lock-free parallel training support.
 * 
 * <p>Updates parameters using: param = param - learning_rate * gradient
 * 
 * <p>SGD is the fundamental optimization algorithm that forms the basis
 * for most other optimizers. Simple and effective for many tasks.
 * 
 * <h3>Parallel Training - Hogwild! Approach</h3>
 * This implementation uses the "Hogwild!" algorithm for lock-free parallel training:
 * <ul>
 *   <li><b>Race conditions are allowed</b> on weight updates for maximum parallelism</li>
 *   <li><b>Mathematically proven</b> to converge for SGD when gradients are sparse</li>
 *   <li><b>Excellent performance</b> due to zero synchronization overhead</li>
 *   <li><b>Minimal interference</b> because neural network gradients are typically sparse</li>
 * </ul>
 * 
 * <p><b>Key insight:</b> The benefits of lock-free parallelism far outweigh the minimal
 * interference from race conditions on individual weight updates.
 * 
 * <p><b>Reference:</b> "Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent"
 * by Recht et al. (2011)
 */
public class SgdOptimizer implements Optimizer, Serializable {
    
    private volatile float learningRate; // Made volatile for thread-safe updates
    
    /**
     * Create SGD optimizer with specified learning rate.
     * 
     * @param learningRate the learning rate (typically 0.001 to 0.1)
     */
    public SgdOptimizer(float learningRate) {
        if (learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be positive: " + learningRate);
        this.learningRate = learningRate;
    }
    
    @Override
    public void optimize(float[][] weights, float[] biases, float[][] weightGradients, float[] biasGradients) {
        if (weights.length != weightGradients.length)
            throw new IllegalArgumentException("Weight and gradient arrays must have same outer dimension");
        if (biases.length != biasGradients.length)
            throw new IllegalArgumentException("Bias and gradient arrays must have same length");

        // Hogwild! lock-free weight updates - allows race conditions for maximum parallelism
        // Use vectorized updates for better memory throughput
        for (int i = 0; i < weights.length; i++) {
            NetMath.parameterUpdate(weights[i], weightGradients[i], learningRate);
        }

        NetMath.parameterUpdate(biases, biasGradients, learningRate);
    }
    
    /**
     * @return the learning rate for this optimizer
     */
    public float getLearningRate() {
        return learningRate;
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        out.writeFloat(learningRate);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use readFrom(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize an SgdOptimizer from stream.
     */
    public static SgdOptimizer deserialize(DataInputStream in, int version) throws IOException {
        float learningRate = in.readFloat();
        return new SgdOptimizer(learningRate);
    }
    
    @Override
    public int getSerializedSize(int version) {
        return 4; // float learning rate
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_SGD_OPTIMIZER;
    }
    
    @Override
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }
    
    @Override
    public void optimize(float[] parameters, float[] gradients) {
        if (parameters.length != gradients.length)
            throw new IllegalArgumentException("Parameter and gradient arrays must have same length");
        
        // Hogwild! lock-free parameter updates - same approach as 2D
        NetMath.parameterUpdate(parameters, gradients, learningRate);
    }

    @Override
    public void sparseOptimize(Object stateKey, float[][] allWeights, int[] indicesToUpdate,
                               float[][] gradients, java.util.concurrent.ExecutorService executor) {
        if (indicesToUpdate.length != gradients.length) {
            throw new IllegalArgumentException(String.format(
                "Mismatched inputs for sparse update: %d indices but %d gradients.",
                indicesToUpdate.length, gradients.length));
        }
        if (indicesToUpdate.length == 0) {
            return; // Nothing to do
        }

        // SGD is stateless, so we can just iterate and apply the updates.
        // No need for a stateKey or complex state management.
        for (int i = 0; i < indicesToUpdate.length; i++) {
            int weightIndex = indicesToUpdate[i];
            float[] gradient = gradients[i];

            if (weightIndex < 0 || weightIndex >= allWeights.length) {
                 System.err.printf("Optimizer Warning: Index %d is out of bounds for weights (len=%d). Skipping.\n",
                                  weightIndex, allWeights.length);
                continue;
            }

            NetMath.parameterUpdate(allWeights[weightIndex], gradient, learningRate);
        }
    }
}